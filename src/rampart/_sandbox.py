"""Subprocess sandboxing for ``@node(sandbox=True)`` nodes.

Layer 3 of Rampart's three-layer isolation model:

  Layer 1 — Python library monkey-patching (_http_intercept.py)
              Intercepts httpx / requests at the Python level.
  Layer 2 — HTTP proxy injection (_config.py ``http_proxy_port``)
              Routes all agent traffic through a configurable local proxy.
  Layer 3 — Subprocess execution (this module)
              Runs the node function in an isolated child process, giving
              crash isolation (OOM / SIGKILL in the node cannot take down
              the orchestrator) and optional OS-level resource limits.

**Limitation**: sandbox=True is only supported for state-only nodes (nodes that
don't declare ``tools``, ``llm``, ``graphs``, or ``artifacts`` parameters).
Those injected contexts are bound to the parent's RunContext and cannot be
serialized across a process boundary.  Nodes that need tool/LLM access with
sandbox isolation should delegate to sub-graphs running in separate processes.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

# Module-level process pool (lazy initialized, shared across all sandboxed runs).
_pool: concurrent.futures.ProcessPoolExecutor | None = None


def _get_pool() -> concurrent.futures.ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)
    return _pool


def _run_node_in_subprocess(
    fn: Any,
    state_dict: dict[str, Any],
    state_class: type,
    max_memory_mb: int | None,
    max_cpu_seconds: int | None,
) -> dict[str, Any]:
    """Executed inside the child process.

    1. Applies OS resource limits (Linux / macOS only; silently skipped elsewhere).
    2. Reconstructs the AgentState from its dict representation.
    3. Runs the async node function via ``asyncio.run()``.
    4. Returns the result as a plain dict for pickling back to the parent.
    """
    import asyncio as _asyncio
    import dataclasses

    # ── Resource limits ──────────────────────────────────────────────────────
    if max_memory_mb is not None or max_cpu_seconds is not None:
        try:
            import resource as _res  # Unix only

            if max_memory_mb is not None:
                mem_bytes = max_memory_mb * 1024 * 1024
                _res.setrlimit(_res.RLIMIT_AS, (mem_bytes, mem_bytes))
            if max_cpu_seconds is not None:
                _res.setrlimit(_res.RLIMIT_CPU, (max_cpu_seconds, max_cpu_seconds))
        except (ImportError, OSError, ValueError):
            pass  # Not available on this platform; continue without limits

    # ── Reconstruct state ────────────────────────────────────────────────────
    known = {f.name for f in dataclasses.fields(state_class)}
    state = state_class(**{k: v for k, v in state_dict.items() if k in known})

    # ── Execute ──────────────────────────────────────────────────────────────
    result = _asyncio.run(fn(state))
    return dataclasses.asdict(result)


async def run_sandboxed(
    fn: Any,
    state: Any,
    state_class: type,
    max_memory_mb: int | None = 512,
    max_cpu_seconds: int | None = None,
) -> Any:
    """Run an async node function in an isolated subprocess.

    Args:
        fn: The node's original async function (must be picklable — true for
            all module-level ``@node`` decorated functions).
        state: The current ``AgentState`` instance to pass to the function.
        state_class: The concrete ``AgentState`` subclass, used to reconstruct
            the result dict back into a typed instance.
        max_memory_mb: Virtual-memory ceiling for the subprocess in megabytes.
            Default 512 MB.  ``None`` disables the limit.
        max_cpu_seconds: CPU-time ceiling in seconds.  ``None`` (default)
            disables the limit.

    Returns:
        A reconstructed ``AgentState`` of the same type as *state*.

    Raises:
        ``concurrent.futures.ProcessPoolExecutor`` propagates any exception
        raised inside the subprocess as a ``concurrent.futures.process.RemoteTraceback``.
    """
    import dataclasses

    state_dict = dataclasses.asdict(state)
    loop = asyncio.get_event_loop()
    pool = _get_pool()

    result_dict: dict[str, Any] = await loop.run_in_executor(
        pool,
        _run_node_in_subprocess,
        fn,
        state_dict,
        state_class,
        max_memory_mb,
        max_cpu_seconds,
    )

    known = {f.name for f in dataclasses.fields(state_class)}
    return state_class(**{k: v for k, v in result_dict.items() if k in known})

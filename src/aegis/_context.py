"""RunContext, ToolContext, LLMContext, GraphContext, and the active-run contextvar."""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from ._models import (
    AgentState,
    BudgetExceededError,
    BudgetStatus,
    LLMCall,
    LLMNotConfiguredError,
    NodeTrace,
    PermissionScope,
    RunTrace,
    ToolCall,
)

if TYPE_CHECKING:
    from ._models import Budget, CassetteRecord, MockContext
    from .checkpointers._base import CheckpointerBase
    from .testing._mock_tools import MockTool


# ── Active run contextvar ─────────────────────────────────────────────────────

_run_context: ContextVar[Optional["RunContext"]] = ContextVar(
    "aegis_run_context", default=None
)


# ── RunContext ────────────────────────────────────────────────────────────────

@dataclass
class RunContext:
    """All mutable state for a single graph run."""

    run_id: str
    thread_id: str
    graph_name: str
    graph_version: str
    checkpointer: "CheckpointerBase"
    trace: RunTrace
    permission_scope: Optional[PermissionScope]
    budget: Optional["Budget"]

    # Tool registry: tool_name -> ToolDef
    tool_registry: dict[str, Any] = field(default_factory=dict)

    # Mock overrides: tool_name -> MockTool (set via mock_tools() context)
    mock_tools: Optional[dict[str, "MockTool"]] = None
    mock_ctx: Optional["MockContext"] = None

    # Cassette state
    cassette_mode: Optional[str] = None          # "record" | "replay" | None
    cassette: Optional["CassetteRecord"] = None
    cassette_index: int = 0                       # next entry to serve in replay
    cassette_override_tools: Optional[dict[str, "MockTool"]] = None

    # Step tracking
    _step_counter: int = 0
    _fast_forward_to_step: int = -1              # replay through this step
    _existing_checkpoints: dict[int, Any] = field(default_factory=dict)
    last_checkpoint_id: Optional[str] = None

    # Budget tracking (mutable)
    _budget_start_time: float = field(default_factory=time.monotonic)
    _budget_tokens_used: int = 0
    _budget_cost_usd: float = 0.0
    _budget_tool_calls: int = 0
    _budget_exceeded_handler: Optional[Any] = None  # async callable

    # Current node context
    current_node_name: Optional[str] = None
    current_node_trace: Optional[NodeTrace] = None

    # LLM model override (for downgrade_model budget policy)
    llm_model_override: Optional[str] = None

    def next_step(self) -> int:
        step = self._step_counter
        self._step_counter += 1
        return step

    def budget_status(self) -> BudgetStatus:
        elapsed = time.monotonic() - self._budget_start_time
        status = BudgetStatus(
            tokens_used=self._budget_tokens_used,
            cost_usd=self._budget_cost_usd,
            tool_calls_made=self._budget_tool_calls,
            wall_time_seconds=elapsed,
        )
        if self.budget:
            status.compute_pct(self.budget)
        return status

    def check_budget(self, checkpoint_id: str = "") -> None:
        """Raise BudgetExceededError if any budget dimension is exceeded."""
        if not self.budget:
            return
        status = self.budget_status()
        exceeded: Optional[str] = None

        if self.budget.max_tokens is not None and status.tokens_used > self.budget.max_tokens:
            exceeded = "tokens"
        elif self.budget.max_llm_cost_usd is not None and status.cost_usd > self.budget.max_llm_cost_usd:
            exceeded = "cost"
        elif self.budget.max_tool_calls is not None and status.tool_calls_made > self.budget.max_tool_calls:
            exceeded = "tool_calls"
        elif self.budget.max_wall_time_seconds is not None:
            elapsed = time.monotonic() - self._budget_start_time
            if elapsed > self.budget.max_wall_time_seconds:
                exceeded = "wall_time"

        if exceeded:
            from ._models import BudgetExceededEvent
            event = BudgetExceededEvent(
                run_id=self.run_id,
                thread_id=self.thread_id,
                exceeded_dimension=exceeded,
                budget=self.budget,
                current_status=status,
                checkpoint_id=checkpoint_id or self.last_checkpoint_id or "",
            )
            raise BudgetExceededError(event)

    async def execute_tool(self, tool_name: str, kwargs: dict[str, Any]) -> Any:
        """Dispatch a tool call through permission checks, mocks, cassette, and real execution."""
        import time as _time
        from datetime import datetime

        start = _time.monotonic()
        node_name = self.current_node_name or "unknown"
        call_id = str(uuid.uuid4())[:8]

        # Track budget for ALL tool calls (mocked or real) before dispatch
        self._budget_tool_calls += 1
        self.check_budget(self.last_checkpoint_id or "")

        # 1. Cassette override_tools take highest priority
        if self.cassette_override_tools and tool_name in self.cassette_override_tools:
            mock = self.cassette_override_tools[tool_name]
            return await self._execute_mock_tool(
                tool_name, kwargs, mock, call_id, node_name, start
            )

        # 2. mock_tools() context
        if self.mock_tools and tool_name in self.mock_tools:
            mock = self.mock_tools[tool_name]
            return await self._execute_mock_tool(
                tool_name, kwargs, mock, call_id, node_name, start, record_in_mock_ctx=True
            )

        # 3. Cassette replay
        if self.cassette_mode == "replay" and self.cassette:
            return await self._serve_tool_from_cassette(tool_name, kwargs, call_id, node_name, start)

        # 4. Permission check
        self._check_tool_permission(tool_name, node_name)

        # 5. Real tool execution
        tool_def = self.tool_registry.get(tool_name)
        if tool_def is None:
            raise KeyError(f"Tool '{tool_name}' not found in registry")

        error: Optional[str] = None
        result: Any = None
        try:
            result = await tool_def.fn(**kwargs)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            latency_ms = int((_time.monotonic() - start) * 1000)
            tc = ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                args=kwargs,
                result=result,
                error=error,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow(),
                node_name=node_name,
                permission_checked=True,
                permission_granted=True,
            )
            if self.current_node_trace:
                self.current_node_trace.tool_calls.append(tc)

            # Record in cassette if recording
            if self.cassette_mode == "record" and self.cassette is not None:
                from ._models import CassetteEntry
                entry = CassetteEntry(
                    type="tool_call",
                    call_id=call_id,
                    step=self._step_counter - 1,
                    node_name=node_name,
                    request={"tool_name": tool_name, "args": _safe_serialize(kwargs)},
                    response={"result": _safe_serialize(result), "error": error},
                    timestamp=datetime.utcnow(),
                )
                self.cassette.entries.append(entry)

        return result

    async def _execute_mock_tool(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        mock: "MockTool",
        call_id: str,
        node_name: str,
        start: float,
        record_in_mock_ctx: bool = False,
    ) -> Any:
        import time as _time
        from datetime import datetime

        error: Optional[str] = None
        result: Any = None
        try:
            result = await mock.execute(kwargs)
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            latency_ms = int((_time.monotonic() - start) * 1000)
            tc = ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                args=kwargs,
                result=result,
                error=error,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow(),
                node_name=node_name,
                was_mocked=True,
                permission_checked=False,
                permission_granted=True,
            )
            if self.current_node_trace:
                self.current_node_trace.tool_calls.append(tc)

            if record_in_mock_ctx and self.mock_ctx is not None:
                from ._models import MockCallRecord
                if tool_name not in self.mock_ctx.calls:
                    self.mock_ctx.calls[tool_name] = MockCallRecord(tool_name=tool_name)
                rec = self.mock_ctx.calls[tool_name]
                rec.count += 1
                rec.calls.append(tc)

            # Record mock result in cassette if we're in record mode
            if self.cassette_mode == "record" and self.cassette is not None:
                from ._models import CassetteEntry
                cassette_entry = CassetteEntry(
                    type="tool_call",
                    call_id=call_id,
                    step=self._step_counter - 1,
                    node_name=node_name,
                    request={"tool_name": tool_name, "args": _safe_serialize(kwargs)},
                    response={"result": _safe_serialize(result), "error": error},
                    timestamp=datetime.utcnow(),
                )
                self.cassette.entries.append(cassette_entry)

    async def _serve_tool_from_cassette(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        call_id: str,
        node_name: str,
        start: float,
    ) -> Any:
        import time as _time
        from datetime import datetime
        from ._models import AegisCassetteStaleError

        assert self.cassette is not None
        # Find next tool_call entry in cassette
        entries = self.cassette.entries
        idx = self.cassette_index
        while idx < len(entries) and entries[idx].type != "tool_call":
            idx += 1
        if idx >= len(entries):
            raise AegisCassetteStaleError(
                f"Cassette exhausted: no more tool calls recorded "
                f"(tool='{tool_name}', step={self._step_counter})"
            )

        entry = entries[idx]
        self.cassette_index = idx + 1

        # Validate tool name match
        recorded_tool = entry.request.get("tool_name")
        if recorded_tool != tool_name:
            raise AegisCassetteStaleError(
                f"Cassette divergence at index {idx}: "
                f"expected tool '{recorded_tool}', got '{tool_name}'"
            )

        if entry.response.get("error"):
            raise RuntimeError(entry.response["error"])

        result = entry.response.get("result")
        latency_ms = int((_time.monotonic() - start) * 1000)
        tc = ToolCall(
            call_id=call_id,
            tool_name=tool_name,
            args=kwargs,
            result=result,
            error=None,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
            node_name=node_name,
            was_mocked=False,
        )
        if self.current_node_trace:
            self.current_node_trace.tool_calls.append(tc)
        return result

    def _check_tool_permission(self, tool_name: str, node_name: str) -> None:
        if not self.permission_scope:
            return
        scope = self.permission_scope
        if scope.tools is not None and tool_name not in scope.tools:
            from datetime import datetime
            from ._models import PermissionViolationEvent, PermissionDeniedError
            event = PermissionViolationEvent(
                run_id=self.run_id,
                thread_id=self.thread_id,
                node_name=node_name,
                violation_type="tool_not_in_whitelist",
                attempted_action=f"call tool '{tool_name}'",
                declared_scope=scope,
                timestamp=datetime.utcnow(),
            )
            raise PermissionDeniedError(event)

    async def execute_llm_call(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> "LLMResponse":
        """Execute an LLM call, serving from cassette if in replay mode."""
        import time as _time
        from datetime import datetime

        # Apply model override from budget downgrade
        effective_model = self.llm_model_override or model
        node_name = self.current_node_name or "unknown"
        call_id = str(uuid.uuid4())[:8]
        start = _time.monotonic()

        # Cassette replay
        if self.cassette_mode == "replay" and self.cassette:
            return await self._serve_llm_from_cassette(
                effective_model, prompt, system, call_id, node_name, start
            )

        # Live LLM call — try litellm
        try:
            import litellm  # type: ignore[import]
        except ImportError:
            raise LLMNotConfiguredError(
                "No LLM provider configured. Install aegis[litellm] and set your API key, "
                "or use cassette replay / mock_tools() for testing."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await litellm.acompletion(model=effective_model, messages=messages, **kwargs)
        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = getattr(response.usage, "cache_read_input_tokens", 0)
        cost_usd = getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0

        latency_ms = int((_time.monotonic() - start) * 1000)
        llm_call = LLMCall(
            call_id=call_id,
            model=effective_model,
            system_prompt=system,
            user_prompt=prompt,
            response=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
            node_name=node_name,
        )

        self._record_llm_call(llm_call)

        # Record in cassette if recording
        if self.cassette_mode == "record" and self.cassette is not None:
            from ._models import CassetteEntry
            entry = CassetteEntry(
                type="llm_call",
                call_id=call_id,
                step=self._step_counter - 1,
                node_name=node_name,
                request=_safe_serialize(
                    {"model": effective_model, "system": system, "prompt": prompt, **kwargs}
                ),
                response=_safe_serialize(
                    {"text": text, "input_tokens": input_tokens,
                     "output_tokens": output_tokens, "cost_usd": cost_usd}
                ),
                timestamp=datetime.utcnow(),
            )
            self.cassette.entries.append(entry)

        return LLMResponse(text=text, call=llm_call)

    async def _serve_llm_from_cassette(
        self,
        model: str,
        prompt: str,
        system: Optional[str],
        call_id: str,
        node_name: str,
        start: float,
    ) -> "LLMResponse":
        import time as _time
        from datetime import datetime
        from ._models import AegisCassetteStaleError, CassetteEntry

        assert self.cassette is not None
        entries = self.cassette.entries
        idx = self.cassette_index
        while idx < len(entries) and entries[idx].type != "llm_call":
            idx += 1
        if idx >= len(entries):
            raise AegisCassetteStaleError(
                f"Cassette exhausted: no more LLM calls recorded (node={node_name})"
            )

        entry = entries[idx]
        self.cassette_index = idx + 1

        # Stale detection: compare prompt hash
        import hashlib
        current_req = {"model": model, "system": system, "prompt": prompt}
        current_hash = hashlib.sha256(
            str(sorted(current_req.items())).encode()
        ).hexdigest()[:16]
        recorded_hash = hashlib.sha256(
            str(sorted(entry.request.items())).encode()
        ).hexdigest()[:16]

        if current_hash != recorded_hash:
            raise AegisCassetteStaleError(
                f"Cassette '{node_name}' is stale at index {idx} (llm_call):\n"
                f"  prompt hash changed\n"
                f"  Recorded:  sha256:{recorded_hash}\n"
                f"  Current:   sha256:{current_hash}\n\n"
                f"Re-record with: pytest --aegis-record"
            )

        resp = entry.response
        latency_ms = int((_time.monotonic() - start) * 1000)
        llm_call = LLMCall(
            call_id=call_id,
            model=model,
            system_prompt=system,
            user_prompt=prompt,
            response=resp.get("text", ""),
            input_tokens=resp.get("input_tokens", 0),
            output_tokens=resp.get("output_tokens", 0),
            cached_tokens=0,
            cost_usd=resp.get("cost_usd", 0.0),
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
            node_name=node_name,
            was_replayed=True,
        )
        self._record_llm_call(llm_call)
        return LLMResponse(text=resp.get("text", ""), call=llm_call)

    def _record_llm_call(self, llm_call: LLMCall) -> None:
        self._budget_tokens_used += llm_call.input_tokens + llm_call.output_tokens
        self._budget_cost_usd += llm_call.cost_usd
        self.trace.add_llm_call(llm_call)
        if self.current_node_trace:
            self.current_node_trace.llm_calls.append(llm_call)
        self.check_budget(self.last_checkpoint_id or "")


# ── LLM Response ──────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    text: str
    call: LLMCall


# ── ToolContext ───────────────────────────────────────────────────────────────

class ToolContext:
    """Injected into node functions via 'tools' parameter.

    Attribute access returns an async callable that dispatches to the named tool.
    """

    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        async def _dispatch(**kwargs: Any) -> Any:
            return await self._ctx.execute_tool(name, kwargs)

        _dispatch.__name__ = name
        return _dispatch


# ── LLMContext ────────────────────────────────────────────────────────────────

class LLMContext:
    """Injected into node functions via 'llm' parameter."""

    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx

    async def complete(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self._ctx.execute_llm_call(
            model=model, prompt=prompt, system=system, **kwargs
        )


# ── GraphContext ──────────────────────────────────────────────────────────────

class GraphContext:
    """Injected into node functions via 'graphs' parameter.

    Allows calling sub-graphs from within a node with automatic OTel propagation.
    """

    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        # Return a proxy that delegates .run() etc. to the named graph
        from ._decorators import _GRAPH_REGISTRY

        graph_def = _GRAPH_REGISTRY.get(name)
        if graph_def is None:
            raise KeyError(f"Graph '{name}' not found in registry")

        class _SubGraphProxy:
            async def run(proxy_self, input: AgentState, config: "RunConfig", **kw: Any) -> Any:  # noqa: N805
                from ._models import RunConfig as _RC
                # Propagate parent run_id for trace nesting
                return await graph_def.run(input=input, config=config, **kw)

        return _SubGraphProxy()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_serialize(obj: Any) -> Any:
    """Best-effort JSON-safe serialization."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    try:
        import dataclasses as dc
        if dc.is_dataclass(obj):
            return dc.asdict(obj)
    except Exception:
        pass
    return str(obj)

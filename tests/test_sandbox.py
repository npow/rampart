"""Tests for subprocess sandboxing (@node sandbox=True)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import pytest

import rampart


# ── State types ────────────────────────────────────────────────────────────────


@dataclass
class SbState(rampart.AgentState):
    x: int = 0
    label: str = ""


# ── Module-level sandbox nodes (must be at module level for pickling) ──────────


@rampart.node(sandbox=True)
async def _sb_double(state: SbState) -> SbState:
    return state.update(x=state.x * 2)


@rampart.node(sandbox=True)
async def _sb_label(state: SbState) -> SbState:
    return state.update(label=f"sandboxed:{state.x}")


@rampart.node(sandbox=True, retries=0)
async def _sb_crash(state: SbState) -> SbState:
    raise ValueError("subprocess boom")


@rampart.node(sandbox=True)
async def _sb_with_tools(state: SbState, tools: rampart.ToolContext) -> SbState:
    """Sandbox=True but declares tools — should fall back to in-process with warning."""
    return state.update(x=99)


@rampart.graph(name="_sb_graph_double")
async def _sb_graph_double(state: SbState) -> SbState:
    return await _sb_double(state)


@rampart.graph(name="_sb_graph_label")
async def _sb_graph_label(state: SbState) -> SbState:
    return await _sb_label(state)


@rampart.graph(name="_sb_graph_crash")
async def _sb_graph_crash(state: SbState) -> SbState:
    return await _sb_crash(state)


@rampart.graph(name="_sb_graph_with_tools")
async def _sb_graph_with_tools(state: SbState) -> SbState:
    return await _sb_with_tools(state)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sandbox_basic_execution():
    """A sandboxed node runs in subprocess and returns correct state."""
    cfg = rampart.RunConfig(thread_id="sb-basic")
    result = await _sb_graph_double.run(SbState(x=7), cfg)
    assert result.status == "completed"
    assert result.state.x == 14


@pytest.mark.asyncio
async def test_sandbox_string_field():
    cfg = rampart.RunConfig(thread_id="sb-label")
    result = await _sb_graph_label.run(SbState(x=5), cfg)
    assert result.state.label == "sandboxed:5"


@pytest.mark.asyncio
async def test_sandbox_exception_propagates():
    """Exceptions raised in the subprocess surface as run failures."""
    cfg = rampart.RunConfig(thread_id="sb-crash")
    result = await _sb_graph_crash.run(SbState(), cfg)
    assert result.status == "failed"
    assert result.error is not None
    assert "boom" in result.error.message


@pytest.mark.asyncio
async def test_sandbox_warns_and_falls_back_with_injected_context():
    """sandbox=True + tools parameter: warn and run in-process."""
    cfg = rampart.RunConfig(thread_id="sb-tools")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await _sb_graph_with_tools.run(SbState(x=0), cfg)

    assert result.state.x == 99  # ran in-process
    warn_msgs = [str(w.message) for w in caught]
    assert any("sandbox" in m.lower() for m in warn_msgs)

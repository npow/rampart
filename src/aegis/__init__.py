"""Aegis — A runtime that makes LLM agents production-safe by default."""

from __future__ import annotations

# Install HTTP intercept at import time (no-op if httpx/requests not installed)
from ._http_intercept import install as _install_intercept
_install_intercept()

# ── Public API ────────────────────────────────────────────────────────────────

from ._models import (
    # State
    AgentState,
    RunConfig,
    RunResult,
    RunError,
    # Checkpointing
    Checkpoint,
    CheckpointBackendConfig,
    # Tracing
    LLMCall,
    ToolCall,
    NodeTrace,
    RunTrace,
    # Budget
    Budget,
    BudgetStatus,
    BudgetExceededEvent,
    BudgetDecision,
    # Permissions
    NetworkPermission,
    FilesystemPermission,
    ApprovalPolicy,
    PermissionScope,
    PermissionViolationEvent,
    # Testing models
    CassetteRecord,
    CassetteEntry,
    MockCallRecord,
    MockContext,
    CassetteReplayContext,
    # Eval models
    EvalAssertion,
    ToolCallAssertion,
    SchemaAssertion,
    TraceSnapshotAssertion,
    EvalCase,
    EvalCaseResult,
    EvalSuiteResult,
    # Exceptions
    AegisError,
    BudgetExceededError,
    PermissionDeniedError,
    AegisCassetteStaleError,
    EvalGateFailure,
    GraphVersionConflict,
    NoCheckpointError,
    LLMNotConfiguredError,
)

from ._decorators import (
    graph,
    node,
    tool,
    GraphDef,
    NodeDef,
    ToolDef,
    get_tool_registry,
    get_graph_registry,
)

from ._context import (
    ToolContext,
    LLMContext,
    GraphContext,
    LLMResponse,
)

from ._multi_agent import chain, parallel, supervisor

from .checkpointers import (
    CheckpointerBase,
    MemoryCheckpointer,
    SqliteCheckpointer,
)

# Configuration helper (global defaults)
from ._config import configure, PostgresCheckpointer, OTelTracer


__all__ = [
    # State
    "AgentState",
    "RunConfig",
    "RunResult",
    "RunError",
    # Decorators
    "graph",
    "node",
    "tool",
    "GraphDef",
    "NodeDef",
    "ToolDef",
    "get_tool_registry",
    "get_graph_registry",
    # Context
    "ToolContext",
    "LLMContext",
    "GraphContext",
    "LLMResponse",
    # Checkpointing
    "Checkpoint",
    "CheckpointBackendConfig",
    "CheckpointerBase",
    "MemoryCheckpointer",
    "SqliteCheckpointer",
    # Tracing
    "LLMCall",
    "ToolCall",
    "NodeTrace",
    "RunTrace",
    # Budget
    "Budget",
    "BudgetStatus",
    "BudgetExceededEvent",
    "BudgetDecision",
    # Permissions
    "NetworkPermission",
    "FilesystemPermission",
    "ApprovalPolicy",
    "PermissionScope",
    "PermissionViolationEvent",
    # Testing
    "CassetteRecord",
    "CassetteEntry",
    "MockCallRecord",
    "MockContext",
    "CassetteReplayContext",
    # Eval
    "EvalAssertion",
    "ToolCallAssertion",
    "SchemaAssertion",
    "TraceSnapshotAssertion",
    "EvalCase",
    "EvalCaseResult",
    "EvalSuiteResult",
    # Multi-agent
    "chain",
    "parallel",
    "supervisor",
    # Configuration
    "configure",
    "PostgresCheckpointer",
    "OTelTracer",
    # Exceptions
    "AegisError",
    "BudgetExceededError",
    "PermissionDeniedError",
    "AegisCassetteStaleError",
    "EvalGateFailure",
    "GraphVersionConflict",
    "NoCheckpointError",
    "LLMNotConfiguredError",
]

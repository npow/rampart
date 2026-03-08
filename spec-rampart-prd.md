# Aegis — Product Requirements Document

**Status:** Draft v1.0
**Last Updated:** 2026-02-27
**Tagline:** "A runtime that makes LLM agents production-safe by default."
**Positioning:** The framework built for teams who have gotten burned shipping LangGraph to production.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Goals](#2-goals)
3. [Non-Goals](#3-non-goals)
4. [Target Users](#4-target-users)
5. [Core Concepts](#5-core-concepts)
6. [Differentiators](#6-differentiators)
7. [API / Interface](#7-api--interface)
8. [Data Model](#8-data-model)
9. [Architecture](#9-architecture)
10. [Deployment Model](#10-deployment-model)
11. [Failure Modes](#11-failure-modes)
12. [Success Metrics](#12-success-metrics)
13. [Open Questions](#13-open-questions)

---

## 1. Problem Statement

Teams building production agentic systems face a consistent failure pattern: prototypes work in demos but collapse under real-world conditions. Process crashes lose in-flight state entirely. LLM non-determinism makes bugs impossible to reproduce — you can't re-run a failed test and get the same behavior. Runaway agents exceed API budgets with no circuit-breaker to stop them. The Replit incident (July 2025) — where an agent deleted a production database containing 1,200+ records despite an explicit code-and-action freeze — was not an edge case; it was the predictable result of a framework with no permission enforcement model.

Existing frameworks (LangGraph, CrewAI, AutoGen) optimize for developer ergonomics in controlled environments, then leave production concerns — durability, testability, governance, cost control — as exercises for the team. Every team shipping agents in 2025 has independently assembled some combination of Temporal + LangGraph + Langfuse + unittest.mock, paying the integration tax on each project. The result: only 11% of enterprises had production-ready agentic systems by mid-2025, and Gartner projects 40%+ project cancellation by 2027.

Aegis closes the gap between "it works in dev" and "it runs safely in prod" by treating production readiness as the default, not the graduation requirement. It is the first agentic framework where durable execution, deterministic testing, enforced budgets, and permission scopes ship in the box.

---

## 2. Goals

1. An agent that crashes mid-run resumes from the last committed node within 5 seconds, without data loss and without re-executing completed steps.
2. An agent survives a full process restart or rolling deployment without manual intervention or state reconstruction.
3. 100% of agent test suites run without calling live LLM or tool APIs — tool mocks and cassette replay are first-class primitives, not third-party add-ons.
4. A CI eval pipeline for an agent with 10 test cases completes in under 60 seconds using cassette replay.
5. Budget overruns are architecturally impossible — a declared budget envelope is a hard runtime constraint enforced before each node execution, not a soft suggestion checked after the fact.
6. Every tool invocation is checked against a declared permission scope before execution; violations are blocked and logged, never silently allowed.
7. HTTP calls made directly (not through a registered tool) are intercepted and subject to the same permission checks as tool calls — agents cannot escape the permission model by calling `httpx.get()` directly.
8. A multi-agent execution produces a single causal trace tree — sub-agent spans are nested under parent spans automatically, with no manual OTel context propagation.
9. A failing eval case blocks deployment; passing all deterministic cases is a required promotion gate.
10. Framework overhead (excluding LLM/tool latency) is p99 < 20ms per node.

---

## 3. Non-Goals

- **Not a model fine-tuning or training platform.** Aegis orchestrates inference. Training pipelines belong to dedicated ML platforms.
- **Not a visual no-code builder.** The visual debugger is read-only and derived from running code at runtime. There is no drag-and-drop graph editor that generates code.
- **Not an LLM gateway or proxy.** Token routing, model load balancing, semantic caching, and rate limit management are out of scope. Use LiteLLM or OpenRouter as a complement.
- **Not a data pipeline scheduler.** Cron-based DAG scheduling (Airflow/Prefect-style) is out of scope. Aegis is for LLM-driven dynamic workflows, not ETL.
- **Not a replacement for Temporal in non-LLM workflows.** If there is no LLM involved, use Temporal directly.
- **Not a multi-tenant SaaS platform in v1.** The managed deployment target is single-tenant per organization.
- **Not a guaranteed prompt injection detector.** The permission model limits blast radius by enforcing declared scopes regardless of LLM instruction state. It does not claim to detect all injection attacks.
- **Not a RAG platform.** Document ingestion, chunking, and embedding pipelines are out of scope. LlamaIndex or Haystack are the right complements.

---

## 4. Target Users

### Primary — Senior engineers 6–24 months into an agentic AI investment

They built a prototype on LangGraph or CrewAI, shipped it, and hit production problems: a crash lost state mid-run, a runaway agent billed $800 in an afternoon, a prompt change silently broke behavior they didn't catch until a user report, or debugging a failure required replaying logs by hand because there was no way to reproduce it. They know Python well. They do not want to integrate Temporal + LangGraph + Langfuse + unittest.mock on every project. They will pay for a framework that removes that burden.

**What they need from Aegis:** Drop-in replacement for LangGraph with durable execution and a testing story that works in CI without burning API credits.

### Secondary — Platform engineers building internal agent infrastructure

They are building a shared agentic platform for their organization. Multiple product teams will build on top of it. They need governance (permission scopes, audit logs, budget envelopes per team), the ability for product teams to build agents without accidentally deleting production data or exceeding allocated budgets, and a self-hosted deployment model for data sovereignty.

**What they need from Aegis:** Permission enforcement, audit trail, self-hosted deployment, `thread_id`-scoped isolation between workloads.

### Out of scope for v1
Non-technical users, data scientists doing exploratory analysis, teams without Python expertise, and teams with no existing LLM agent investment (they should start with LangGraph to learn the space before adopting Aegis).

---

## 5. Core Concepts

### Graph
The top-level unit of execution. A directed graph of Nodes connected by Edges, defined entirely in Python code. The visual representation is derived from the graph at runtime and updated live during execution — it is not a separate specification that can diverge from the code.

### Node
A single unit of work within a graph. Every node is an idempotent activity: its input state and output state are persisted before and after execution. If a node fails, it is retried according to its retry policy without re-running prior nodes. If the process crashes mid-node, the node re-executes from its persisted input on resume — it never continues from an unknown intermediate state. Idempotency is a contract the developer must uphold; Aegis provides the enforcement infrastructure.

### Edge
A transition between nodes. Edges are unconditional (always traverse) or conditional (route based on current state). Fan-out (parallel) and fan-in (join) edges are supported. Cycles are permitted for retry loops and iterative refinement patterns.

### AgentState
A typed, JSON-serializable dataclass that flows through the graph. Every node receives the current state and returns an updated copy. State is immutable within a node — nodes cannot mutate state in place, only return a new version. State is checkpointed after every successful node completion.

### Checkpoint
A persisted, immutable snapshot of AgentState at a specific step in a specific thread. Checkpoints are append-only. They are the basis for crash recovery (resume), debugging (time-travel), and testing (fork with injected state). Checkpoint IDs are deterministic: `ckpt_{graph_name}_{thread_id}_{step}_{state_hash}`.

### Thread
A named execution context identified by `thread_id`. Multiple runs of the same graph share a thread to maintain continuity across interactions (e.g., a multi-turn conversation is one thread). All checkpoints, traces, and memory for a thread are scoped to that thread. Thread IDs are caller-provided and must be globally unique within a deployment.

### Tool
A registered Python callable with a declared permission scope. Tools are the only sanctioned I/O boundary in Aegis. All network access, filesystem access, database calls, and external API calls must go through a registered tool. Aegis enforces this at both the tool call boundary (explicit check) and the HTTP transport layer (monkey-patching + proxy injection).

### Cassette
A structured JSON file recording every LLM call (prompt, model, parameters, response) and every tool call (name, arguments, result) from a real execution, in order. In replay mode, calls are served from the cassette instead of executing live. Cassettes make test suites fully deterministic and free of API costs. Cassettes carry a content hash; a mismatch on replay indicates the cassette is stale and must be re-recorded.

### Budget
A declared cost envelope for a run, specifying maximum values for LLM cost (USD), total tokens, tool call count, and wall time. When any limit is exceeded, the configured `on_exceeded` policy fires synchronously before the next node begins. Budget enforcement is a hard runtime constraint, not a monitoring alert.

### EvalCase
A single test case in the eval pipeline: an input state, one or more assertions (tool call expectations, state schema checks, golden trace diffs), and an optional cassette for deterministic execution. EvalCases are the unit of CI-gated evaluation.

### PermissionScope
A declarative Python or YAML specification of what an agent is allowed to do: which tools it may call, which network domains it may reach, which filesystem paths it may read or write, and which operations require explicit human approval before execution. Scopes are enforced at runtime, not advisory.

---

## 6. Differentiators

Aegis has five properties that no existing framework provides together. Each is a hard runtime guarantee, not a best-effort feature.

### 6.1 Durable Graph Execution

Every node is an idempotent activity. State is checkpointed after every node completion. A process crash at any point — including mid-deployment — resumes from the last committed checkpoint without data loss and without re-executing completed work. This is architecturally different from LangGraph's checkpoint model, which provides resume within a process but does not survive rolling deployments.

**What this unlocks:** Agents that run for minutes, hours, or days without babysitting. Long-running research agents, overnight data synthesis pipelines, and multi-day approval workflows become viable without custom crash-recovery logic.

### 6.2 First-Class Testing Without Live APIs

Tool mocking and cassette replay are framework primitives, not third-party integrations. `mock_tools()` replaces tool implementations for a test scope and records every call, argument, and return value. `cassette.record()` captures a real execution to file; `cassette.replay()` plays it back with zero live API calls. A standard `pytest` run against cassette-based tests makes no network calls, costs nothing, and produces identical results on every run.

**What this unlocks:** CI pipelines that test agent reasoning without API spend. Reproduction of production failures by replaying the exact cassette from that run. Testing error-handling paths by editing cassette entries to inject failures.

### 6.3 Eval as a Deployment Gate

`EvalSuite` is a first-class runtime concept. Deterministic eval cases (tool call assertions, schema checks, cassette diffs) run in CI and block promotion if any case fails. LLM-as-judge scoring runs as a parallel advisory signal. Teams cannot deploy an agent that regresses on its defined acceptance criteria.

**What this unlocks:** The same confidence for agents that unit tests provide for functions. Prompt changes that silently break tool selection logic are caught in CI before they reach production users.

### 6.4 Declarative Least-Agency Permission Scopes

Every graph and every tool carries a declared permission scope. Scopes specify which tools are callable, which network domains are reachable, which filesystem paths are accessible, and which operations require human approval. Scopes are enforced at three layers: the tool call boundary (explicit pre-execution check), the Python HTTP transport layer (monkey-patching of `httpx`, `requests`, `aiohttp`, `urllib`), and an optional sandboxed execution context for code-executing agents. An agent operating outside its declared scope is blocked and logged — it cannot circumvent enforcement by calling HTTP directly.

**What this unlocks:** Agents that provably cannot delete production data, exfiltrate information, or exceed their authorized surface area — regardless of what the LLM is instructed to do. Immutable audit logs for compliance. Human approval gates for high-risk operations.

### 6.5 Budget Envelopes with On-Exceed Policies

Every run declares a budget: maximum LLM cost in USD, maximum token count, maximum tool calls, maximum wall time. The budget enforcer checks after every LLM call and blocks execution before the next node if any limit is exceeded. The `on_exceeded` policy is configurable per run: `hard_stop`, `pause_and_notify` (await human decision), `downgrade_model` (switch to a cheaper model for remaining nodes), or `compress_context` (summarize conversation history to free token budget).

**What this unlocks:** Runaway agent loops cannot bankrupt a team's API account. Per-agent and per-team budget allocations are enforceable in shared platform deployments. Cost-per-task telemetry is available without external tooling.

---

## 7. API / Interface

### 7.1 Graph and Node Definition

```python
from aegis import graph, node, AgentState, ToolContext, LLMContext
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ResearchState(AgentState):
    query: str
    search_results: list[str] = field(default_factory=list)
    summary: str = ""
    status: Literal["searching", "summarizing", "done", "failed"] = "searching"

@graph(
    name="research-pipeline",
    version="1.0.0",
    checkpointer="postgres",   # "memory" | "sqlite" | "postgres" | "redis" | "dynamodb"
)
async def research_pipeline(state: ResearchState) -> ResearchState:
    state = await research_node(state)
    state = await summarize_node(state)
    return state

@node(
    retries=3,
    retry_backoff="exponential",  # "none" | "linear" | "exponential"
    retry_on=(TimeoutError, ConnectionError),
    timeout_seconds=30,
)
async def research_node(state: ResearchState, tools: ToolContext) -> ResearchState:
    results = await tools.web_search(query=state.query, max_results=5)
    return state.update(
        search_results=[r.text for r in results],
        status="summarizing",
    )

@node(timeout_seconds=60)
async def summarize_node(state: ResearchState, llm: LLMContext) -> ResearchState:
    response = await llm.complete(
        model="anthropic/claude-opus-4-6",
        system="You are a research summarizer. Be concise and factual.",
        prompt=f"Summarize these search results:\n\n{chr(10).join(state.search_results)}",
    )
    return state.update(summary=response.text, status="done")
```

### 7.2 Running a Graph

```python
from aegis import RunConfig, Budget, BudgetDecision, BudgetExceededEvent

# Basic run
result = await research_pipeline.run(
    input=ResearchState(query="quantum computing breakthroughs 2025"),
    config=RunConfig(thread_id="session-abc-123"),
)

print(result.state.summary)
print(result.trace.total_cost_usd)

# Run with budget constraints
result = await research_pipeline.run(
    input=ResearchState(query="quantum computing"),
    config=RunConfig(thread_id="session-abc-123"),
    budget=Budget(
        max_llm_cost_usd=2.00,
        max_tokens=100_000,
        max_tool_calls=20,
        max_wall_time_seconds=120,
        on_exceeded="pause_and_notify",
        notify_at_pct=0.80,  # alert at 80% consumption of any dimension
    ),
)

# Resume a failed or interrupted run from last checkpoint
result = await research_pipeline.resume(thread_id="session-abc-123")

# Fork from a specific checkpoint — inject state at that point, re-run from there
fork = await research_pipeline.fork(
    thread_id="session-abc-123",
    checkpoint_id="ckpt_research-pipeline_session-abc-123_3_a1b2c3",
    inject_state={"search_results": ["Custom injected result for debugging"]},
)

# Inspect checkpoint history for a thread
history = await research_pipeline.get_checkpoint_history(thread_id="session-abc-123")
for ckpt in history:
    print(ckpt.step, ckpt.node_name, ckpt.created_at)

# Stream intermediate state events
async for event in research_pipeline.stream(
    input=ResearchState(query="quantum computing"),
    config=RunConfig(thread_id="session-stream-001"),
):
    # event.type: "node_started" | "node_completed" | "node_failed" | "tool_called" | "llm_called"
    print(event.type, event.node_name, event.state.status)
```

### 7.3 Budget Exceeded Handler

```python
@research_pipeline.on_budget_exceeded
async def handle_budget(event: BudgetExceededEvent) -> BudgetDecision:
    """Called synchronously when any budget dimension is exceeded."""
    if event.exceeded_dimension == "cost" and event.current_status.cost_usd < 3.00:
        # Extend the budget if we're close but not outrageously over
        return BudgetDecision.extend(max_llm_cost_usd=3.00)
    if event.exceeded_dimension == "tokens":
        # Switch to cheaper model for remaining nodes
        return BudgetDecision.downgrade(model="openai/gpt-4o-mini")
    return BudgetDecision.hard_stop()
```

### 7.4 Tool Definition and Permissions

```python
from aegis import tool, ToolPermission, NetworkPermission, FilesystemPermission

@tool(
    name="web_search",
    description="Search the web for information. Returns a list of results with title, URL, and snippet.",
    permissions=ToolPermission(
        network=NetworkPermission(
            allowed_domains=["*.wikipedia.org", "arxiv.org", "news.ycombinator.com"],
            deny_all_others=True,
        ),
    ),
)
async def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    # implementation
    ...

@tool(
    name="write_report",
    description="Write a report to the output directory.",
    permissions=ToolPermission(
        filesystem=FilesystemPermission(
            write=True,
            write_allowed_paths=["/tmp/reports/**"],
        ),
    ),
)
async def write_report(filename: str, content: str) -> None:
    ...

@tool(
    name="send_email",
    description="Send an email. Requires human approval before execution.",
    require_human_approval=True,
    approval_timeout_seconds=3600,
    approval_on_timeout="hard_stop",  # "hard_stop" | "deny" | "approve"
)
async def send_email(to: str, subject: str, body: str) -> None:
    ...
```

### 7.5 Graph-Level Permission Scope

```python
from aegis import PermissionScope, NetworkPermission, FilesystemPermission, ApprovalPolicy

@graph(
    name="research-pipeline",
    version="1.0.0",
    permissions=PermissionScope(
        # Whitelist: only these tools may be called by any node in this graph
        tools=["web_search", "write_report"],
        network=NetworkPermission(
            allowed_domains=["*.wikipedia.org", "arxiv.org"],
            deny_all_others=True,
            max_bytes_out_per_run=10_000,      # exfiltration limit
            max_bytes_in_per_run=5_000_000,    # 5MB max inbound
        ),
        filesystem=FilesystemPermission(
            read=True,
            read_allowed_paths=["/tmp/input/**"],
            write=True,
            write_allowed_paths=["/tmp/output/**"],
        ),
        approval=ApprovalPolicy(
            require_for_patterns=["delete_*", "send_*", "publish_*"],
            timeout_seconds=3600,
            on_timeout="hard_stop",
        ),
    ),
)
async def research_pipeline(state: ResearchState) -> ResearchState:
    ...
```

### 7.6 Testing — Tool Mocking

```python
import pytest
from aegis.testing import MockTool

@pytest.mark.asyncio
async def test_research_calls_web_search_once():
    async with research_pipeline.mock_tools({
        "web_search": MockTool.returns([
            SearchResult(text="Quantum entanglement breakthrough", url="https://arxiv.org/1"),
            SearchResult(text="New qubit stability record", url="https://arxiv.org/2"),
        ]),
        "write_report": MockTool.noop(),
    }) as ctx:
        result = await research_pipeline.run(
            input=ResearchState(query="quantum computing"),
            config=RunConfig(thread_id="test-mock-001"),
        )

    assert result.state.status == "done"
    assert ctx.calls["web_search"].count == 1
    assert ctx.calls["web_search"].calls[0].args["query"] == "quantum computing"
    assert ctx.calls["write_report"].count == 0

@pytest.mark.asyncio
async def test_research_retries_on_timeout():
    async with research_pipeline.mock_tools({
        "web_search": MockTool.raises(TimeoutError("upstream unavailable")),
    }) as ctx:
        result = await research_pipeline.run(
            input=ResearchState(query="quantum computing"),
            config=RunConfig(thread_id="test-mock-002"),
        )

    assert result.status == "failed"
    assert "TimeoutError" in result.error.message
    assert ctx.calls["web_search"].count == 3  # 3 retries exhausted

@pytest.mark.asyncio
async def test_research_handles_empty_results():
    async with research_pipeline.mock_tools({
        "web_search": MockTool.returns([]),  # empty results
    }) as ctx:
        result = await research_pipeline.run(
            input=ResearchState(query="xyzzy gibberish 98765"),
            config=RunConfig(thread_id="test-mock-003"),
        )

    # Agent should gracefully handle empty results, not crash
    assert result.state.status in ("done", "failed")
    assert result.status != "crashed"
```

### 7.7 Testing — Cassette Recording and Replay

```python
from aegis.testing import cassette, MockTool

# Record: run once against live APIs, save all I/O to cassette
# Only runs when pytest is invoked with --aegis-record flag
@pytest.mark.aegis_record
@pytest.mark.asyncio
async def test_record_research():
    async with cassette.record("tests/fixtures/research_quantum.json"):
        result = await research_pipeline.run(
            input=ResearchState(query="quantum computing"),
            config=RunConfig(thread_id="cassette-record-001"),
        )
    assert result.state.status == "done"
    # cassette now contains all LLM prompts/responses and tool call I/O

# Replay: serve all calls from cassette — zero live API calls
@pytest.mark.asyncio
async def test_replay_research():
    async with cassette.replay("tests/fixtures/research_quantum.json") as ctx:
        result = await research_pipeline.run(
            input=ResearchState(query="quantum computing"),
            config=RunConfig(thread_id="cassette-replay-001"),
        )

    assert result.state.status == "done"
    assert ctx.live_calls_made == 0              # no API calls
    assert ctx.replay_calls_served == ctx.total_recorded_calls

# Partial override: replay most calls, mock one tool to test error handling
@pytest.mark.asyncio
async def test_replay_with_search_failure():
    async with cassette.replay(
        "tests/fixtures/research_quantum.json",
        override_tools={"web_search": MockTool.raises(TimeoutError("injected"))},
    ):
        result = await research_pipeline.run(
            input=ResearchState(query="quantum computing"),
            config=RunConfig(thread_id="cassette-override-001"),
        )
    assert result.status == "failed"
    assert "TimeoutError" in result.error.message
```

### 7.8 Eval Pipeline

```python
from aegis.eval import EvalSuite, EvalCase, ToolCallAssertion, SchemaAssertion, TraceSnapshotAssertion

suite = EvalSuite(
    name="research-pipeline-eval",
    graph=research_pipeline,
    cases=[
        EvalCase(
            id="basic-query",
            input=ResearchState(query="quantum computing breakthroughs 2025"),
            cassette="tests/fixtures/research_quantum.json",
            assertions=[
                ToolCallAssertion(
                    tool_name="web_search",
                    called=True,
                    min_times=1,
                    max_times=3,
                    description="must search web at least once",
                ),
                SchemaAssertion(
                    predicate=lambda s: len(s.summary) > 100,
                    description="summary must be non-trivial (>100 chars)",
                ),
                SchemaAssertion(
                    predicate=lambda s: s.status == "done",
                    description="must reach done status",
                ),
            ],
        ),
        EvalCase(
            id="empty-results-graceful",
            input=ResearchState(query="xyzzy gibberish 98765"),
            cassette="tests/fixtures/research_empty.json",
            assertions=[
                SchemaAssertion(
                    predicate=lambda s: s.status in ("done", "failed"),
                    description="must terminate cleanly, not crash",
                ),
            ],
        ),
        EvalCase(
            id="tool-call-sequence-regression",
            input=ResearchState(query="quantum computing"),
            cassette="tests/fixtures/research_quantum.json",
            assertions=[
                TraceSnapshotAssertion(
                    golden_trace_path="tests/golden/research_quantum_trace.json",
                    description="tool call sequence must not regress from golden",
                    normalize_fields=["timestamp", "latency_ms", "run_id", "call_id"],
                ),
            ],
        ),
    ],
    pass_rate_gate=1.0,     # 100% of deterministic cases must pass — hard deploy gate
    llm_judge_gate=0.85,    # LLM-as-judge advisory score threshold — non-blocking by default
    llm_judge_model=None,   # None = must be configured explicitly; no unsafe default
)

# In CI:
results = await suite.run()
results.assert_gates()  # raises EvalGateFailure if pass_rate_gate not met
print(results.summary())
```

### 7.9 Multi-Agent Composition

```python
from aegis import chain, parallel, supervisor, RunConfig

# Sequential: researcher → writer → reviewer
pipeline = chain(researcher_graph, writer_graph, reviewer_graph)
result = await pipeline.run(input=..., config=RunConfig(thread_id="chain-001"))

# Parallel fan-out with join: two researchers → synthesizer
pipeline = parallel(researcher_a, researcher_b).join(synthesizer_graph)
result = await pipeline.run(input=..., config=RunConfig(thread_id="parallel-001"))

# Supervisor routing: router decides which specialist handles the request
pipeline = supervisor(
    router=topic_router_graph,
    specialists={
        "legal": legal_agent_graph,
        "financial": financial_agent_graph,
        "technical": technical_agent_graph,
    },
    max_handoffs=5,        # prevent infinite routing loops
    handoff_timeout=300,   # seconds per handoff
)

# Sub-graph: call a child graph from within a node
@node(timeout_seconds=120)
async def delegate_to_researcher(state: ParentState, graphs: GraphContext) -> ParentState:
    sub_result = await graphs.research_pipeline.run(
        input=ResearchState(query=state.query),
        # OTel trace context automatically propagated — sub-graph spans nest under this node's span
        config=RunConfig(thread_id=f"{state.thread_id}-research"),
    )
    return state.update(research_output=sub_result.state.summary)
```

### 7.10 CLI

```bash
# Initialize a new Aegis project
aegis init my-agent

# Run a graph
aegis run research_pipeline --input '{"query": "quantum computing"}' --thread-id session-001

# Resume a failed run
aegis resume research_pipeline --thread-id session-001

# Show checkpoint history for a thread
aegis history research_pipeline --thread-id session-001

# Record a cassette
aegis record research_pipeline --input '{"query": "quantum computing"}' --cassette tests/fixtures/research_quantum.json

# Run eval suite
aegis eval research_pipeline --suite tests/eval_suite.py

# Show live trace for a running thread
aegis trace --thread-id session-001 --follow

# Show permission scope for a graph
aegis permissions research_pipeline
```

---

## 8. Data Model

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from datetime import datetime

# ── State ──────────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """
    Base class for all graph state objects.
    Must be JSON-serializable. Fields must be typed.
    Subclasses define domain-specific state.
    """
    thread_id: str = ""
    run_id: str = ""

    def update(self, **kwargs) -> "AgentState":
        """Return a new instance with specified fields replaced. Does not mutate self."""
        from dataclasses import replace
        return replace(self, **kwargs)

# ── Checkpointing ──────────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    id: str                              # "ckpt_{graph}_{thread}_{step}_{state_hash}"
    thread_id: str
    run_id: str
    graph_name: str
    graph_version: str
    step: int                            # monotonically increasing within a run
    node_name: str                       # node that just completed
    state_snapshot: dict                 # JSON-serialized AgentState
    created_at: datetime
    parent_checkpoint_id: Optional[str]  # None for step 0; set for sequential steps
    is_fork_root: bool = False           # True if this checkpoint was created via fork()

@dataclass
class CheckpointBackendConfig:
    type: Literal["memory", "sqlite", "postgres", "redis", "dynamodb"]
    connection_string: Optional[str] = None
    table_name: str = "aegis_checkpoints"
    ttl_days: Optional[int] = None       # None = retain indefinitely

# ── Execution Tracing ──────────────────────────────────────────────────────

@dataclass
class LLMCall:
    call_id: str
    model: str
    system_prompt: Optional[str]
    user_prompt: str                     # full rendered prompt after variable substitution
    response: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int                   # from provider cache; affects cost calculation
    cost_usd: float
    latency_ms: int
    timestamp: datetime
    node_name: str
    was_replayed: bool = False           # True if served from cassette

@dataclass
class ToolCall:
    call_id: str
    tool_name: str
    args: dict                           # arguments passed by the LLM
    result: Any                          # return value of the tool
    error: Optional[str]                 # set if the tool raised an exception
    latency_ms: int
    timestamp: datetime
    node_name: str
    was_mocked: bool = False             # True if served by a MockTool
    permission_checked: bool = True
    permission_granted: bool = True
    required_human_approval: bool = False
    human_approved: Optional[bool] = None  # None if no approval was required

@dataclass
class NodeTrace:
    node_name: str
    started_at: datetime
    completed_at: Optional[datetime]
    input_state: dict
    output_state: Optional[dict]
    llm_calls: list[LLMCall]
    tool_calls: list[ToolCall]
    attempt: int                         # 1 for first attempt, 2+ for retries
    status: Literal["running", "completed", "failed", "retrying"]
    error: Optional[str]

@dataclass
class RunTrace:
    run_id: str
    thread_id: str
    graph_name: str
    graph_version: str
    started_at: datetime
    completed_at: Optional[datetime]
    status: Literal["running", "completed", "failed", "paused", "budget_exceeded", "resumed"]
    nodes_executed: list[NodeTrace]
    total_input_tokens: int
    total_output_tokens: int
    total_cached_tokens: int
    total_cost_usd: float
    wall_time_seconds: float
    final_state: Optional[dict]
    error: Optional[str]
    parent_run_id: Optional[str]         # set when this is a sub-graph run
    otel_trace_id: str                   # W3C TraceContext trace ID for external correlation

# ── Budget ─────────────────────────────────────────────────────────────────

@dataclass
class Budget:
    max_tokens: Optional[int] = None
    max_llm_cost_usd: Optional[float] = None
    max_tool_calls: Optional[int] = None
    max_wall_time_seconds: Optional[int] = None
    on_exceeded: Literal[
        "hard_stop",
        "pause_and_notify",
        "downgrade_model",
        "compress_context",
    ] = "hard_stop"
    downgrade_to: Optional[str] = None  # required if on_exceeded="downgrade_model"
    notify_at_pct: float = 0.80         # fire on_budget_warning at this consumption level

@dataclass
class BudgetStatus:
    tokens_used: int
    cost_usd: float
    tool_calls_made: int
    wall_time_seconds: float
    pct_consumed: dict[str, float]      # {"tokens": 0.45, "cost": 0.72, ...}
    exceeded_dimension: Optional[str]   # first dimension that exceeded its limit

@dataclass
class BudgetExceededEvent:
    run_id: str
    thread_id: str
    exceeded_dimension: str             # "tokens" | "cost" | "tool_calls" | "wall_time"
    budget: Budget
    current_status: BudgetStatus
    checkpoint_id: str                  # run can be resumed from here after budget decision

@dataclass
class BudgetDecision:
    action: Literal["hard_stop", "extend", "downgrade"]
    updated_budget: Optional[Budget] = None

    @staticmethod
    def hard_stop() -> "BudgetDecision":
        return BudgetDecision(action="hard_stop")

    @staticmethod
    def extend(**budget_kwargs) -> "BudgetDecision":
        return BudgetDecision(action="extend", updated_budget=Budget(**budget_kwargs))

    @staticmethod
    def downgrade(model: str) -> "BudgetDecision":
        return BudgetDecision(action="downgrade", updated_budget=Budget(downgrade_to=model))

# ── Permissions ────────────────────────────────────────────────────────────

@dataclass
class NetworkPermission:
    allowed_domains: list[str] = field(default_factory=list)  # glob patterns, e.g. "*.wikipedia.org"
    deny_all_others: bool = True
    max_bytes_out_per_run: Optional[int] = None
    max_bytes_in_per_run: Optional[int] = None

@dataclass
class FilesystemPermission:
    read: bool = False
    write: bool = False
    read_allowed_paths: list[str] = field(default_factory=list)   # glob patterns
    write_allowed_paths: list[str] = field(default_factory=list)  # glob patterns

@dataclass
class ApprovalPolicy:
    require_for_patterns: list[str] = field(default_factory=list)  # tool name globs
    timeout_seconds: int = 3600
    on_timeout: Literal["hard_stop", "deny", "approve"] = "hard_stop"
    delivery: Literal["webhook", "slack", "email"] = "webhook"
    delivery_target: Optional[str] = None  # URL, channel, or address

@dataclass
class PermissionScope:
    tools: Optional[list[str]] = None          # None = all registered tools allowed
    network: NetworkPermission = field(default_factory=NetworkPermission)
    filesystem: FilesystemPermission = field(default_factory=FilesystemPermission)
    approval: ApprovalPolicy = field(default_factory=ApprovalPolicy)

@dataclass
class PermissionViolationEvent:
    run_id: str
    thread_id: str
    node_name: str
    violation_type: Literal["tool_not_in_whitelist", "network_domain_denied", "filesystem_path_denied", "http_intercept_blocked"]
    attempted_action: str
    declared_scope: PermissionScope
    timestamp: datetime

# ── Testing ────────────────────────────────────────────────────────────────

@dataclass
class CassetteRecord:
    format_version: str = "1.0"
    graph_name: str = ""
    graph_version: str = ""
    recorded_at: datetime = field(default_factory=datetime.utcnow)
    python_version: str = ""
    entries: list["CassetteEntry"] = field(default_factory=list)
    content_hash: str = ""               # SHA-256 of all entries; mismatch = stale cassette

@dataclass
class CassetteEntry:
    type: Literal["llm_call", "tool_call"]
    call_id: str
    step: int
    node_name: str
    request: dict                        # serialized call inputs (prompt, args, model, params)
    response: dict                       # serialized call outputs (text, result, error)
    timestamp: datetime

@dataclass
class MockCallRecord:
    tool_name: str
    count: int = 0
    calls: list[ToolCall] = field(default_factory=list)

@dataclass
class MockContext:
    calls: dict[str, MockCallRecord]     # tool_name → MockCallRecord
    live_calls_made: int = 0             # should be 0 in a properly mocked test

@dataclass
class CassetteReplayContext:
    cassette: CassetteRecord
    replay_calls_served: int = 0
    total_recorded_calls: int = 0
    live_calls_made: int = 0             # should be 0 in a cassette replay

# ── Eval ───────────────────────────────────────────────────────────────────

@dataclass
class EvalAssertion:
    description: str

@dataclass
class ToolCallAssertion(EvalAssertion):
    tool_name: str
    called: bool = True
    min_times: int = 1
    max_times: Optional[int] = None
    args_match: Optional[dict] = None   # partial dict match on call args

@dataclass
class SchemaAssertion(EvalAssertion):
    predicate: callable                 # (final_state: AgentState) -> bool
    description: str = ""

@dataclass
class TraceSnapshotAssertion(EvalAssertion):
    """Fails if the tool call sequence diverges from the golden trace file."""
    golden_trace_path: str
    normalize_fields: list[str] = field(
        default_factory=lambda: ["timestamp", "latency_ms", "run_id", "call_id", "cost_usd"]
    )

@dataclass
class EvalCase:
    id: str
    input: AgentState
    assertions: list[EvalAssertion]
    cassette: Optional[str] = None       # path to cassette; None = live run (slow, costs money)
    tags: list[str] = field(default_factory=list)
    expected_status: Literal["completed", "failed"] = "completed"

@dataclass
class EvalCaseResult:
    case_id: str
    passed: bool
    assertion_results: list[tuple[EvalAssertion, bool, str]]  # (assertion, passed, message)
    trace: RunTrace
    duration_seconds: float
    live_calls_made: int

@dataclass
class EvalSuiteResult:
    suite_name: str
    total_cases: int
    passed_cases: int
    pass_rate: float
    llm_judge_score: Optional[float]
    case_results: list[EvalCaseResult]
    gate_passed: bool
    duration_seconds: float
    total_cost_usd: float               # 0.00 if all cases used cassettes

    def assert_gates(self) -> None:
        """Raises EvalGateFailure if pass_rate_gate not met. Called in CI."""
        ...

    def summary(self) -> str:
        """Human-readable summary for CI output."""
        ...
```

---

## 9. Architecture

### 9.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  User Code                                                                  │
│  @graph  @node  @tool  AgentState  EvalSuite  Budget  PermissionScope       │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────────────┐
│  Aegis Graph Runtime                                                        │
│                                                                             │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────────────────┐ │
│  │  Node        │  │  State        │  │  Checkpoint Manager              │ │
│  │  Scheduler   │  │  Manager      │  │  · write after each node         │ │
│  │  · DAG exec  │  │  · immutable  │  │  · atomic writes                 │ │
│  │  · retries   │  │    state flow │  │  · resume on crash               │ │
│  │  · timeouts  │  │  · validation │  │  · version conflict detection    │ │
│  └──────────────┘  └───────────────┘  └──────────────────────────────────┘ │
│                                                                             │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────────────────┐ │
│  │  Budget      │  │  Permission   │  │  Trace Collector                 │ │
│  │  Enforcer    │  │  Enforcer     │  │  · OTel span per node            │ │
│  │  · post-LLM  │  │  · pre-tool   │  │  · W3C context propagation       │ │
│  │    check     │  │  · pre-HTTP   │  │  · causal nesting across agents  │ │
│  │  · on_exceed │  │  · violation  │  │  · cost + token accounting       │ │
│  │    policy    │  │    logging    │  │                                  │ │
│  └──────────────┘  └───────────────┘  └──────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┬─────────────────┐
          ▼                ▼                ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────────────┐
│  Checkpoint  │  │  Tool        │  │  HTTP     │  │  Eval Runner     │
│  Backend     │  │  Registry    │  │  Intercept│  │  · cassette I/O  │
│  · SQLite    │  │  · schema    │  │  Layer 1: │  │  · mock dispatch │
│  · Postgres  │  │    registry  │  │   monkey- │  │  · assertion     │
│  · Redis     │  │  · permission│  │   patch   │  │    evaluation    │
│  · DynamoDB  │  │    check     │  │  Layer 2: │  │  · gate check    │
│  · Memory    │  │  · mock/     │  │   proxy   │  │                  │
│              │  │    cassette  │  │  Layer 3: │  │                  │
│              │  │    dispatch  │  │   sandbox │  │                  │
└──────────────┘  └──────────────┘  └───────────┘  └──────────────────┘
```

### 9.2 HTTP Interception Stack

Three layers are applied at framework init, before any user code runs. Each layer is active by default unless explicitly disabled.

**Layer 1 — Python Transport Patching (always active)**
Patches the `send` method of `httpx.Client`, `httpx.AsyncClient`, `requests.Session`, `aiohttp.ClientSession`, and `urllib.request.OpenerDirector` at import time. Every outbound HTTP request is intercepted, checked against the active permission scope, and either allowed or blocked before leaving the process. Covers ~95% of real-world Python HTTP calls.

**Layer 2 — Proxy Injection (always active)**
Sets `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` environment variables at process start to point at an embedded sidecar proxy. Catches subprocess HTTP calls (`curl`, `wget`, `git fetch`) and clients compiled with proxy support that bypass Python transport patching. The embedded proxy enforces the same permission rules as Layer 1 and writes to the same violation log.

**Layer 3 — Sandboxed Execution (opt-in per node)**
For nodes that execute LLM-generated code (AutoGen-style code execution), the generated code runs in a Linux network namespace with iptables egress rules matching the declared permission scope. Catches raw `socket.connect()` calls that bypass Layers 1 and 2. Adds ~150ms cold start per sandboxed execution. Disabled by default; opt-in via `@node(sandbox=True)`.

### 9.3 Checkpoint Recovery Flow

```
Process crash during execution of node_n
              │
              ▼
Developer or orchestrator calls:
  aegis resume --thread-id <id>
  or: await graph.resume(thread_id="...")
              │
              ▼
Aegis loads last committed checkpoint
(node_{n-1} completed successfully)
              │
              ▼
node_n re-executes from its persisted input state
(node contract: idempotent — safe to re-run)
              │
              ▼
Execution continues from node_{n+1} as normal
              │
              ▼
Full RunTrace reconstructed from checkpoint history
+ new execution events; appears as a single run
```

### 9.4 Cassette Staleness Detection

Every cassette file stores a `content_hash` (SHA-256 of all recorded request/response pairs). On replay, Aegis recomputes the hash of what it would send and compares to the stored hash. If any LLM prompt, model, tool arguments, or tool implementation has changed since recording, the hash mismatches and the test fails immediately with:

```
AegisCassetteStaleError: Cassette 'tests/fixtures/research_quantum.json' is stale.
  The following calls no longer match the recorded cassette:
  - Step 2 / research_node / llm_call: prompt hash changed
    Recorded:  sha256:a1b2c3...
    Current:   sha256:d4e5f6...

  Re-record with: pytest --aegis-record tests/test_research.py::test_replay_research
```

### 9.5 Graph Versioning on Deployment

Every checkpoint stores the `graph_name` and `graph_version` of the graph that created it. When `graph.resume(thread_id)` is called, Aegis checks whether the current graph version matches the checkpoint's version. On mismatch:

- **Minor version change** (same major): resume proceeds with a warning logged
- **Major version change**: raises `GraphVersionConflict` with instructions to either migrate the thread or abandon and restart it

Graph authors increment the major version manually when making structural changes to the graph (adding/removing nodes, changing edge logic) that would make existing checkpoints incompatible.

---

## 10. Deployment Model

### 10.1 Three Deployment Targets

| Target | Checkpoint Backend | HTTP Interception | Trace Export | Intended Use |
|--------|-------------------|-------------------|--------------|--------------|
| **Local** | SQLite, auto-created at `~/.aegis/checkpoints.db` | Layer 1 only | Console + `~/.aegis/traces/` | Development, unit testing |
| **Self-hosted** | Postgres or Redis, developer-configured | Layers 1 + 2 | OTel exporter to any backend (Honeycomb, Datadog, Jaeger, stdout) | Enterprise, data sovereignty, regulated industries |
| **Managed (SaaS)** | Managed Postgres, Aegis-hosted | Layers 1 + 2 | Built-in trace UI + OTel export | Teams wanting zero infrastructure |

Dev-to-prod is a configuration change, not a code change. A graph that runs locally with SQLite checkpoints and console traces runs identically in production with Postgres checkpoints and Datadog traces — the only change is the `AegisConfig` passed at startup.

```python
# local dev — zero config
import aegis
# aegis uses defaults: SQLite, console traces

# production — env-var driven
import aegis
aegis.configure(
    checkpointer=aegis.PostgresCheckpointer(os.environ["DATABASE_URL"]),
    tracer=aegis.OTelTracer(endpoint=os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]),
    http_proxy_port=int(os.environ.get("AEGIS_PROXY_PORT", "7890")),
)
```

### 10.2 Pricing Model (Managed Tier)

Pricing is usage-based, not seat-based. Agentic workloads are automation workloads — a team running 10,000 automated runs per day should not pay 10,000× what a team running 1 run per day pays per user seat.

**Dimensions:**
- **Compute time**: billed per second of active node execution (excludes time spent awaiting human approval or external events)
- **Checkpoint storage**: billed per GB-month of checkpoint data retained
- **LLM tokens**: passed through at cost with a fixed percentage margin; no markup by model

**Not billed:** wall-clock time while paused for human-in-the-loop decisions, idle thread time between runs, eval runs using cassettes (no LLM calls to bill).

---

## 11. Failure Modes

| Failure | Probability | Impact | Mitigation |
|---------|-------------|--------|------------|
| LLM API error mid-node | Medium | Single node fails | Configurable retry policy (backoff, max attempts, retry-on exception types) per node; on exhaustion, node fails and run status → `failed`; checkpoint preserved for manual `resume` |
| Process crash during node execution | Low | In-flight node re-executed | Checkpoint written before node start; node idempotency contract ensures safe re-execution; run resumes from last committed state automatically |
| Process crash during checkpoint write | Low | One checkpoint lost | Checkpoint writes are atomic (write-then-rename for SQLite; single DB transaction for Postgres); partial writes detected by hash check; prior checkpoint used on mismatch |
| Budget exceeded | Medium | Unexpected API spend | Budget enforcer checks after every LLM call; fires synchronously before next node; `on_exceeded` policy executes before any further spend; hard_stop is the safe default |
| Tool permission violation | Low | Agent escapes declared scope | Pre-execution check blocks tool call; `PermissionViolationEvent` logged and emitted to trace; node receives `PermissionDeniedError` and handles via retry or failure |
| Direct HTTP permission violation | Low | Agent bypasses tool boundary | Layers 1 + 2 of HTTP intercept block the request; 403-equivalent error returned to calling code; violation logged to audit trail |
| Cassette stale | High (during active development) | Test serves wrong LLM response, masking behavioral regression | Content hash checked on replay; any prompt, model, or tool implementation change causes hash mismatch; test fails immediately with re-record instructions |
| Checkpoint backend unavailable | Low | Durability lost for that run | Configurable: `fail_fast` (default — run refuses to start without checkpointer) or `warn_and_continue` (run proceeds, logs warning, zero durability for that run) |
| Graph version conflict on resume | Low (during deployments) | Resume fails on structurally changed graph | Graph version embedded in every checkpoint; major version mismatch raises `GraphVersionConflict`; developer must migrate or abandon the in-flight thread |
| Multi-agent trace gap | Medium | Disconnected spans, causal chain broken | OTel W3C TraceContext propagated automatically through `graphs.run()` and `chain/parallel/supervisor` primitives; direct HTTP inter-agent calls that bypass Aegis are logged as trace gaps in the parent span |
| LLM-as-judge eval flakiness | High | False-positive CI blocks | LLM-as-judge is non-blocking advisory by default; only deterministic assertions (tool call counts, schema predicates, cassette trace diffs) gate deploys; `llm_judge_gate` must be explicitly opted into as blocking |
| Prompt injection via tool output | Medium | Agent takes unauthorized action | Permission scope enforced at tool and HTTP layer regardless of LLM instruction state; agent cannot exceed its declared scope even if instructed to; blast radius structurally limited |
| Embedding model change breaks semantic memory | Low | Semantic memory returns semantically incorrect results | Every semantic memory entry stores the embedding model ID; queries from a different model fail with `EmbeddingModelMismatch`; explicit `aegis memory reindex --namespace <n>` command required to migrate |
| Sandboxed execution cold start too high | Medium (for code-executing agents) | 150ms overhead per sandboxed node is too high for rapid iteration patterns | Sandbox is opt-in per node (`@node(sandbox=True)`); pre-warmed sandbox pool configurable; sandbox disabled by default so non-code-executing agents are unaffected |

---

## 12. Success Metrics

### Reliability
- 100% of runs that crash mid-node resume correctly from the prior checkpoint within 10 seconds with zero data loss, as measured by the integration test suite's crash-injection tests.
- Zero state loss events in the first 90 days of managed production deployment.
- Zero budget overruns: no run exceeds its declared `Budget` limits after the enforcer is active. Measured as a hard invariant in the test suite, not a trend metric.

### Developer Experience
- Time to first working agent with mocked tools: under 30 minutes for a senior Python engineer with no prior Aegis experience. Measured via usability study with at least 5 engineers before GA.
- CI eval suite (10 cassette-based cases): completes in under 60 seconds. Measured on standard CI hardware (2 vCPU, 4GB RAM).
- `ctx.live_calls_made == 0` in all standard `pytest` runs. Zero API calls during testing is a hard pass criterion for the framework's own test suite.

### Production Safety
- 100% of tool permission violations blocked before execution. Measured via adversarial test suite that attempts to call non-whitelisted tools.
- 100% of direct HTTP violations blocked. Measured via adversarial test suite that attempts `httpx.get()` to non-whitelisted domains from within a node.
- Immutable audit log: every production run produces a complete, append-only trace covering all LLM calls, tool calls, permission checks, and budget consumption.

### Performance
- Node execution overhead (framework cost, excluding LLM and tool latency): p50 < 5ms, p99 < 20ms. Measured with no-op nodes on reference hardware.
- Checkpoint write latency (Postgres backend): p99 < 50ms.
- Cassette replay speed: no slower than 10× wall time of a live run for the same graph (i.e., a 60-second live run replays in under 6 seconds in cassette mode).

### Adoption
- 3 production deployments (real workloads, not experiments) within 6 months of public release.
- 80% of teams using Aegis have a passing eval suite running in CI within 60 days of onboarding. Measured via managed-tier telemetry.

---

## 13. Open Questions

1. **Graph versioning migration tooling.** When a graph structure changes with in-flight threads, the developer currently must manually migrate or abandon old runs. Should Aegis ship a migration DSL analogous to database schema migrations? What is the simplest version of this that is actually usable?

2. **Cassette lifecycle management.** A cassette becomes stale the moment any prompt, model, or tool implementation changes. Should Aegis provide a `--aegis-refresh-cassettes` CI mode that automatically re-records stale cassettes against live APIs and fails if any output schema changes are detected? Who pays the API cost for re-recording?

3. **Sandboxed execution cold start.** The 150ms cold start for network-namespace sandboxing is prohibitive for agents that execute many short code snippets (e.g., a data analysis agent running 50 Python snippets per run). Is a pre-warmed sandbox pool the right answer? At what pool size does the memory overhead become unacceptable?

4. **LLM-as-judge default model.** There is no safe default. Using the same model as the agent being tested creates circular evaluation bias. Using a different model family introduces inter-model disagreement that makes scores incomparable across model upgrades. The current design requires explicit configuration with no default. Is this the right call, or should there be a recommended default with documented caveats?

5. **Multi-tenancy path.** v1 is single-tenant per organization. The v2 path to multi-tenancy requires: separate checkpoint namespaces per tenant, separate permission scopes per tenant, and budget accounting per tenant. Is `thread_id` prefix scoping sufficient for logical isolation in v2, or does production multi-tenancy require separate database schemas or separate checkpoint backend instances?

6. **Approval gate delivery mechanism.** When `require_human_approval=True` fires, the current design specifies `delivery: "webhook" | "slack" | "email"` with a target. Should Aegis ship a default web UI for approval queues, or define the webhook protocol and let teams integrate with their own tooling (Slack bots, Jira, PagerDuty)? Shipping a UI increases scope significantly.

7. **TypeScript support.** Mastra has demonstrated real demand for TypeScript-first agent frameworks. Is TypeScript runtime support a v1 requirement, a v2 roadmap item, or a "never" (Python only, but define an agent protocol so TypeScript agents can participate in Aegis-orchestrated graphs)? This decision affects the core architecture of the tool registry and state serialization layer.

8. **Semantic memory embedding model versioning.** When a user changes their embedding model, all stored semantic memory is incompatible. The current design requires an explicit `aegis memory reindex` command. Is there a lazy reindex strategy (recompute embeddings on read-miss during a migration window) that is safe enough, or does the inconsistency during migration create too much risk for production systems?

9. **Pricing model validation.** The proposed managed-tier pricing (compute time + checkpoint storage + LLM pass-through) needs validation against realistic customer workload profiles before committing. Specifically: what fraction of wall time do real production agents spend in active node execution vs. paused for human approval vs. awaiting external events? If most time is pause time, compute-time billing dramatically undercharges relative to infrastructure cost.

"""EvalSuite, EvalCase, EvalSuiteResult, and EvalGateFailure."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .._models import (
    AgentState,
    EvalCase,
    EvalCaseResult,
    EvalGateFailure,
    EvalSuiteResult,
    RunConfig,
)
from ._assertions import evaluate_assertion


class EvalSuite:
    """Collection of EvalCases that gate deployment."""

    def __init__(
        self,
        name: str,
        graph: Any,          # GraphDef
        cases: list[EvalCase],
        *,
        pass_rate_gate: float = 1.0,
        llm_judge_gate: float = 0.85,
        llm_judge_model: Optional[str] = None,
    ) -> None:
        self.name = name
        self.graph = graph
        self.cases = cases
        self.pass_rate_gate = pass_rate_gate
        self.llm_judge_gate = llm_judge_gate
        self.llm_judge_model = llm_judge_model

    async def run(
        self, *, base_thread_id: str = "eval"
    ) -> EvalSuiteResult:
        """Run all eval cases and return a result summary."""
        suite_start = time.monotonic()
        case_results: list[EvalCaseResult] = []
        total_cost = 0.0

        for case in self.cases:
            result = await self._run_case(case, base_thread_id=base_thread_id)
            case_results.append(result)
            total_cost += result.trace.total_cost_usd

        passed = sum(1 for r in case_results if r.passed)
        total = len(case_results)
        pass_rate = passed / total if total > 0 else 0.0
        gate_passed = pass_rate >= self.pass_rate_gate

        return EvalSuiteResult(
            suite_name=self.name,
            total_cases=total,
            passed_cases=passed,
            pass_rate=pass_rate,
            llm_judge_score=None,   # LLM-as-judge is non-blocking advisory; not implemented in v1
            case_results=case_results,
            gate_passed=gate_passed,
            duration_seconds=time.monotonic() - suite_start,
            total_cost_usd=total_cost,
        )

    async def _run_case(
        self, case: EvalCase, base_thread_id: str
    ) -> EvalCaseResult:
        import asyncio

        thread_id = f"{base_thread_id}-{case.id}"
        config = RunConfig(thread_id=thread_id)
        case_start = time.monotonic()
        live_calls = 0

        # Set up cassette replay if provided
        if case.cassette:
            from ..testing._cassette import cassette as _cassette
            async with _cassette.replay(case.cassette) as replay_ctx:
                run_result = await self.graph.run(
                    input=case.input, config=config
                )
                live_calls = replay_ctx.live_calls_made
        else:
            run_result = await self.graph.run(input=case.input, config=config)

        # Status assertion
        assertion_results = []
        if run_result.status != case.expected_status:
            assertion_results.append(
                (
                    None,
                    False,
                    f"Expected status '{case.expected_status}', got '{run_result.status}'",
                )
            )

        # Run all assertions
        for assertion in case.assertions:
            passed, msg = evaluate_assertion(
                assertion, run_result.state, run_result.trace
            )
            assertion_results.append((assertion, passed, msg))

        overall_passed = all(ok for _, ok, _ in assertion_results)

        return EvalCaseResult(
            case_id=case.id,
            passed=overall_passed,
            assertion_results=assertion_results,  # type: ignore[arg-type]
            trace=run_result.trace,
            duration_seconds=time.monotonic() - case_start,
            live_calls_made=live_calls,
        )

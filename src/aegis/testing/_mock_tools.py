"""MockTool — first-class tool mocking for Aegis tests."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ── Shared testing contextvar ─────────────────────────────────────────────────

@dataclass
class _TestingState:
    mock_tools: Optional[dict[str, "MockTool"]] = None
    mock_ctx_holder: Optional[Any] = None     # MockToolsContext
    cassette_mode: Optional[str] = None
    cassette: Optional[Any] = None
    cassette_override_tools: Optional[dict[str, "MockTool"]] = None


_mock_testing_context: ContextVar[Optional[_TestingState]] = ContextVar(
    "aegis_testing_context", default=None
)


# ── MockTool ──────────────────────────────────────────────────────────────────

class MockTool:
    """A tool replacement that returns a fixed value, raises an exception, or executes a custom callable."""

    def __init__(
        self,
        *,
        return_value: Any = None,
        raise_exception: Optional[BaseException] = None,
        side_effect: Optional[Callable[..., Any]] = None,
        noop: bool = False,
    ) -> None:
        self._return_value = return_value
        self._raise_exception = raise_exception
        self._side_effect = side_effect
        self._noop = noop

    async def execute(self, kwargs: dict[str, Any]) -> Any:
        if self._raise_exception is not None:
            raise self._raise_exception
        if self._noop:
            return None
        if self._side_effect is not None:
            result = self._side_effect(**kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
        return self._return_value

    # ── Factory methods (matches PRD API) ─────────────────────────────────────

    @classmethod
    def returns(cls, value: Any) -> "MockTool":
        """Return a fixed value on every call."""
        return cls(return_value=value)

    @classmethod
    def noop(cls) -> "MockTool":
        """Return None on every call (no-op)."""
        return cls(noop=True)

    @classmethod
    def raises(cls, exc: BaseException) -> "MockTool":
        """Raise the given exception on every call."""
        return cls(raise_exception=exc)

    @classmethod
    def calls(cls, fn: Callable[..., Any]) -> "MockTool":
        """Delegate to a callable (sync or async)."""
        return cls(side_effect=fn)

"""Abstract base protocol for checkpoint backends."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .._models import Checkpoint


@runtime_checkable
class CheckpointerBase(Protocol):
    """Protocol that all checkpoint backends must implement."""

    async def save(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint atomically."""
        ...

    async def get_latest(self, thread_id: str, graph_name: str) -> Optional[Checkpoint]:
        """Return the most recent checkpoint for a thread, or None."""
        ...

    async def get_by_step(
        self, thread_id: str, graph_name: str, step: int
    ) -> Optional[Checkpoint]:
        """Return the checkpoint at a specific step, or None."""
        ...

    async def get_history(
        self, thread_id: str, graph_name: str
    ) -> list[Checkpoint]:
        """Return all checkpoints for a thread, ordered by step ascending."""
        ...

    async def delete_thread(self, thread_id: str, graph_name: str) -> None:
        """Delete all checkpoints for a thread (e.g., for cleanup after fork)."""
        ...

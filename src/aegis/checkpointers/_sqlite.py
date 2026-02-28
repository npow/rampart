"""SQLite checkpoint backend — for local development with persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .._models import Checkpoint
from ._base import CheckpointerBase

_DEFAULT_DB_PATH = Path.home() / ".aegis" / "checkpoints.db"


class SqliteCheckpointer:
    """Persistent checkpoint store backed by SQLite via aiosqlite.

    Creates the database and table on first use.
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def _get_db(self) -> "aiosqlite.Connection":
        try:
            import aiosqlite  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "aiosqlite is required for SqliteCheckpointer. "
                "Install it with: pip install aegis[sqlite]"
            ) from exc
        db = await aiosqlite.connect(str(self._db_path))
        db.row_factory = aiosqlite.Row
        if not self._initialized:
            await self._init_schema(db)
            self._initialized = True
        return db

    async def _init_schema(self, db: "aiosqlite.Connection") -> None:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS aegis_checkpoints (
                id           TEXT NOT NULL,
                thread_id    TEXT NOT NULL,
                run_id       TEXT NOT NULL,
                graph_name   TEXT NOT NULL,
                graph_version TEXT NOT NULL,
                step         INTEGER NOT NULL,
                node_name    TEXT NOT NULL,
                state_snapshot TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                is_fork_root INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (thread_id, graph_name, step)
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_thread_graph "
            "ON aegis_checkpoints(thread_id, graph_name)"
        )
        await db.commit()

    @staticmethod
    def _row_to_checkpoint(row: Any) -> Checkpoint:
        return Checkpoint(
            id=row["id"],
            thread_id=row["thread_id"],
            run_id=row["run_id"],
            graph_name=row["graph_name"],
            graph_version=row["graph_version"],
            step=row["step"],
            node_name=row["node_name"],
            state_snapshot=json.loads(row["state_snapshot"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            parent_checkpoint_id=row["parent_checkpoint_id"],
            is_fork_root=bool(row["is_fork_root"]),
        )

    async def save(self, checkpoint: Checkpoint) -> None:
        db = await self._get_db()
        try:
            await db.execute(
                """
                INSERT OR REPLACE INTO aegis_checkpoints
                  (id, thread_id, run_id, graph_name, graph_version, step,
                   node_name, state_snapshot, created_at, parent_checkpoint_id,
                   is_fork_root)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    checkpoint.id,
                    checkpoint.thread_id,
                    checkpoint.run_id,
                    checkpoint.graph_name,
                    checkpoint.graph_version,
                    checkpoint.step,
                    checkpoint.node_name,
                    json.dumps(checkpoint.state_snapshot, default=str),
                    checkpoint.created_at.isoformat(),
                    checkpoint.parent_checkpoint_id,
                    int(checkpoint.is_fork_root),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    async def get_latest(self, thread_id: str, graph_name: str) -> Optional[Checkpoint]:
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT * FROM aegis_checkpoints "
                "WHERE thread_id=? AND graph_name=? "
                "ORDER BY step DESC LIMIT 1",
                (thread_id, graph_name),
            ) as cursor:
                row = await cursor.fetchone()
                return self._row_to_checkpoint(row) if row else None
        finally:
            await db.close()

    async def get_by_step(
        self, thread_id: str, graph_name: str, step: int
    ) -> Optional[Checkpoint]:
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT * FROM aegis_checkpoints "
                "WHERE thread_id=? AND graph_name=? AND step=?",
                (thread_id, graph_name, step),
            ) as cursor:
                row = await cursor.fetchone()
                return self._row_to_checkpoint(row) if row else None
        finally:
            await db.close()

    async def get_history(
        self, thread_id: str, graph_name: str
    ) -> list[Checkpoint]:
        db = await self._get_db()
        try:
            async with db.execute(
                "SELECT * FROM aegis_checkpoints "
                "WHERE thread_id=? AND graph_name=? ORDER BY step ASC",
                (thread_id, graph_name),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_checkpoint(r) for r in rows]
        finally:
            await db.close()

    async def delete_thread(self, thread_id: str, graph_name: str) -> None:
        db = await self._get_db()
        try:
            await db.execute(
                "DELETE FROM aegis_checkpoints WHERE thread_id=? AND graph_name=?",
                (thread_id, graph_name),
            )
            await db.commit()
        finally:
            await db.close()



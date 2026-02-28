"""Global Aegis configuration helpers."""

from __future__ import annotations

from typing import Any, Optional


def configure(
    *,
    checkpointer: Optional[Any] = None,
    tracer: Optional[Any] = None,
    http_proxy_port: Optional[int] = None,
) -> None:
    """Configure global Aegis defaults.

    In production, call this at application startup:

        import aegis, os
        aegis.configure(
            checkpointer=aegis.PostgresCheckpointer(os.environ["DATABASE_URL"]),
            tracer=aegis.OTelTracer(endpoint=os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]),
        )
    """
    import aegis._globals as _g
    if checkpointer is not None:
        _g.DEFAULT_CHECKPOINTER = checkpointer
    if tracer is not None:
        _g.DEFAULT_TRACER = tracer
    if http_proxy_port is not None:
        _g.HTTP_PROXY_PORT = http_proxy_port


class PostgresCheckpointer:
    """Postgres-backed checkpoint store (requires asyncpg: pip install aegis[postgres])."""

    def __init__(self, connection_string: str, table_name: str = "aegis_checkpoints") -> None:
        self.connection_string = connection_string
        self.table_name = table_name

    async def save(self, checkpoint: Any) -> None:
        raise NotImplementedError(
            "PostgresCheckpointer not yet implemented. Install asyncpg and contribute!"
        )

    async def get_latest(self, thread_id: str, graph_name: str) -> Any:
        raise NotImplementedError

    async def get_by_step(self, thread_id: str, graph_name: str, step: int) -> Any:
        raise NotImplementedError

    async def get_history(self, thread_id: str, graph_name: str) -> list[Any]:
        raise NotImplementedError

    async def delete_thread(self, thread_id: str, graph_name: str) -> None:
        raise NotImplementedError


class OTelTracer:
    """OpenTelemetry trace exporter stub."""

    def __init__(self, endpoint: Optional[str] = None) -> None:
        self.endpoint = endpoint

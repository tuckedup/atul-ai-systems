"""LangGraph checkpoint saver backed by SQLite.

Use this when Postgres is unavailable (amendment: Postgres -> SQLite). The full
PostgresSaver path is still the production default; this exists so forgecode's
pause/resume can be verified without docker.

Usage in graph.py::

    from aisys.checkpoint_sqlite import SQLiteSaver
    with SQLiteSaver.from_path(".local/checkpoints.db") as saver:
        saver.setup()
        app = graph.compile(checkpointer=saver)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    MessageContent,
    Output,
)
from langgraph.utils import get_configurable


class SQLiteSaver(BaseCheckpointSaver):
    """A single-file SQLite checkpoint store. Thread-safe enough for single-process demos."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints "
            "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
            "checkpoint TEXT, metadata TEXT, PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS channel_values "
            "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
            "channel TEXT, value TEXT, PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, channel))"
        )
        self._conn.commit()

    @classmethod
    def from_path(cls, path: str | Path) -> "SQLiteSaver":
        return cls(path)

    # ---- BaseCheckpointSaver API ----

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_channels: list[tuple[str, list[MessageContent]]],
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        ns = config.get("checkpoint_ns", "")
        cpid = checkpoint["id"]
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata) "
            "VALUES (?,?,?,?,?)",
            (thread_id, ns, cpid, json.dumps(checkpoint, default=str), json.dumps(metadata, default=str)),
        )
        self._conn.execute(
            "DELETE FROM channel_values WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
            (thread_id, ns, cpid),
        )
        for channel, values in new_channels:
            self._conn.execute(
                "INSERT INTO channel_values (thread_id, checkpoint_ns, checkpoint_id, channel, value) "
                "VALUES (?,?,?,?,?)",
                (thread_id, ns, cpid, channel, json.dumps(values, default=str)),
            )
        self._conn.commit()

    def get(
        self,
        config: dict[str, Any],
        checkpoint_id: str,
        *,
        code_executor_id: str | None = None,
    ) -> tuple[Checkpoint, CheckpointMetadata] | None:
        thread_id = config["configurable"]["thread_id"]
        ns = config.get("checkpoint_ns", "")
        row = self._conn.execute(
            "SELECT checkpoint, metadata FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
            (thread_id, ns, checkpoint_id),
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0]), json.loads(row[1])

    def list(
        self,
        config: dict[str, Any],
        *,
        before: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, Checkpoint, CheckpointMetadata]]:
        thread_id = config["configurable"]["thread_id"]
        ns = config.get("checkpoint_ns", "")
        rows = self._conn.execute(
            "SELECT checkpoint_id, checkpoint, metadata FROM checkpoints "
            "WHERE thread_id=? AND checkpoint_ns=? "
            "AND checkpoint_id < ? "
            "ORDER BY checkpoint_id DESC LIMIT ?",
            (thread_id, ns, before or "zzz", limit),
        ).fetchall()
        return [(r[0], json.loads(r[1]), json.loads(r[2])) for r in rows]

    def delete(self, config: dict[str, Any], checkpoint_id: str) -> None:
        thread_id = config["configurable"]["thread_id"]
        ns = config.get("checkpoint_ns", "")
        self._conn.execute(
            "DELETE FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
            (thread_id, ns, checkpoint_id),
        )
        self._conn.execute(
            "DELETE FROM channel_values WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
            (thread_id, ns, checkpoint_id),
        )
        self._conn.commit()

    def list_indexes(self, config: dict[str, Any]) -> list[str]:
        thread_id = config["configurable"]["thread_id"]
        ns = config.get("checkpoint_ns", "")
        rows = self._conn.execute(
            "SELECT DISTINCT checkpoint_id FROM checkpoints WHERE thread_id=? AND checkpoint_ns=?",
            (thread_id, ns),
        ).fetchall()
        return [r[0] for r in rows]

    def cleanup(self, config: dict[str, Any], threshold: str, *, last_versions: int = 1) -> list[str]:
        thread_id = config["configurable"]["thread_id"]
        ns = config.get("checkpoint_ns", "")
        keep = self._conn.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? "
            "ORDER BY checkpoint_id DESC LIMIT ?",
            (thread_id, ns, last_versions),
        ).fetchall()
        keep_ids = {r[0] for r in keep}
        rows = self._conn.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id < ?",
            (thread_id, ns, threshold),
        ).fetchall()
        deleted: list[str] = []
        for (cid,) in rows:
            if cid not in keep_ids:
                self._conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
                    (thread_id, ns, cid),
                )
                self._conn.execute(
                    "DELETE FROM channel_values WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
                    (thread_id, ns, cid),
                )
                deleted.append(cid)
        self._conn.commit()
        return deleted

    def setup(self) -> None:
        """No-op: schema is created in __init__."""
        pass

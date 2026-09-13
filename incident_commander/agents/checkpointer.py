"""SQLite checkpointer for Incident Commander — Degraded Mode durable state."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver


_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", ".local", "ic_checkpoints.db")


class Checkpointer:
    """Wraps SqliteSaver with convenience methods for IC workflow."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or _DEFAULT_DB
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._saver = SqliteSaver(self._conn)
        self._saver.setup()

    @property
    def saver(self) -> SqliteSaver:
        return self._saver

    def list_threads(self) -> list[dict[str, Any]]:
        """List all checkpoint threads."""
        threads = []
        for config_tuple in self._saver.list(None):
            thread_id = config_tuple[1].get("configurable", {}).get("thread_id")
            if thread_id and thread_id not in [t["thread_id"] for t in threads]:
                threads.append({"thread_id": thread_id})
        return threads

    def get_latest(self, thread_id: str) -> dict[str, Any] | None:
        """Get the latest checkpoint state for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = self._saver.get(config)
        if checkpoint is None:
            return None
        return checkpoint.get("channel_values", {})

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        self._saver.delete(config)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


def get_checkpointer(db_path: str | None = None) -> Checkpointer:
    """Get or create a checkpointer instance."""
    return Checkpointer(db_path)

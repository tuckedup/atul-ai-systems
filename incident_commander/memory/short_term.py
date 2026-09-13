"""Short-term memory — per-incident working memory for evidence, hypotheses, decisions."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from aisys.tracing import traced


_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", ".local", "ic_memory.db")

DDL = """
CREATE TABLE IF NOT EXISTS short_term (
    incident_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (incident_id, key)
);
"""


class ShortTermMemory:
    """Per-incident working memory backed by SQLite."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or _DEFAULT_DB
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(DDL)
        self._conn.commit()

    @traced(kind="tool")
    def store(self, incident_id: str, key: str, value: Any) -> None:
        """Store a key-value pair for an incident."""
        self._conn.execute(
            "INSERT OR REPLACE INTO short_term (incident_id, key, value, created_at) VALUES (?, ?, ?, ?)",
            (incident_id, key, json.dumps(value, default=str), time.time()),
        )
        self._conn.commit()

    @traced(kind="tool")
    def recall(self, incident_id: str, key: str) -> Any | None:
        """Recall a value by incident_id and key."""
        row = self._conn.execute(
            "SELECT value FROM short_term WHERE incident_id = ? AND key = ?",
            (incident_id, key),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    @traced(kind="tool")
    def recall_all(self, incident_id: str) -> dict[str, Any]:
        """Recall all key-value pairs for an incident."""
        rows = self._conn.execute(
            "SELECT key, value FROM short_term WHERE incident_id = ? ORDER BY created_at",
            (incident_id,),
        ).fetchall()
        return {key: json.loads(val) for key, val in rows}

    @traced(kind="tool")
    def list_incidents(self) -> list[str]:
        """List all incident IDs with stored memory."""
        rows = self._conn.execute(
            "SELECT DISTINCT incident_id FROM short_term"
        ).fetchall()
        return [row[0] for row in rows]

    def clear(self, incident_id: str) -> None:
        """Clear all memory for an incident."""
        self._conn.execute("DELETE FROM short_term WHERE incident_id = ?", (incident_id,))
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

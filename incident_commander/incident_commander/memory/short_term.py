"""Per-incident evidence and decisions stored in the shared database."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ShortTermMemory:
    def __init__(self, dsn: str):
        path = Path(dsn.removeprefix("sqlite:///"))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS incident_memory (incident TEXT, kind TEXT, value TEXT)")
        self.db.commit()

    def add(self, incident: str, kind: str, value: dict[str, Any]) -> None:
        self.db.execute("INSERT INTO incident_memory VALUES (?,?,?)", (incident, kind, json.dumps(value)))
        self.db.commit()

    def get(self, incident: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT kind, value FROM incident_memory WHERE incident=?", (incident,)).fetchall()
        return [{"kind": kind, **json.loads(value)} for kind, value in rows]


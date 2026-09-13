"""Cross-incident lexical similarity memory."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path


class LongTermMemory:
    def __init__(self, dsn: str):
        path = Path(dsn.removeprefix("sqlite:///"))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS incident_history (id TEXT PRIMARY KEY, summary TEXT, outcome TEXT)")
        self.db.commit()

    def remember(self, incident_id: str, summary: str, outcome: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO incident_history VALUES (?,?,?)", (incident_id, summary, outcome))
        self.db.commit()

    def recall(self, query: str, limit: int = 3) -> list[dict[str, str]]:
        terms = set(re.findall(r"\w+", query.lower()))
        rows = self.db.execute("SELECT id, summary, outcome FROM incident_history").fetchall()
        ranked = sorted(rows, key=lambda row: -len(terms & set(re.findall(r"\w+", row[1].lower()))))
        return [{"id": row[0], "summary": row[1], "outcome": row[2]} for row in ranked[:limit] if terms & set(re.findall(r"\w+", row[1].lower()))]


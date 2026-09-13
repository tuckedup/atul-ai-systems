"""Model registry and quality matrix stored through the single database DSN."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class ModelState:
    def __init__(self, dsn: str):
        if not dsn.startswith("sqlite:///"):
            raise ValueError("this host's state adapter requires the configured SQLite DSN")
        path = Path(dsn.removeprefix("sqlite:///"))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS model_state (model TEXT PRIMARY KEY, provider TEXT, quality TEXT, weight REAL)"
        )
        self.connection.commit()

    def upsert(self, model: str, provider: str, quality: dict[str, float], weight: float = 1.0) -> None:
        self.connection.execute(
            "INSERT INTO model_state VALUES (?,?,?,?) ON CONFLICT(model) DO UPDATE SET "
            "provider=excluded.provider, quality=excluded.quality, weight=excluded.weight",
            (model, provider, json.dumps(quality, sort_keys=True), weight),
        )
        self.connection.commit()

    def quality_matrix(self) -> dict[str, dict[str, float]]:
        rows = self.connection.execute("SELECT model, quality FROM model_state").fetchall()
        return {model: json.loads(quality) for model, quality in rows}

    def rows(self) -> list[dict[str, object]]:
        rows = self.connection.execute("SELECT model, provider, quality, weight FROM model_state").fetchall()
        return [{"model": r[0], "provider": r[1], "quality": json.loads(r[2]), "weight": r[3]} for r in rows]

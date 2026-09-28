"""Model registry, traffic weights and quality matrix, stored through the single DSN.

Two changes from the previous version.

**Backend neutrality.** It used to open `sqlite3` directly and reject anything else:

    if not dsn.startswith("sqlite:///"):
        raise ValueError("this host's state adapter requires the configured SQLite DSN")

That defeats the standing decision that PostgreSQL can be restored by changing one settings
value, and it is inconsistent with `aisys.audit.AuditLog`, which already dispatches on the DSN
scheme. This module now follows that same pattern: parameter style and DDL differ per backend,
the rest of the code does not care.

**Quality is bound to the judge that produced it.** A stored quality score is only meaningful
alongside the calibration artifact, rubric set and dataset it came from. Persisting a bare float
means that after a rubric change the router keeps serving numbers measured by a judge that no
longer exists, with nothing in the schema to reveal it. `upsert` therefore records the bundle id,
rubric hash and dataset hash, and `quality_matrix()` can filter to rows that match the artifact
currently in force.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS model_state (
  model TEXT PRIMARY KEY, provider TEXT, quality TEXT, weight REAL,
  calibration_bundle_id TEXT, rubric_bundle_hash TEXT, dataset_hash TEXT,
  n_samples TEXT, measured_at TEXT)
"""
DDL_PG = """
CREATE TABLE IF NOT EXISTS model_state (
  model TEXT PRIMARY KEY, provider TEXT, quality JSONB, weight DOUBLE PRECISION,
  calibration_bundle_id TEXT, rubric_bundle_hash TEXT, dataset_hash TEXT,
  n_samples JSONB, measured_at TEXT)
"""

_COLUMNS = (
    "model", "provider", "quality", "weight", "calibration_bundle_id",
    "rubric_bundle_hash", "dataset_hash", "n_samples", "measured_at",
)


class ModelState:
    """Backend-neutral model registry keyed on the configured DSN."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._lock = threading.RLock()
        self.sqlite = dsn.startswith("sqlite")
        if self.sqlite:
            path = Path(dsn.removeprefix("sqlite:///"))
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(path, check_same_thread=False)
            self.connection.execute(DDL_SQLITE)
            self.connection.commit()
        else:
            import psycopg

            self.connection = psycopg.connect(dsn, autocommit=True)  # type: ignore[assignment]
            self.connection.execute(DDL_PG)  # type: ignore[union-attr]

    @property
    def _ph(self) -> str:
        return "?" if self.sqlite else "%s"

    @staticmethod
    def _dump(value: Any) -> str:
        # Canonical JSON text. SQLite has no JSON type so this is stored as TEXT; psycopg casts
        # the same string into JSONB on insert, so one representation serves both backends.
        return json.dumps(value, sort_keys=True)

    @staticmethod
    def _loads(value: Any) -> Any:
        return json.loads(value) if isinstance(value, (str, bytes)) else value

    def upsert(
        self,
        model: str,
        provider: str,
        quality: dict[str, float],
        weight: float = 1.0,
        *,
        calibration_bundle_id: str = "",
        rubric_bundle_hash: str = "",
        dataset_hash: str = "",
        n_samples: dict[str, int] | None = None,
        measured_at: str = "",
    ) -> None:
        ph = ", ".join([self._ph] * len(_COLUMNS))
        updates = ", ".join(f"{c}=excluded.{c}" for c in _COLUMNS[1:])
        row = (
            model, provider, self._dump(quality), weight, calibration_bundle_id,
            rubric_bundle_hash, dataset_hash, self._dump(n_samples or {}), measured_at,
        )
        with self._lock:
            self.connection.execute(
                f"INSERT INTO model_state ({', '.join(_COLUMNS)}) VALUES ({ph}) "
                f"ON CONFLICT(model) DO UPDATE SET {updates}",
                row,
            )
            if self.sqlite:
                self.connection.commit()

    def quality_matrix(
        self,
        *,
        rubric_bundle_hash: str | None = None,
        calibration_bundle_id: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """The `{model: {task_class: quality}}` shape the router consumes.

        Passing `rubric_bundle_hash` or `calibration_bundle_id` excludes rows measured under a
        different judge. A caller that omits both gets everything, including stale rows -- so the
        router's wiring should pass the hashes of the artifact currently in force.
        """
        with self._lock:
            rows = self.connection.execute(
                "SELECT model, quality, rubric_bundle_hash, calibration_bundle_id FROM model_state"
            ).fetchall()
        out: dict[str, dict[str, float]] = {}
        for model, quality, rubric_hash, bundle_id in rows:
            if rubric_bundle_hash is not None and rubric_hash != rubric_bundle_hash:
                continue
            if calibration_bundle_id is not None and bundle_id != calibration_bundle_id:
                continue
            out[model] = self._loads(quality)
        return out

    def stale(self, *, rubric_bundle_hash: str) -> list[str]:
        """Models whose stored quality was measured under a different rubric set."""
        with self._lock:
            rows = self.connection.execute(
                "SELECT model, rubric_bundle_hash FROM model_state"
            ).fetchall()
        return sorted(model for model, h in rows if h != rubric_bundle_hash)

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM model_state"
            ).fetchall()
        return [
            {
                "model": r[0], "provider": r[1], "quality": self._loads(r[2]), "weight": r[3],
                "calibration_bundle_id": r[4], "rubric_bundle_hash": r[5],
                "dataset_hash": r[6], "n_samples": self._loads(r[7]), "measured_at": r[8],
            }
            for r in rows
        ]

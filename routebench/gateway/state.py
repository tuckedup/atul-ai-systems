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
import math
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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

ROLLOUT_DDL_SQLITE = (
    """CREATE TABLE IF NOT EXISTS rollout_records (
      rollout_id TEXT PRIMARY KEY, candidate TEXT NOT NULL, incumbent TEXT NOT NULL,
      task_class TEXT NOT NULL, calibration_bundle_id TEXT NOT NULL,
      rubric_bundle_hash TEXT NOT NULL, dataset_hash TEXT NOT NULL,
      stage TEXT NOT NULL, status TEXT NOT NULL, started_at REAL NOT NULL,
      stage_started_at REAL NOT NULL, deadline_at REAL, baseline_weights TEXT NOT NULL,
      offline_evidence TEXT NOT NULL, reason TEXT NOT NULL, updated_at REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS rollout_events (
      sequence INTEGER PRIMARY KEY AUTOINCREMENT, rollout_id TEXT NOT NULL,
      timestamp REAL NOT NULL, elapsed_seconds REAL NOT NULL, from_stage TEXT NOT NULL,
      to_stage TEXT NOT NULL, reason TEXT NOT NULL, weights TEXT NOT NULL,
      details TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS rollout_baselines (
      rollout_id TEXT PRIMARY KEY, captured_at REAL NOT NULL, weights TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS rollout_observations (
      observation_id TEXT PRIMARY KEY, rollout_id TEXT NOT NULL, stage TEXT NOT NULL,
      observed_at REAL NOT NULL, value REAL NOT NULL, incumbent_value REAL,
      evidence TEXT NOT NULL
    )""",
    "CREATE UNIQUE INDEX IF NOT EXISTS rollout_one_active ON rollout_records (status) WHERE status='active'",
)

ROLLOUT_DDL_PG = (
    ROLLOUT_DDL_SQLITE[0].replace("REAL", "DOUBLE PRECISION").replace(" TEXT NOT NULL", " TEXT NOT NULL"),
    """CREATE TABLE IF NOT EXISTS rollout_events (
      sequence BIGSERIAL PRIMARY KEY, rollout_id TEXT NOT NULL,
      timestamp DOUBLE PRECISION NOT NULL, elapsed_seconds DOUBLE PRECISION NOT NULL,
      from_stage TEXT NOT NULL, to_stage TEXT NOT NULL, reason TEXT NOT NULL,
      weights JSONB NOT NULL, details JSONB NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS rollout_baselines (
      rollout_id TEXT PRIMARY KEY, captured_at DOUBLE PRECISION NOT NULL, weights JSONB NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS rollout_observations (
      observation_id TEXT PRIMARY KEY, rollout_id TEXT NOT NULL, stage TEXT NOT NULL,
      observed_at DOUBLE PRECISION NOT NULL, value DOUBLE PRECISION NOT NULL,
      incumbent_value DOUBLE PRECISION, evidence JSONB NOT NULL
    )""",
    "CREATE UNIQUE INDEX IF NOT EXISTS rollout_one_active ON rollout_records (status) WHERE status='active'",
)

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
            for ddl in ROLLOUT_DDL_SQLITE:
                self.connection.execute(ddl)
            self.connection.commit()
        else:
            import psycopg

            self.connection = psycopg.connect(dsn, autocommit=True)  # type: ignore[assignment]
            self.connection.execute(DDL_PG)  # type: ignore[union-attr]
            for ddl in ROLLOUT_DDL_PG:
                self.connection.execute(ddl)  # type: ignore[union-attr]

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

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """Serialize a state change and commit or roll it back as one unit."""
        with self._lock:
            if self.sqlite:
                try:
                    self.connection.execute("BEGIN IMMEDIATE")
                    yield self.connection
                except BaseException:
                    self.connection.rollback()
                    raise
                else:
                    self.connection.commit()
            else:
                with self.connection.transaction():  # type: ignore[union-attr]
                    yield self.connection

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
        dataset_hash: str | None = None,
        strict_provenance: bool = False,
    ) -> dict[str, dict[str, float]]:
        """The `{model: {task_class: quality}}` shape the router consumes.

        Provenance filters exclude rows measured under a different judge or corpus.  With
        ``strict_provenance=True`` every returned row must have non-empty bundle, rubric,
        dataset, timestamp and per-task sample counts, and every quality value must be finite.
        Strict mode raises instead of silently routing from unverifiable measurements.
        """
        with self._lock:
            rows = self.connection.execute(
                "SELECT model, quality, rubric_bundle_hash, calibration_bundle_id, "
                "dataset_hash, n_samples, measured_at FROM model_state"
            ).fetchall()
        out: dict[str, dict[str, float]] = {}
        for model, quality, rubric_hash, bundle_id, row_dataset, samples, measured_at in rows:
            if rubric_bundle_hash is not None and rubric_hash != rubric_bundle_hash:
                continue
            if calibration_bundle_id is not None and bundle_id != calibration_bundle_id:
                continue
            if dataset_hash is not None and row_dataset != dataset_hash:
                continue
            quality_values = self._loads(quality)
            sample_values = self._loads(samples)
            if strict_provenance:
                missing = [
                    name for name, value in (
                        ("calibration_bundle_id", bundle_id),
                        ("rubric_bundle_hash", rubric_hash),
                        ("dataset_hash", row_dataset),
                        ("measured_at", measured_at),
                    ) if not isinstance(value, str) or not value.strip()
                ]
                if missing:
                    raise ValueError(f"model {model!r} has incomplete measured provenance: {missing}")
                if not isinstance(quality_values, dict) or not quality_values:
                    raise ValueError(f"model {model!r} has no measured quality values")
                if not isinstance(sample_values, dict):
                    raise ValueError(f"model {model!r} has no per-task sample counts")
                for task, value in quality_values.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"model {model!r} task {task!r} has non-finite quality")
                    count = sample_values.get(task)
                    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                        raise ValueError(f"model {model!r} task {task!r} has no measured sample count")
            out[model] = quality_values
        return out

    def weight_map(self) -> dict[str, float]:
        """Return an immutable-by-value snapshot of all registered traffic weights."""
        with self._lock:
            rows = self.connection.execute("SELECT model, weight FROM model_state").fetchall()
        return {str(model): float(weight) for model, weight in rows}

    snapshot_weights = weight_map

    @staticmethod
    def _validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
        if not weights:
            raise ValueError("a traffic weight mapping must not be empty")
        clean: dict[str, float] = {}
        for model, value in weights.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError("traffic weight model ids must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"traffic weight for {model!r} must be numeric")
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"traffic weight for {model!r} must be finite and non-negative")
            clean[model] = number
        return clean

    def _apply_weight_map(self, connection: Any, weights: Mapping[str, float], *, exact: bool) -> None:
        existing = {row[0] for row in connection.execute("SELECT model FROM model_state").fetchall()}
        requested = set(weights)
        missing = requested - existing
        if missing:
            raise KeyError(f"unknown models in traffic weights: {sorted(missing)}")
        if exact and requested != existing:
            raise ValueError(
                "an exact traffic weight map must cover every registered model; "
                f"missing={sorted(existing - requested)}, extra={sorted(requested - existing)}"
            )
        for model, weight in weights.items():
            connection.execute(
                f"UPDATE model_state SET weight={self._ph} WHERE model={self._ph}",
                (weight, model),
            )

    def apply_weight_map(self, weights: Mapping[str, float], *, exact: bool = True) -> dict[str, float]:
        """Validate and atomically apply a complete weight map, returning its prior snapshot."""
        clean = self._validate_weights(weights)
        with self.transaction() as connection:
            before = {
                str(model): float(weight)
                for model, weight in connection.execute("SELECT model, weight FROM model_state").fetchall()
            }
            self._apply_weight_map(connection, clean, exact=exact)
        return before

    apply_weights = apply_weight_map

    def create_rollout(
        self,
        *,
        rollout_id: str,
        candidate: str,
        incumbent: str,
        task_class: str,
        calibration_bundle_id: str,
        rubric_bundle_hash: str,
        dataset_hash: str,
        stage: str,
        status: str,
        now: float,
        deadline_at: float | None,
        baseline_weights: Mapping[str, float],
        offline_evidence: Mapping[str, Any],
        reason: str,
        weights: Mapping[str, float],
    ) -> None:
        """Persist the baseline, initial record, weights and audit event atomically."""
        clean = self._validate_weights(weights)
        with self.transaction() as connection:
            if connection.execute("SELECT rollout_id FROM rollout_records WHERE status='active'").fetchone():
                raise RuntimeError("another rollout is active; concurrent global weight changes are forbidden")
            self._apply_weight_map(connection, clean, exact=True)
            connection.execute(
                f"INSERT INTO rollout_baselines (rollout_id, captured_at, weights) VALUES "
                f"({self._ph}, {self._ph}, {self._ph})",
                (rollout_id, now, self._dump(dict(baseline_weights))),
            )
            values = (
                rollout_id, candidate, incumbent, task_class, calibration_bundle_id,
                rubric_bundle_hash, dataset_hash, stage, status, now, now, deadline_at,
                self._dump(dict(baseline_weights)), self._dump(dict(offline_evidence)), reason, now,
            )
            connection.execute(
                "INSERT INTO rollout_records "
                "(rollout_id, candidate, incumbent, task_class, calibration_bundle_id, "
                "rubric_bundle_hash, dataset_hash, stage, status, started_at, stage_started_at, "
                "deadline_at, baseline_weights, offline_evidence, reason, updated_at) VALUES ("
                + ", ".join([self._ph] * 16) + ")",
                values,
            )
            self._insert_rollout_event(
                connection, rollout_id=rollout_id, now=now, elapsed=0.0,
                from_stage="not-started", to_stage=stage, reason=reason, weights=clean,
                details={"offline_evidence": dict(offline_evidence)},
            )

    def _insert_rollout_event(
        self,
        connection: Any,
        *,
        rollout_id: str,
        now: float,
        elapsed: float,
        from_stage: str,
        to_stage: str,
        reason: str,
        weights: Mapping[str, float],
        details: Mapping[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO rollout_events "
            "(rollout_id, timestamp, elapsed_seconds, from_stage, to_stage, reason, weights, details) "
            "VALUES (" + ", ".join([self._ph] * 8) + ")",
            (
                rollout_id, now, elapsed, from_stage, to_stage, reason,
                self._dump(dict(weights)), self._dump(dict(details)),
            ),
        )

    def transition_rollout(
        self,
        *,
        rollout_id: str,
        from_stage: str,
        to_stage: str,
        status: str,
        now: float,
        started_at: float,
        deadline_at: float | None,
        reason: str,
        weights: Mapping[str, float],
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically change SQL weights, rollout stage, and its audit event."""
        clean = self._validate_weights(weights)
        with self.transaction() as connection:
            self._apply_weight_map(connection, clean, exact=True)
            cursor = connection.execute(
                "UPDATE rollout_records SET stage=" + self._ph + ", status=" + self._ph
                + ", stage_started_at=" + self._ph + ", deadline_at=" + self._ph
                + ", reason=" + self._ph + ", updated_at=" + self._ph
                + " WHERE rollout_id=" + self._ph + " AND stage=" + self._ph,
                (to_stage, status, now, deadline_at, reason, now, rollout_id, from_stage),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown or stale rollout stage for {rollout_id!r}")
            self._insert_rollout_event(
                connection, rollout_id=rollout_id, now=now, elapsed=max(0.0, now - started_at),
                from_stage=from_stage, to_stage=to_stage, reason=reason, weights=clean,
                details=details or {},
            )

    def record_rollout_observation(
        self,
        *,
        rollout_id: str,
        stage: str,
        observation_id: str,
        observed_at: float,
        value: float,
        incumbent_value: float | None,
        evidence: Mapping[str, Any],
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO rollout_observations "
                "(observation_id, rollout_id, stage, observed_at, value, incumbent_value, evidence) "
                "VALUES (" + ", ".join([self._ph] * 7) + ")",
                (
                    observation_id, rollout_id, stage, observed_at, value, incumbent_value,
                    self._dump(dict(evidence)),
                ),
            )

    def active_rollout(self) -> dict[str, Any] | None:
        """The single active rollout record (parsed), or None. Used to reconstruct a controller after a restart."""
        with self._lock:
            row = self.connection.execute(
                "SELECT rollout_id, candidate, incumbent, task_class, calibration_bundle_id, rubric_bundle_hash, "
                "dataset_hash, stage, status, started_at, stage_started_at, deadline_at, baseline_weights, "
                "offline_evidence, reason FROM rollout_records WHERE status='active'"
            ).fetchone()
        if row is None:
            return None
        keys = ("rollout_id", "candidate", "incumbent", "task_class", "calibration_bundle_id", "rubric_bundle_hash",
                "dataset_hash", "stage", "status", "started_at", "stage_started_at", "deadline_at", "baseline_weights",
                "offline_evidence", "reason")
        record = dict(zip(keys, row, strict=True))
        record["baseline_weights"] = self._loads(record["baseline_weights"])
        record["offline_evidence"] = self._loads(record["offline_evidence"])
        return record

    def rollout_observation_rows(self, rollout_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT observation_id, stage, observed_at, value, incumbent_value FROM rollout_observations "
                f"WHERE rollout_id={self._ph} ORDER BY observed_at, observation_id",
                (rollout_id,),
            ).fetchall()
        return [{"observation_id": r[0], "stage": r[1], "observed_at": r[2], "value": r[3], "incumbent_value": r[4]} for r in rows]

    def rollout_events(self, rollout_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT timestamp, elapsed_seconds, from_stage, to_stage, reason, weights, details "
                f"FROM rollout_events WHERE rollout_id={self._ph} ORDER BY sequence",
                (rollout_id,),
            ).fetchall()
        return [
            {
                "timestamp": r[0], "elapsed_seconds": r[1], "from_stage": r[2],
                "to_stage": r[3], "reason": r[4], "weights": self._loads(r[5]),
                "details": self._loads(r[6]),
            }
            for r in rows
        ]

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

"""Operational, caller-driven model rollout state machine.

This module changes persisted routing weights; it does not manufacture online measurements.
The caller must feed independently collected, provenance-bound observations through ``observe``
and call ``tick`` (or ``check_deadline``).  There are no provider calls or background threads.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any

from gateway.state import ModelState

from .calibrate import ROUTEBENCH_MIN_KAPPA, CalibrationBundle, validate_bundle

STAGES = ("shadow", "canary-5", "canary-25", "full")
STAGE_WEIGHTS = {
    "shadow": (0.0, 1.0),
    "canary-5": (0.05, 0.95),
    "canary-25": (0.25, 0.75),
    "full": (1.0, 0.0),
}
CANARY_STAGES = frozenset({"canary-5", "canary-25"})


def _serialized(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)
    return call


@dataclass(frozen=True)
class OfflineEvidence:
    """Identity and sample-size evidence required before any weight is changed."""

    candidate: str
    task_class: str
    calibration_bundle_id: str
    rubric_bundle_hash: str
    dataset_hash: str
    n_samples: int
    min_samples: int


@dataclass(frozen=True)
class OnlineObservation:
    """One externally measured candidate quality observation."""

    observation_id: str
    candidate: str
    task_class: str
    calibration_bundle_id: str
    dataset_hash: str
    value: float
    incumbent_value: float | None = None
    rubric_bundle_hash: str | None = None


class RolloutController:
    """Persist and apply a shadow -> 5% -> 25% -> full rollout.

    Traffic is deliberately scoped to the candidate/incumbent cohort: every other registered
    model is assigned weight zero while a rollout is active.  The complete pre-rollout map is
    retained and restored byte-for-value on regression, deadline, or callback exception.
    """

    def __init__(
        self,
        state: ModelState,
        *,
        candidate: str,
        incumbent: str,
        task_class: str,
        calibration: str | Path,
        expected_dataset_hash: str,
        on_weights_changed: Callable[[Mapping[str, float]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        deadline_seconds: float = 90.0,
        min_observations: int = 20,
        max_drop: float = 0.02,
        min_kappa: float = ROUTEBENCH_MIN_KAPPA,
        rollout_id: str | None = None,
    ) -> None:
        if calibration is None:  # type: ignore[comparison-overlap]
            raise ValueError("calibration is required for an operational rollout")
        if not isinstance(expected_dataset_hash, str) or not expected_dataset_hash.strip():
            raise ValueError("expected_dataset_hash is required and must be non-empty")
        if candidate == incumbent:
            raise ValueError("candidate and incumbent must be different registered models")
        if not all(isinstance(v, str) and v.strip() for v in (candidate, incumbent, task_class)):
            raise ValueError("candidate, incumbent and task_class must be non-empty")
        if isinstance(min_observations, bool) or not isinstance(min_observations, int) or min_observations < 20:
            raise ValueError("min_observations must be at least 20")
        if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be finite and positive")
        if not math.isfinite(max_drop) or max_drop < 0:
            raise ValueError("max_drop must be finite and non-negative")
        if (isinstance(min_kappa, bool) or not isinstance(min_kappa, (int, float))
                or not math.isfinite(min_kappa) or min_kappa < ROUTEBENCH_MIN_KAPPA):
            raise ValueError(f"min_kappa cannot be lower than {ROUTEBENCH_MIN_KAPPA}")

        self.state = state
        self.candidate = candidate
        self.incumbent = incumbent
        self.task_class = task_class
        self.calibration = Path(calibration)
        self.expected_dataset_hash = expected_dataset_hash
        self.on_weights_changed = on_weights_changed
        self.clock = clock
        self.deadline_seconds = float(deadline_seconds)
        self.min_observations = int(min_observations)
        self.max_drop = float(max_drop)
        self.min_kappa = float(min_kappa)
        self.rollout_id = rollout_id or str(uuid.uuid4())
        self._lock = threading.RLock()

        self._stage = "not-started"
        self._status = "not-started"
        self._reason = ""
        self._started_at: float | None = None
        self._stage_started_at: float | None = None
        self._deadline_at: float | None = None
        self._baseline_weights: dict[str, float] | None = None
        self._observations: list[tuple[float, float | None]] = []
        self._observation_ids: set[str] = set()
        self._incumbent_quality: float | None = None
        self._bundle: CalibrationBundle | None = None

    # ------------------------------------------------------------------ restart safety
    #: Stored timestamps must be wall-clock (epoch) seconds to be comparable after a restart. A rollout recorded with the default
    #: monotonic clock cannot be resumed safely; `rollback_active` restores its baseline instead.
    MIN_EPOCH = 1_000_000_000.0

    @classmethod
    def resume(
        cls,
        state: ModelState,
        *,
        calibration: str | Path,
        expected_dataset_hash: str,
        clock: Callable[[], float] = time.time,
        on_weights_changed: Callable[[Mapping[str, float]], None] | None = None,
        deadline_seconds: float = 90.0,
        min_observations: int = 20,
        max_drop: float = 0.02,
        min_kappa: float = ROUTEBENCH_MIN_KAPPA,
    ) -> RolloutController:
        """Rebuild the controller for the single active rollout after a process restart, from the persisted records only.

        Everything is re-validated (calibration artifact, identities, registry quality). The current stage's observations and every observation id
        are reloaded from SQL; the router is re-synchronised to the persisted weights; and an expired canary deadline is enforced immediately, so
        downtime can never extend a canary. A rollout whose stored weights no longer match its stage is rolled back to its baseline, and one whose
        timestamps are not wall-clock is not resumed at all (use `rollback_active`).
        """
        record = state.active_rollout()
        if record is None:
            raise RuntimeError("no active rollout to resume")
        if record["dataset_hash"] != expected_dataset_hash:
            raise ValueError("the active rollout was recorded for a different dataset identity")
        now = float(clock())
        stamps = [record["started_at"], record["stage_started_at"]] + ([record["deadline_at"]] if record["deadline_at"] is not None else [])
        if not math.isfinite(now) or now < cls.MIN_EPOCH or any((not math.isfinite(x)) or x < cls.MIN_EPOCH for x in stamps):
            raise ValueError("the rollout timestamps and the resume clock must be wall-clock epoch seconds; use rollback_active to restore the baseline")
        if record["stage_started_at"] > now + 5.0 or record["started_at"] > now + 5.0:
            raise ValueError("the resume clock is earlier than the recorded rollout timestamps; refusing to guess")
        evidence = record["offline_evidence"]
        self = cls(
            state, candidate=record["candidate"], incumbent=record["incumbent"], task_class=record["task_class"],
            calibration=calibration, expected_dataset_hash=expected_dataset_hash, on_weights_changed=on_weights_changed, clock=clock,
            deadline_seconds=deadline_seconds, min_observations=min_observations, max_drop=max_drop, min_kappa=min_kappa,
            rollout_id=record["rollout_id"],
        )
        with self._lock:
            self._validate_start(cls._offline_evidence({**evidence, "candidate": record["candidate"], "task_class": record["task_class"],
                                                        "calibration_bundle_id": record["calibration_bundle_id"],
                                                        "rubric_bundle_hash": record["rubric_bundle_hash"], "dataset_hash": record["dataset_hash"]}))
            self._baseline_weights = {str(k): float(v) for k, v in record["baseline_weights"].items()}
            self._stage, self._status, self._reason = record["stage"], "active", record["reason"]
            self._started_at, self._stage_started_at, self._deadline_at = record["started_at"], record["stage_started_at"], record["deadline_at"]
            rows = state.rollout_observation_rows(self.rollout_id)
            self._observation_ids = {r["observation_id"] for r in rows}
            self._observations = [(r["value"], r["incumbent_value"]) for r in rows if r["stage"] == self._stage and r["observed_at"] >= self._stage_started_at]
            expected = self._weights_for(self._stage)
            current = state.weight_map()
            if {m: round(w, 9) for m, w in current.items()} != {m: round(w, 9) for m, w in expected.items()}:
                self._rollback("persisted weights diverged from the rollout stage at restart; baseline restored")
                return self
            self._notify(current)  # the router starts from its own defaults; bring it to the persisted weights
            self.check_deadline()
        return self

    @staticmethod
    def rollback_active(state: ModelState, *, clock: Callable[[], float] = time.time,
                        on_weights_changed: Callable[[Mapping[str, float]], None] | None = None,
                        reason: str = "operator rollback of an active rollout that cannot be resumed") -> dict[str, Any] | None:
        """Restore the persisted baseline of the active rollout without rebuilding a controller (for non-resumable records). Returns the record, or None."""
        record = state.active_rollout()
        if record is None:
            return None
        baseline = {str(k): float(v) for k, v in record["baseline_weights"].items()}
        now = float(clock())
        state.transition_rollout(rollout_id=record["rollout_id"], from_stage=record["stage"], to_stage="rollback", status="rolled-back", now=now,
                                 started_at=record["started_at"], deadline_at=None, reason=reason, weights=baseline, details={"restored_baseline": True, "operator": True})
        if on_weights_changed is not None:
            on_weights_changed(dict(baseline))
        return record

    @staticmethod
    def _offline_evidence(value: OfflineEvidence | Mapping[str, Any]) -> OfflineEvidence:
        if isinstance(value, OfflineEvidence):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("offline evidence must be OfflineEvidence or a mapping")
        aliases = {
            "candidate": value.get("candidate", value.get("candidate_id")),
            "task_class": value.get("task_class", value.get("task")),
            "calibration_bundle_id": value.get(
                "calibration_bundle_id", value.get("bundle_id", value.get("bundle"))
            ),
            "rubric_bundle_hash": value.get(
                "rubric_bundle_hash", value.get("rubric_hash", value.get("rubric"))
            ),
            "dataset_hash": value.get("dataset_hash", value.get("dataset")),
            "n_samples": value.get("n_samples", value.get("sample_count")),
            "min_samples": value.get("min_samples"),
        }
        try:
            return OfflineEvidence(**aliases)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"incomplete offline evidence: {exc}") from exc

    @staticmethod
    def _online_observation(
        value: OnlineObservation | Mapping[str, Any] | None,
        fields: Mapping[str, Any],
    ) -> OnlineObservation:
        if isinstance(value, OnlineObservation):
            if fields:
                raise TypeError("pass either OnlineObservation or keyword fields, not both")
            return value
        raw: dict[str, Any] = dict(value or {})
        raw.update(fields)
        aliases = {
            "observation_id": raw.get("observation_id", raw.get("id")),
            "candidate": raw.get("candidate", raw.get("candidate_id", raw.get("model"))),
            "task_class": raw.get("task_class", raw.get("task")),
            "calibration_bundle_id": raw.get(
                "calibration_bundle_id", raw.get("bundle_id", raw.get("bundle"))
            ),
            "dataset_hash": raw.get("dataset_hash", raw.get("dataset")),
            "value": raw.get("value", raw.get("quality", raw.get("score", raw.get("candidate_quality")))),
            "incumbent_value": raw.get("incumbent_value", raw.get("incumbent_quality")),
            "rubric_bundle_hash": raw.get("rubric_bundle_hash", raw.get("rubric_hash")),
        }
        try:
            return OnlineObservation(**aliases)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"incomplete online observation: {exc}") from exc

    def _validate_start(self, evidence: OfflineEvidence) -> None:
        ok, reasons = validate_bundle(
            self.calibration,
            min_kappa=self.min_kappa,
            require_headline=True,
            expected_dataset_hash=self.expected_dataset_hash,
        )
        if not ok:
            raise ValueError("invalid calibration artifact: " + "; ".join(reasons))
        bundle = CalibrationBundle.load(self.calibration)
        required = {
            "candidate": (evidence.candidate, self.candidate),
            "task_class": (evidence.task_class, self.task_class),
            "calibration_bundle_id": (evidence.calibration_bundle_id, bundle.bundle_id),
            "rubric_bundle_hash": (evidence.rubric_bundle_hash, bundle.rubric_bundle_hash),
            "dataset_hash": (evidence.dataset_hash, self.expected_dataset_hash),
        }
        mismatches = [name for name, (actual, expected) in required.items() if actual != expected]
        if mismatches:
            raise ValueError(f"offline evidence identity mismatch: {mismatches}")
        if isinstance(evidence.n_samples, bool) or not isinstance(evidence.n_samples, int):
            raise TypeError("offline evidence n_samples must be an integer")
        if isinstance(evidence.min_samples, bool) or not isinstance(evidence.min_samples, int):
            raise TypeError("offline evidence min_samples must be an integer")
        if evidence.min_samples < 20 or evidence.n_samples < evidence.min_samples:
            raise ValueError("offline evidence does not meet its declared minimum sample count")

        matrix = self.state.quality_matrix(
            calibration_bundle_id=bundle.bundle_id,
            rubric_bundle_hash=bundle.rubric_bundle_hash,
            dataset_hash=self.expected_dataset_hash,
            strict_provenance=True,
        )
        missing = [model for model in (self.candidate, self.incumbent) if model not in matrix]
        if missing:
            raise ValueError(f"candidate/incumbent missing matching measured quality: {missing}")
        for model in (self.candidate, self.incumbent):
            if self.task_class not in matrix[model]:
                raise ValueError(f"model {model!r} has no quality for task {self.task_class!r}")

        rows = {row["model"]: row for row in self.state.rows()}
        for model in (self.candidate, self.incumbent):
            samples = rows[model]["n_samples"].get(self.task_class)
            if isinstance(samples, bool) or not isinstance(samples, int) or samples < evidence.min_samples:
                raise ValueError(
                    f"model {model!r} has {samples!r} samples for {self.task_class!r}; "
                    f"at least {evidence.min_samples} required"
                )
        if rows[self.candidate]["n_samples"][self.task_class] < evidence.n_samples:
            raise ValueError("offline evidence claims more candidate samples than the registry records")

        candidate_quality = float(matrix[self.candidate][self.task_class])
        incumbent_quality = float(matrix[self.incumbent][self.task_class])
        if not all(0 <= value <= 1 for value in (candidate_quality, incumbent_quality)):
            raise ValueError("offline quality must be in [0, 1]")
        if candidate_quality < incumbent_quality - self.max_drop:
            raise ValueError(
                f"offline candidate quality {candidate_quality} regresses from incumbent "
                f"{incumbent_quality} by more than {self.max_drop}"
            )
        self._incumbent_quality = incumbent_quality
        self._bundle = bundle

    def _weights_for(self, stage: str) -> dict[str, float]:
        assert self._baseline_weights is not None
        candidate_weight, incumbent_weight = STAGE_WEIGHTS[stage]
        weights = {model: 0.0 for model in self._baseline_weights}
        weights[self.candidate] = candidate_weight
        weights[self.incumbent] = incumbent_weight
        return weights

    def _notify(self, weights: Mapping[str, float]) -> None:
        if self.on_weights_changed is not None:
            self.on_weights_changed(dict(weights))

    @_serialized
    def start(self, offline_evidence: OfflineEvidence | Mapping[str, Any]) -> dict[str, Any]:
        """Validate all evidence, capture the baseline, and enter shadow mode."""
        if self._status != "not-started":
            raise RuntimeError("this rollout controller has already been started")
        evidence = self._offline_evidence(offline_evidence)
        self._validate_start(evidence)
        baseline = self.state.snapshot_weights()
        if not baseline:
            raise ValueError("the model registry is empty")
        self._baseline_weights = baseline
        now = float(self.clock())
        if not math.isfinite(now):
            raise ValueError("clock returned a non-finite timestamp")
        weights = self._weights_for("shadow")
        self.state.create_rollout(
            rollout_id=self.rollout_id,
            candidate=self.candidate,
            incumbent=self.incumbent,
            task_class=self.task_class,
            calibration_bundle_id=evidence.calibration_bundle_id,
            rubric_bundle_hash=evidence.rubric_bundle_hash,
            dataset_hash=evidence.dataset_hash,
            stage="shadow",
            status="active",
            now=now,
            deadline_at=None,
            baseline_weights=baseline,
            offline_evidence=asdict(evidence),
            reason="validated calibration and offline evidence; entered shadow",
            weights=weights,
        )
        self._stage = "shadow"
        self._status = "active"
        self._reason = "validated calibration and offline evidence; entered shadow"
        self._started_at = now
        self._stage_started_at = now
        try:
            self._notify(weights)
        except BaseException as exc:
            self._rollback(f"weight callback failed entering shadow: {type(exc).__name__}: {exc}")
            raise
        return self.status()

    @_serialized
    def observe(
        self,
        observation: OnlineObservation | Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Persist one matching observation for the current window; never advances by itself."""
        self.check_deadline()
        if self._status != "active" or self._stage == "full":
            raise RuntimeError(f"rollout is not accepting observations ({self._status}/{self._stage})")
        item = self._online_observation(observation, fields)
        assert self._bundle is not None
        expected = {
            "candidate": (item.candidate, self.candidate),
            "task_class": (item.task_class, self.task_class),
            "calibration_bundle_id": (item.calibration_bundle_id, self._bundle.bundle_id),
            "dataset_hash": (item.dataset_hash, self.expected_dataset_hash),
        }
        mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if mismatches:
            raise ValueError(f"observation identity mismatch: {mismatches}")
        if item.rubric_bundle_hash is not None and item.rubric_bundle_hash != self._bundle.rubric_bundle_hash:
            raise ValueError("observation rubric_bundle_hash does not match the rollout")
        if not isinstance(item.observation_id, str) or not item.observation_id.strip():
            raise ValueError("observation_id must be a non-empty string")
        if item.observation_id in self._observation_ids:
            raise ValueError(f"duplicate observation_id {item.observation_id!r}")
        if (isinstance(item.value, bool) or not isinstance(item.value, (int, float))
                or not math.isfinite(item.value) or not 0 <= item.value <= 1):
            raise ValueError("observation value must be a finite number in [0, 1]")
        incumbent_value = item.incumbent_value
        if incumbent_value is not None and (
            isinstance(incumbent_value, bool)
            or not isinstance(incumbent_value, (int, float))
            or not math.isfinite(incumbent_value)
            or not 0 <= incumbent_value <= 1
        ):
            raise ValueError("observation incumbent_value must be a finite number when supplied")
        now = float(self.clock())
        self.state.record_rollout_observation(
            rollout_id=self.rollout_id,
            stage=self._stage,
            observation_id=item.observation_id,
            observed_at=now,
            value=float(item.value),
            incumbent_value=None if incumbent_value is None else float(incumbent_value),
            evidence=asdict(item),
        )
        self._observation_ids.add(item.observation_id)
        self._observations.append(
            (float(item.value), None if incumbent_value is None else float(incumbent_value))
        )
        return self.status()

    def _transition(self, to_stage: str, reason: str, details: Mapping[str, Any]) -> None:
        assert self._started_at is not None
        now = float(self.clock())
        deadline = now + self.deadline_seconds if to_stage in CANARY_STAGES else None
        weights = self._weights_for(to_stage)
        from_stage = self._stage
        status = "completed" if to_stage == "full" else "active"
        self.state.transition_rollout(
            rollout_id=self.rollout_id,
            from_stage=from_stage,
            to_stage=to_stage,
            status=status,
            now=now,
            started_at=self._started_at,
            deadline_at=deadline,
            reason=reason,
            weights=weights,
            details=details,
        )
        self._stage = to_stage
        self._status = status
        self._reason = reason
        self._stage_started_at = now
        self._deadline_at = deadline
        self._observations = []
        try:
            self._notify(weights)
        except BaseException as exc:
            self._rollback(
                f"weight callback failed entering {to_stage}: {type(exc).__name__}: {exc}"
            )
            raise

    def _rollback(self, reason: str) -> None:
        if self._baseline_weights is None or self._started_at is None:
            return
        now = float(self.clock())
        from_stage = self._stage
        baseline = dict(self._baseline_weights)
        self.state.transition_rollout(
            rollout_id=self.rollout_id,
            from_stage=from_stage,
            to_stage="rollback",
            status="rolled-back",
            now=now,
            started_at=self._started_at,
            deadline_at=None,
            reason=reason,
            weights=baseline,
            details={"restored_baseline": True},
        )
        self._stage = "rollback"
        self._status = "rolled-back"
        self._reason = reason
        self._stage_started_at = now
        self._deadline_at = None
        self._observations = []
        self._notify(baseline)

    @_serialized
    def tick(self) -> dict[str, Any]:
        """Evaluate the current observation window and advance or restore the baseline."""
        if self.check_deadline():
            return self.status()
        if self._status != "active":
            return self.status()
        if len(self._observations) < self.min_observations:
            return self.status()
        assert self._incumbent_quality is not None
        deltas = [
            candidate - (incumbent if incumbent is not None else self._incumbent_quality)
            for candidate, incumbent in self._observations
        ]
        mean_delta = sum(deltas) / len(deltas)
        details = {
            "observation_count": len(deltas),
            "mean_delta": mean_delta,
            "max_drop": self.max_drop,
        }
        if mean_delta < -self.max_drop:
            self._rollback(
                f"online regression in {self._stage}: mean delta {mean_delta:.6g} "
                f"below {-self.max_drop:.6g}"
            )
            return self.status()
        next_stage = STAGES[STAGES.index(self._stage) + 1]
        self._transition(
            next_stage,
            f"{self._stage} window passed with {len(deltas)} matching observations",
            details,
        )
        return self.status()

    @_serialized
    def check_deadline(self) -> bool:
        """Rollback an expired canary.  Returns True only when it performed that rollback."""
        if self._status != "active" or self._stage not in CANARY_STAGES:
            return False
        assert self._deadline_at is not None
        now = float(self.clock())
        if now < self._deadline_at:
            return False
        self._rollback(
            f"{self._stage} deadline expired after {self.deadline_seconds:g} seconds "
            f"with {len(self._observations)} matching observations"
        )
        return True

    poll = check_deadline

    @_serialized
    def status(self) -> dict[str, Any]:
        now = float(self.clock())
        elapsed = None if self._started_at is None else max(0.0, now - self._started_at)
        stage_elapsed = (
            None if self._stage_started_at is None else max(0.0, now - self._stage_started_at)
        )
        return {
            "rollout_id": self.rollout_id,
            "candidate": self.candidate,
            "incumbent": self.incumbent,
            "task_class": self.task_class,
            "stage": self._stage,
            "status": self._status,
            "reason": self._reason,
            "observation_count": len(self._observations),
            "minimum_observations": self.min_observations,
            "deadline_at": self._deadline_at,
            "elapsed_seconds": elapsed,
            "stage_elapsed_seconds": stage_elapsed,
            "weights": self.state.weight_map(),
            "baseline_weights": None if self._baseline_weights is None else dict(self._baseline_weights),
        }


__all__ = ["STAGES", "STAGE_WEIGHTS", "OfflineEvidence", "OnlineObservation", "RolloutController"]

"""Bounded production sampling and rolling per-model quality.

The previous implementation was a `defaultdict` of `deque`s with a `record`/`quality` pair. As a
rolling mean that was fine; as an online eval loop it was missing everything that makes the
number safe to act on:

*   **A minimum sample count.** `quality()` returned `0.0` for an unseen model and a mean over a
    single observation otherwise. Both feed the router and the promotion gate as though they
    were measurements. One unlucky sample could roll back a good model.
*   **Sampling.** "Sample production traffic" was not implemented; every observation was
    recorded. Judging every request is neither affordable nor necessary.
*   **Error separation.** A failed judgment had nowhere to go except in as a score.
*   **Version binding.** Observations graded by different judge configurations were pooled, so a
    judge change looked like a quality change.
*   **Isolation from the sealed evaluation.** Monitoring samples must never join the frozen
    calibration corpus; if they did, the held-out set would decay into training data.

`OnlineQuality` keeps the original two-method surface so existing callers still work, and adds
the guarantees above around it.
"""
from __future__ import annotations

import hashlib
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from .taxonomy import canonical

#: Below this many graded observations a rolling mean is not reported as a quality measurement.
MIN_OBSERVATIONS = 30


@dataclass(frozen=True)
class Observation:
    model: str
    task_class: str
    score: float
    judge_variant: str
    calibration_bundle_id: str


@dataclass
class OnlineQuality:
    """Thread-safe rolling quality per (model, task_class), bound to one judge version.

    `window` bounds memory and makes the estimate recent. `sample_rate` decides which requests
    are judged at all; sampling is deterministic in the request id so the same request is never
    judged twice by two workers, and so a replay is reproducible.
    """

    window: int = 100
    sample_rate: float = 0.05
    min_observations: int = MIN_OBSERVATIONS
    judge_variant: str = ""
    calibration_bundle_id: str = ""
    scores: dict[tuple[str, str], deque[float]] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    n_sampled: int = 0
    n_seen: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0, 1], got {self.sample_rate}")

    # ---------------- sampling ----------------
    def should_sample(self, request_id: str) -> bool:
        """Deterministic per-request sampling decision.

        Hashing the request id rather than calling `random()` means the decision is stable
        across workers and reproducible from the trace, which is what makes an online quality
        figure auditable after the fact.
        """
        with self._lock:
            self.n_seen += 1
        if self.sample_rate >= 1.0:
            sampled = True
        elif self.sample_rate <= 0.0:
            sampled = False
        else:
            digest = hashlib.sha256(f"online:{request_id}".encode()).digest()
            sampled = (int.from_bytes(digest[:8], "big") / 2**64) < self.sample_rate
        if sampled:
            with self._lock:
                self.n_sampled += 1
        return sampled

    # ---------------- recording ----------------
    def record(
        self,
        model: str,
        score: float,
        task_class: str = "chat",
        *,
        judge_variant: str | None = None,
    ) -> None:
        """Record one graded observation.

        Refuses an observation graded by a different judge than this window is bound to: pooling
        two judges' scores in one rolling mean makes a judge change indistinguishable from a
        quality change.
        """
        if judge_variant is not None and self.judge_variant and judge_variant != self.judge_variant:
            raise ValueError(
                f"observation graded by {judge_variant!r} cannot join a window bound to "
                f"{self.judge_variant!r}; start a new window when the judge changes"
            )
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")
        key = (model, canonical(task_class))
        with self._lock:
            self.scores.setdefault(key, deque(maxlen=self.window)).append(float(score))

    def record_error(self, model: str, status: str = "judge_error") -> None:
        """A failed judgment is counted, never recorded as a score of zero."""
        with self._lock:
            self.errors[f"{model}:{status}"] += 1

    # ---------------- reading ----------------
    def quality(self, model: str, task_class: str = "chat") -> float | None:
        """Rolling mean, or `None` when there is not yet enough evidence.

        `None` rather than `0.0`: a model with no observations is unknown, not bad, and the
        original `0.0` would have made an unobserved model look like the worst available one to
        the router's scoring function.
        """
        key = (model, canonical(task_class))
        with self._lock:
            values = list(self.scores.get(key, ()))
        if len(values) < self.min_observations:
            return None
        return sum(values) / len(values)

    def n(self, model: str, task_class: str = "chat") -> int:
        with self._lock:
            return len(self.scores.get((model, canonical(task_class)), ()))

    def matrix(self) -> dict[str, dict[str, float]]:
        """Only cells with enough observations, in the shape the router consumes."""
        out: dict[str, dict[str, float]] = {}
        with self._lock:
            items = {k: list(v) for k, v in self.scores.items()}
        for (model, task), values in sorted(items.items()):
            if len(values) >= self.min_observations:
                out.setdefault(model, {})[task] = sum(values) / len(values)
        return out

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            items = {k: list(v) for k, v in self.scores.items()}
            errors = dict(self.errors)
            seen, sampled = self.n_seen, self.n_sampled
        return {
            "judge_variant": self.judge_variant,
            "calibration_bundle_id": self.calibration_bundle_id,
            "window": self.window,
            "sample_rate": self.sample_rate,
            "min_observations": self.min_observations,
            "requests_seen": seen,
            "requests_sampled": sampled,
            "errors": errors,
            "cells": {
                f"{model}/{task}": {
                    "n": len(values),
                    "quality": (sum(values) / len(values)) if values else None,
                    "reportable": len(values) >= self.min_observations,
                }
                for (model, task), values in sorted(items.items())
            },
        }

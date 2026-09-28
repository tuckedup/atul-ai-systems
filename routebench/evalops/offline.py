"""Offline quality matrix: real suites, per backend, bound to a calibration bundle.

What changed and why. The previous version was:

    def evaluate(model, suite_path, invoke):
        suite = EvalSuite.load(suite_path)
        result = run(suite, lambda case: invoke(case, model), name=model)
        ... return {tag: mean(scores)}

Three things made its output unsafe to publish as "measured quality":

1.  `EvalSuite.load` silently returned zero cases for a file path, so `by_tag` came back empty
    and the caller wrote an empty matrix without noticing. (Fixed in `aisys.evals`; this module
    additionally refuses an empty or partial run.)
2.  Tags were the suite's spelling (`coding`, `summarization`) while the router looks up
    `code`, `summarize`. Every lookup missed and the router substituted its `0.5` default, so
    the "data-driven" routing policy was running on a constant. Quality is now keyed by
    canonical task class.
3.  A quality number carried no provenance. A score produced by an uncalibrated judge looks
    exactly like one produced by a calibrated judge. `QualityMatrix` therefore carries the
    calibration bundle id, the rubric hash, per-cell sample counts and an error count, and
    `publishable()` refuses cells that rest on too little evidence.

A provider failure and a genuine quality failure are also kept apart. Counting a 503 as a score
of zero silently converts an outage into a quality regression, which the promotion gate would
then act on.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aisys.evals import EvalCase, EvalSuite, run

from .calibrate import CalibrationBundle, validate_bundle
from .taxonomy import UnknownTaskClass, canonical

#: A cell below this many graded cases is reported but not published to the router.
MIN_SAMPLES_PER_CELL = 20


@dataclass
class Cell:
    task_class: str
    quality: float
    n_graded: int
    n_errors: int

    @property
    def publishable(self) -> bool:
        return self.n_graded >= MIN_SAMPLES_PER_CELL


@dataclass
class QualityMatrix:
    """Measured quality plus everything needed to decide whether to trust it."""

    measured_at: str
    calibration_bundle_id: str | None
    rubric_bundle_hash: str | None
    judge_variant: str | None
    cells: dict[str, dict[str, Cell]] = field(default_factory=dict)  # model -> task -> Cell
    provider_errors: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    def router_matrix(self) -> dict[str, dict[str, float]]:
        """The `{model: {task_class: quality}}` shape `gateway/router.py` consumes.

        Only publishable cells are included. A cell with too few samples is omitted rather than
        emitted with a low confidence flag the router would ignore -- the router's own default
        is at least visibly a default.
        """
        return {
            model: {t: c.quality for t, c in tasks.items() if c.publishable}
            for model, tasks in self.cells.items()
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "measured_at": self.measured_at,
            "calibration_bundle_id": self.calibration_bundle_id,
            "rubric_bundle_hash": self.rubric_bundle_hash,
            "judge_variant": self.judge_variant,
            "provider_errors": self.provider_errors,
            "notes": self.notes,
            "cells": {
                m: {t: asdict(c) for t, c in tasks.items()} for m, tasks in self.cells.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return p


class QualityMeasurementError(RuntimeError):
    pass


def evaluate(
    model: str,
    suite_path: str | Path,
    invoke: Callable[[EvalCase, str], dict[str, Any]],
    *,
    concurrency: int = 4,
) -> tuple[dict[str, Cell], int]:
    """Run one suite against one backend. Returns (cells by canonical task, provider errors)."""
    suite = EvalSuite.load(suite_path)  # raises on a missing path or zero cases
    result = run(suite, lambda case: invoke(case, model), name=model, concurrency=concurrency)

    by_task: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    provider_errors = 0
    for item in result.results:
        if item.error:
            # `aisys.evals.run` records score 0.0 alongside the error string. For a quality
            # matrix that zero is not evidence, so the case is excluded and counted instead.
            provider_errors += 1
            for tag in item.tags:
                try:
                    errors[canonical(tag)] += 1
                except UnknownTaskClass:
                    continue
            continue
        for tag in item.tags:
            try:
                by_task[canonical(tag)].append(item.score)
            except UnknownTaskClass as e:
                raise QualityMeasurementError(
                    f"case {item.case_id} carries tag {tag!r} which is not a known task class. "
                    "A tag the router cannot look up becomes a silent 0.5 default, so this is "
                    f"refused rather than skipped. {e}"
                ) from e

    if not by_task:
        raise QualityMeasurementError(
            f"suite {suite_path} produced no graded cases for {model}; refusing to publish an "
            "empty quality matrix"
        )
    return (
        {
            task: Cell(task, sum(scores) / len(scores), len(scores), errors.get(task, 0))
            for task, scores in sorted(by_task.items())
        },
        provider_errors,
    )


def measure(
    backends: dict[str, Callable[[EvalCase, str], dict[str, Any]]],
    suite_path: str | Path,
    *,
    calibration: str | Path | None = None,
    require_calibration: bool = True,
    concurrency: int = 4,
) -> QualityMatrix:
    """Measure every backend over one suite and stamp the result with its judge provenance."""
    bundle_id = rubric_hash = variant = None
    notes = ""
    if calibration is not None:
        ok, reasons = validate_bundle(calibration)
        if not ok and require_calibration:
            raise QualityMeasurementError(
                "refusing to produce a publishable quality matrix from an uncalibrated judge: "
                + "; ".join(reasons)
            )
        b = CalibrationBundle.load(calibration)
        bundle_id, rubric_hash, variant = b.bundle_id, b.rubric_bundle_hash, b.variant_id
        if not ok:
            notes = "CALIBRATION INVALID (measured anyway at caller's request): " + "; ".join(reasons)
    elif require_calibration:
        raise QualityMeasurementError(
            "no calibration artifact supplied. Quality scores come from an LLM judge; without a "
            "valid calibration they are not measurements. Pass require_calibration=False to "
            "produce an explicitly unpublishable matrix."
        )
    else:
        notes = "NOT CALIBRATED: scores are indicative only and must not drive routing."

    matrix = QualityMatrix(
        measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        calibration_bundle_id=bundle_id,
        rubric_bundle_hash=rubric_hash,
        judge_variant=variant,
        notes=notes,
    )
    for model, invoke in sorted(backends.items()):
        cells, provider_errors = evaluate(model, suite_path, invoke, concurrency=concurrency)
        matrix.cells[model] = cells
        matrix.provider_errors[model] = provider_errors
    return matrix


def write_matrix(path: str | Path, matrix: dict[str, dict[str, float]]) -> None:
    """Back-compat writer for the plain `{model: {task: quality}}` shape."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")

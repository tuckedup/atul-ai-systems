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

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aisys.evals import EvalCase, EvalSuite, run

from .calibrate import CalibrationBundle, validate_bundle
from .taxonomy import canonical

#: A cell below this many graded cases is reported but not published to the router.
MIN_SAMPLES_PER_CELL = 20
MIN_CELL_COMPLETION = 0.98


def _file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _suite_hash(path: str | Path) -> str:
    root = Path(path)
    if root.is_file():
        return _file_hash(root)
    files = sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")])
    if not files:
        raise QualityMeasurementError("evaluation suite has no YAML inputs")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass
class Cell:
    task_class: str
    quality: float
    n_graded: int
    n_errors: int

    @property
    def publishable(self) -> bool:
        valid_counts = (
            isinstance(self.n_graded, int) and not isinstance(self.n_graded, bool)
            and isinstance(self.n_errors, int) and not isinstance(self.n_errors, bool)
            and self.n_graded >= MIN_SAMPLES_PER_CELL and self.n_errors >= 0
        )
        valid_score = (isinstance(self.quality, (int, float)) and not isinstance(self.quality, bool)
                       and math.isfinite(self.quality) and 0 <= self.quality <= 1)
        return bool(valid_counts and valid_score
                    and self.n_graded / (self.n_graded + self.n_errors) >= MIN_CELL_COMPLETION)


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
    dataset_hash: str | None = None  # calibration-corpus identity; suite inputs are bound separately
    suite_sha256: str | None = None
    calibration_validated: bool = False
    calibration_artifact: str | None = None
    calibration_artifact_sha256: str | None = None
    fixture_only: bool = False

    def publication_reasons(self) -> list[str]:
        """Revalidate calibration provenance at export time; flags alone are not enough."""
        reasons: list[str] = []
        if self.fixture_only:
            reasons.append("fixture-only evaluation cannot publish routing quality")
        if not self.calibration_validated:
            reasons.append("calibration was not validated")
        if not all((self.calibration_bundle_id, self.rubric_bundle_hash, self.judge_variant,
                    self.dataset_hash, self.suite_sha256, self.calibration_artifact,
                    self.calibration_artifact_sha256)):
            reasons.append("calibration, dataset or suite binding is incomplete")
            return reasons
        artifact = Path(self.calibration_artifact or "")
        if not artifact.is_file():
            reasons.append("bound calibration artifact is absent")
            return reasons
        if _file_hash(artifact) != self.calibration_artifact_sha256:
            reasons.append("calibration artifact changed after quality measurement")
            return reasons
        ok, gate_reasons = validate_bundle(artifact, expected_dataset_hash=self.dataset_hash)
        if not ok:
            reasons.extend(gate_reasons)
            return reasons
        bundle = CalibrationBundle.load(artifact)
        if (bundle.bundle_id, bundle.rubric_bundle_hash, bundle.dataset_hash, bundle.variant_id) != (
                self.calibration_bundle_id, self.rubric_bundle_hash, self.dataset_hash, self.judge_variant):
            reasons.append("matrix provenance differs from its calibration artifact")
        return reasons

    def router_matrix(self) -> dict[str, dict[str, float]]:
        """The `{model: {task_class: quality}}` shape `gateway/router.py` consumes.

        Only publishable cells are included. A cell with too few samples is omitted rather than
        emitted with a low confidence flag the router would ignore -- the router's own default
        is at least visibly a default.
        """
        reasons = self.publication_reasons()
        if reasons:
            raise QualityMeasurementError("quality matrix is unpublishable: " + "; ".join(reasons))
        result: dict[str, dict[str, float]] = {}
        for model, tasks in self.cells.items():
            cells = {canonical(t): c.quality for t, c in tasks.items()
                     if c.publishable and canonical(c.task_class) == canonical(t)}
            if cells:
                result[model] = cells
        if not result:
            raise QualityMeasurementError("quality matrix has no complete, adequately sampled cells")
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "measured_at": self.measured_at,
            "calibration_bundle_id": self.calibration_bundle_id,
            "rubric_bundle_hash": self.rubric_bundle_hash,
            "judge_variant": self.judge_variant,
            "provider_errors": self.provider_errors,
            "notes": self.notes,
            "schema_version": 2,
            "dataset_hash": self.dataset_hash,
            "suite_sha256": self.suite_sha256,
            "calibration_validated": self.calibration_validated,
            "calibration_artifact": self.calibration_artifact,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "fixture_only": self.fixture_only,
            "cells": {
                m: {t: asdict(c) for t, c in tasks.items()} for m, tasks in self.cells.items()
            },
        }

    @classmethod
    def load(cls, path: str | Path) -> QualityMatrix:
        data = json.loads(Path(path).read_text(encoding="utf8"))
        if data.pop("schema_version", None) != 2:
            raise QualityMeasurementError("routing-quality artifact lacks current provenance schema")
        data["cells"] = {model: {task: Cell(**cell) for task, cell in tasks.items()}
                         for model, tasks in data.get("cells", {}).items()}
        return cls(**data)

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
    if any(case.grader in {"llm_judge", "unit_tests"} for case in suite.cases):
        raise QualityMeasurementError(
            "generic llm_judge/unit_tests graders are not the structured calibrated judge "
            "or fixture-pinned sandbox adapter; refusing mismatched grading before any backend call"
        )
    # Validate/canonicalize before invoking a model, and do not double-count two
    # aliases of the same task on one result as two independent examples.
    for case in suite.cases:
        if not case.tags:
            raise QualityMeasurementError(f"case {case.id} has no declared task class")
        for tag in case.tags:
            canonical(tag)
    result = run(suite, lambda case: invoke(case, model), name=model, concurrency=concurrency)

    by_task: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    provider_errors = 0
    for item in result.results:
        tasks = {canonical(tag) for tag in item.tags}
        if item.error:
            # `aisys.evals.run` records score 0.0 alongside the error string. For a quality
            # matrix that zero is not evidence, so the case is excluded and counted instead.
            provider_errors += 1
            for task in tasks:
                errors[task] += 1
            continue
        for task in tasks:
            by_task[task].append(item.score)

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
    dataset_hash = None
    validated = False
    artifact_path = artifact_hash = None
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
        dataset_hash = b.dataset_hash
        validated = ok
        artifact_path = str(Path(calibration).resolve())
        artifact_hash = _file_hash(calibration)
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
        dataset_hash=dataset_hash,
        suite_sha256=_suite_hash(suite_path),
        calibration_validated=validated,
        calibration_artifact=artifact_path,
        calibration_artifact_sha256=artifact_hash,
        fixture_only="fixtures" in {part.lower() for part in Path(suite_path).resolve().parts},
    )
    for model, invoke in sorted(backends.items()):
        cells, provider_errors = evaluate(model, suite_path, invoke, concurrency=concurrency)
        matrix.cells[model] = cells
        matrix.provider_errors[model] = provider_errors
    return matrix


def write_matrix(path: str | Path, matrix: dict[str, dict[str, float]]) -> None:
    """Plain floats lack the evidence required to publish a routing matrix."""
    raise QualityMeasurementError("plain matrices have no provenance; use QualityMatrix.save instead")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a bound quality artifact; never starts provider calls")
    parser.add_argument("--check-artifact", type=Path)
    args = parser.parse_args(argv)
    if args.check_artifact is None:
        print("BLOCKED: live measurement requires explicit backend adapters, a real suite and a valid "
              "calibration artifact. Use measure() with those dependencies; no model was called.")
        return 2
    try:
        value = QualityMatrix.load(args.check_artifact)
        published = value.router_matrix()
    except (OSError, ValueError, TypeError, QualityMeasurementError) as error:
        print(f"BLOCKED: {error}")
        return 2
    print(json.dumps({"publishable": True, "models": sorted(published),
                      "calibration_bundle_id": value.calibration_bundle_id,
                      "dataset_hash": value.dataset_hash, "suite_sha256": value.suite_sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

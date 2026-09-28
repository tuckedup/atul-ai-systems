"""Regression tests for two holes an independent review found in the RouteBench release gate.

Defect 1 -- `validate_bundle` reintroduced exactly the `NaN < threshold is False` defect that
`docs/KAPPA_DESIGN.md` Sec 2.3 describes and that `Agreement.passes()` was written to avoid: a
track whose kappa is NaN, +-inf, or outside [-1, 1] passed the gate because
`float("nan") < min_kappa` is `False`, so the rejection branch never fired. The artifact's own
`defined` / `passes_min_kappa` booleans were also trusted outright, even though they are
self-reported by whatever produced the file.

Defect 2 -- a missing `completion` block is falsy, so `if comp and comp.get(...)` skipped the
coverage check entirely: deleting the block outright made the gate pass.

Defect 3 -- overall completion can be 100% while the primary track is badly covered, because
nothing checked per-track coverage at all.

Artifacts are built as plain dicts and written to `tmp_path` as JSON, exactly as a real
`CalibrationBundle.save()` would leave them on disk, so these tests exercise the same code path
`validate_bundle` uses in production (`CalibrationBundle.load` + `json.loads`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from evalops.calibrate import validate_bundle
from evalops.rubrics import bundle_hash

PRIMARY = "human_judgment_of_response"


def _track(**overrides: object) -> dict[str, object]:
    """A minimal well-formed, passing track record."""
    base: dict[str, object] = {
        "track": PRIMARY,
        "insufficient": False,
        "n": 200,
        "defined": True,
        "kappa": 0.74,
        "observed_agreement": 0.9,
        "passes_min_kappa": True,
        "n_labelled": 200,
        "track_completion": 1.0,
        "reason": "",
    }
    base.update(overrides)
    return base


def _bundle(**overrides: object) -> dict[str, object]:
    """A minimal well-formed, passing calibration artifact so each test can break one field."""
    base: dict[str, object] = {
        "bundle_id": "b1",
        "created_at": "2026-09-27T00:00:00+00:00",
        "variant_id": "v1",
        "judge_config": {"model": "test"},
        "threshold": 0.5,
        "min_kappa": 0.74,
        "rubric_bundle_hash": bundle_hash(),
        "dataset_hash": "dh",
        "split_seed": 1,
        "dev_kappa": 0.8,
        "dev_n": 400,
        "selection_rule": "r",
        "threshold_grid": [],
        "label_provenance_counts": {"human_local_annotation": 200},
        "test_result": {
            "tracks": {PRIMARY: _track()},
            "completion": {
                "paired": 200, "unjudged": 0, "completion_rate": 1.0, "judge_errors": {},
            },
        },
    }
    base.update(overrides)
    return base


def _write(tmp_path: Path, payload: dict[str, object], name: str = "calibration_bundle.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    return p


# ---------------------------------------------------------------- Defect 1: nonfinite / out-of-range kappa


def test_nan_kappa_rejects(tmp_path):
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY]["kappa"] = float("nan")  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("not a finite number" in r for r in reasons)


def test_positive_infinity_kappa_rejects(tmp_path):
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY]["kappa"] = float("inf")  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("not a finite number" in r for r in reasons)


def test_negative_infinity_kappa_rejects(tmp_path):
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY]["kappa"] = float("-inf")  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("not a finite number" in r for r in reasons)


def test_kappa_above_one_rejects(tmp_path):
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY]["kappa"] = 1.5  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("outside the valid range" in r for r in reasons)


def test_kappa_below_negative_one_rejects(tmp_path):
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY]["kappa"] = -2.0  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("outside the valid range" in r for r in reasons)


def test_nan_less_than_threshold_trap_is_documented_and_the_gate_still_rejects(tmp_path):
    """Pins the exact defect from docs/KAPPA_DESIGN.md Sec 2.3: `nan < threshold` is False, so a
    gate written as `if kappa < min_kappa: reject` lets NaN straight through. Confirm the trap
    is real, then confirm the gate rejects the NaN artifact anyway."""
    assert not (float("nan") < 0.74)  # noqa: PLW0177 - the trap itself is the point of this test
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY].update(  # type: ignore[index]
        {"kappa": float("nan"), "defined": True, "passes_min_kappa": True}
    )
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert reasons


def test_defined_false_rejects_regardless_of_a_finite_passing_kappa(tmp_path):
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY].update(  # type: ignore[index]
        {"kappa": 0.95, "defined": False}
    )
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("undefined" in r for r in reasons)


def test_passes_min_kappa_true_contradicting_a_failing_kappa_rejects_and_names_the_inconsistency(tmp_path):
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY].update(  # type: ignore[index]
        {"kappa": 0.5, "defined": True, "passes_min_kappa": True}
    )
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("kappa 0.5 is below the 0.74 policy" in r for r in reasons)
    assert any("self-inconsistent" in r for r in reasons)


# ---------------------------------------------------------------- Defect 2: completion metadata


def test_missing_completion_block_rejects(tmp_path):
    payload = _bundle()
    del payload["test_result"]["completion"]  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("no completion metadata" in r for r in reasons)


def test_completion_as_a_list_rejects(tmp_path):
    payload = _bundle()
    payload["test_result"]["completion"] = ["not", "a", "mapping"]  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("no completion metadata" in r for r in reasons)


def test_empty_completion_dict_rejects(tmp_path):
    payload = _bundle()
    payload["test_result"]["completion"] = {}  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("no completion metadata" in r for r in reasons)


@pytest.mark.parametrize("bad_rate", [None, "1.0", 1.5, -0.1, float("nan")])
def test_invalid_completion_rate_rejects(tmp_path, bad_rate):
    payload = _bundle()
    payload["test_result"]["completion"] = {  # type: ignore[index]
        "paired": 200, "unjudged": 0, "completion_rate": bad_rate, "judge_errors": {},
    }
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("completion_rate" in r for r in reasons)


def test_completion_rate_inconsistent_with_paired_and_unjudged_rejects_and_names_both(tmp_path):
    payload = _bundle()
    payload["test_result"]["completion"] = {  # type: ignore[index]
        "paired": 100, "unjudged": 50, "completion_rate": 0.9, "judge_errors": {},
    }
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("paired=100" in r and "unjudged=50" in r for r in reasons)


def test_paired_plus_unjudged_zero_rejects(tmp_path):
    payload = _bundle()
    payload["test_result"]["completion"] = {  # type: ignore[index]
        "paired": 0, "unjudged": 0, "completion_rate": 0.0, "judge_errors": {},
    }
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("no labelled test items" in r for r in reasons)


def test_completion_rate_098_rejects_under_new_default_of_1_0(tmp_path):
    payload = _bundle()
    payload["test_result"]["completion"] = {  # type: ignore[index]
        "paired": 196, "unjudged": 4, "completion_rate": 0.98, "judge_errors": {},
    }
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("automated decision" in r for r in reasons)


def test_completion_rate_098_passes_when_min_completion_098_passed_explicitly(tmp_path):
    payload = _bundle()
    payload["test_result"]["completion"] = {  # type: ignore[index]
        "paired": 196, "unjudged": 4, "completion_rate": 0.98, "judge_errors": {},
    }
    payload["test_result"]["tracks"][PRIMARY]["track_completion"] = 0.98  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload), min_completion=0.98)
    assert ok is True, reasons


# ---------------------------------------------------------------- Defect 3: per-track coverage


def test_primary_track_completion_below_threshold_rejects_even_with_perfect_overall_completion(tmp_path):
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY].update(  # type: ignore[index]
        {"track_completion": 0.5, "n_labelled": 400}
    )
    # Overall completion is still perfect: the primary track's own shortfall must be caught
    # even though it would be invisible in the top-level number.
    assert payload["test_result"]["completion"]["completion_rate"] == 1.0  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("coverage" in r and PRIMARY in r for r in reasons)


def test_artifact_with_no_per_track_coverage_fields_rejects_by_default(tmp_path):
    payload = _bundle()
    track = payload["test_result"]["tracks"][PRIMARY]  # type: ignore[index]
    del track["track_completion"]
    del track["n_labelled"]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("no per-track coverage" in r for r in reasons)


def test_track_completion_wrong_type_rejects(tmp_path):
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY]["track_completion"] = "1.0"  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("track_completion" in r for r in reasons)


# ---------------------------------------------------------------- must never raise


def test_test_result_as_a_list_returns_false_rather_than_raising(tmp_path):
    payload = _bundle(test_result=[1, 2, 3])
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert reasons


def test_tracks_as_a_string_returns_false_rather_than_raising(tmp_path):
    payload = _bundle()
    payload["test_result"]["tracks"] = "garbage"  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert reasons


def test_track_value_not_a_mapping_returns_false_rather_than_raising(tmp_path):
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY] = "garbage"  # type: ignore[index]
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert reasons


# ---------------------------------------------------------------- the well-formed baseline


def test_fully_well_formed_bundle_passes_with_empty_reasons(tmp_path):
    ok, reasons = validate_bundle(_write(tmp_path, _bundle()))
    assert ok is True
    assert reasons == []


def test_kappa_just_below_policy_fails(tmp_path):
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY].update(  # type: ignore[index]
        {"kappa": 0.7399, "passes_min_kappa": False}
    )
    ok, reasons = validate_bundle(_write(tmp_path, payload))
    assert ok is False
    assert any("0.7399" in r for r in reasons)


def test_json_round_trip_of_nan_kappa_reproduces_the_defect_scenario(tmp_path):
    """`json.dumps` writes a bare `NaN` token for a NaN float, and `json.loads` reads it back as
    `float("nan")`, so this is exactly how a live artifact on disk reaches the gate -- not just
    an in-memory construction."""
    payload = _bundle()
    payload["test_result"]["tracks"][PRIMARY].update(  # type: ignore[index]
        {"kappa": float("nan"), "defined": True, "passes_min_kappa": True}
    )
    p = _write(tmp_path, payload)
    raw = p.read_text(encoding="utf-8")
    assert "NaN" in raw
    reloaded = json.loads(raw)
    assert math.isnan(reloaded["test_result"]["tracks"][PRIMARY]["kappa"])
    ok, _reasons = validate_bundle(p)
    assert ok is False

from __future__ import annotations

import math

import pytest
from aisys.evals import calibrate
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]


def _table(tp: int, tn: int, fp: int, fn: int) -> tuple[list[float], list[int]]:
    """Build (judge_scores, human_labels) whose 2x2 confusion table has the given counts.

    judge_threshold stays at the default 0.5, so a judge score of 1.0 means "judge says positive"
    and 0.0 means "judge says negative".
    """
    judge = [1.0] * tp + [0.0] * fn + [1.0] * fp + [0.0] * tn
    human = [1] * tp + [1] * fn + [0] * fp + [0] * tn
    return judge, human


def test_hand_computed_kappa_matches_sklearn_cross_check() -> None:
    judge, human = _table(tp=87, tn=87, fp=13, fn=13)
    report = calibrate(judge, human)
    assert report["kappa"] == pytest.approx(0.74, abs=1e-9)
    assert report["defined"] is True
    # cross-check against the previous sklearn-based implementation
    assert cohen_kappa_score(human, [int(s >= 0.5) for s in judge]) == pytest.approx(0.74, abs=1e-9)


def test_perfect_agreement_both_classes_present_is_trusted() -> None:
    judge, human = _table(tp=10, tn=10, fp=0, fn=0)
    report = calibrate(judge, human, min_kappa=0.74)
    assert report["kappa"] == pytest.approx(1.0)
    assert report["trusted"] is True


def test_single_class_kappa_is_undefined_and_fails_closed() -> None:
    # every human label (and every judge call) is positive: the old `nan < threshold` idiom
    # evaluates to False and would have let this pass a "reject below 0.6" gate.
    judge = [1.0] * 20
    human = [1] * 20
    report = calibrate(judge, human)
    assert report["kappa"] is None
    assert report["defined"] is False
    assert report["trusted"] is False
    assert not report["trusted"]  # old buggy gate `report["kappa"] < 0.6` would NOT have caught this
    assert report["reason"]


def test_judge_threshold_never_changes_human_marginal() -> None:
    """Regression test for defect 1: human labels must never be re-thresholded."""
    judge = [0.1, 0.3, 0.4, 0.6, 0.7, 0.9, 0.55, 0.2, 0.8, 0.65]
    human = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    expected_prevalence = sum(human) / len(human)
    expected_positive_total = sum(human)
    for threshold in (0.0, 0.1, 0.25, 0.5, 0.51, 0.75, 0.9, 1.0):
        report = calibrate(judge, human, judge_threshold=threshold)
        assert report["human_prevalence"] == pytest.approx(expected_prevalence)
        tp_plus_fn = report["confusion"]["tp"] + report["confusion"]["fn"]
        assert tp_plus_fn == expected_positive_total


@pytest.mark.parametrize(
    "judge,human",
    [
        ([0.1, 0.2], [1]),  # length mismatch
        ([], []),  # empty
        ([0.1, 0.2], [0.5, 1]),  # human label 0.5
        ([0.1, 0.2], [2, 1]),  # human label 2
        ([1.5, 0.2], [1, 0]),  # judge score > 1
        ([-0.1, 0.2], [1, 0]),  # judge score < 0
        ([float("nan"), 0.2], [1, 0]),  # judge score nan
        ([float("inf"), 0.2], [1, 0]),  # judge score inf
        ([0.1, 0.2], [None, 1]),  # human label None
    ],
)
def test_malformed_input_raises_value_error(judge: list[float], human: list[object]) -> None:
    with pytest.raises(ValueError):
        calibrate(judge, human)  # type: ignore[arg-type]


def test_min_kappa_boundary_exact_pass_and_near_miss_fail() -> None:
    judge_pass, human_pass = _table(tp=87, tn=87, fp=13, fn=13)  # kappa == 0.74 exactly
    report_pass = calibrate(judge_pass, human_pass, min_kappa=0.74)
    assert report_pass["kappa"] == pytest.approx(0.74, abs=1e-9)
    assert report_pass["trusted"] is True

    judge_fail, human_fail = _table(tp=86, tn=87, fp=13, fn=14)  # kappa == 0.73, a near miss
    report_fail = calibrate(judge_fail, human_fail, min_kappa=0.74)
    assert report_fail["kappa"] == pytest.approx(0.73, abs=1e-9)
    assert report_fail["kappa"] < 0.74
    assert report_fail["trusted"] is False


def test_judge_threshold_and_min_kappa_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        calibrate([0.1, 0.9], [0, 1], 0.5)  # type: ignore[misc]
    with pytest.raises(TypeError):
        calibrate([0.1, 0.9], [0, 1], 0.5, 0.6)  # type: ignore[misc]


def test_bool_human_labels_are_accepted() -> None:
    report = calibrate([1.0, 0.0, 1.0, 0.0], [True, False, True, False])
    assert report["kappa"] == pytest.approx(1.0)
    assert math.isfinite(report["kappa"])

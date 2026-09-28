import math

import pytest
from evalops.metrics import (
    Confusion,
    MetricInputError,
    agreement,
    binarize,
    confusion_of,
    kappa_of,
)
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------- closed-form kappa


def test_kappa_arithmetic_fixture_exact():
    c = Confusion(tp=87, tn=87, fp=13, fn=13)
    kappa, po, pe, reason = kappa_of(c)
    assert kappa == pytest.approx(0.74, abs=1e-9)
    assert reason == ""
    # Cross-check against sklearn so the hand-rolled closed form cannot drift.
    human = [1] * 100 + [0] * 100
    judge = [1] * 87 + [0] * 13 + [1] * 13 + [0] * 87
    assert confusion_of(human, judge) == c
    assert cohen_kappa_score(human, judge) == pytest.approx(kappa, abs=1e-9)


def test_perfect_agreement_both_classes_present():
    human = [1, 1, 1, 0, 0, 0]
    judge = [1, 1, 1, 0, 0, 0]
    a = agreement(human, judge, bootstrap=0)
    assert a.defined
    assert a.kappa == pytest.approx(1.0)


def test_perfect_disagreement_is_negative():
    human = [1, 1, 1, 0, 0, 0]
    judge = [0, 0, 0, 1, 1, 1]
    a = agreement(human, judge, bootstrap=0)
    assert a.defined
    assert a.kappa < 0


# ---------------------------------------------------------------- the degeneracy defect


def test_all_positive_is_undefined_not_nan():
    # This is the regression test for the live bug the module docstring describes: sklearn's
    # cohen_kappa_score returns nan for a constant, single-class table, and `nan < threshold`
    # is False, so a naive `if kappa < min_kappa: reject` gate lets it straight through.
    human = [1] * 20
    judge = [1] * 20
    a = agreement(human, judge, bootstrap=0)
    assert a.kappa is None
    assert a.defined is False
    assert a.passes(0.6) is False
    # Demonstrate the old idiom would have been fooled by nan on this exact table.
    sk = cohen_kappa_score(human, judge)
    assert math.isnan(sk)
    assert not (sk < 0.6)
    assert not (sk >= 0.6)


def test_all_negative_is_undefined_not_nan():
    human = [0] * 20
    judge = [0] * 20
    a = agreement(human, judge, bootstrap=0)
    assert a.kappa is None
    assert a.defined is False
    assert a.passes(0.6) is False
    sk = cohen_kappa_score(human, judge)
    assert math.isnan(sk)
    assert not (sk < 0.6)


# ---------------------------------------------------------------- input validation


def test_length_mismatch_raises():
    with pytest.raises(MetricInputError, match="length mismatch"):
        confusion_of([1, 0, 1], [1, 0])


def test_empty_input_raises():
    with pytest.raises(MetricInputError, match="no labelled items"):
        confusion_of([], [])


@pytest.mark.parametrize("bad", [0.5, 2, None])
def test_non_binary_human_label_raises(bad):
    with pytest.raises(MetricInputError, match="binary 0/1 labels are required"):
        confusion_of([bad], [1])


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), float("inf")])
def test_binarize_rejects_out_of_range_or_nonfinite(bad):
    with pytest.raises(MetricInputError):
        binarize([bad], 0.5)


def test_binarize_threshold_boundary_is_pass():
    # A score exactly at the threshold is a PASS (>=), not a FAIL.
    assert binarize([0.5], 0.5) == [1]
    assert binarize([0.4999999], 0.5) == [0]


# ---------------------------------------------------------------- group bootstrap


def _mixed_sample(n_groups=12, per_group=10, seed=1):
    import random

    rng = random.Random(seed)
    groups, human, judge = [], [], []
    for g in range(n_groups):
        base = rng.random()
        for _ in range(per_group):
            h = 1 if rng.random() < base else 0
            j = h if rng.random() > 0.15 else 1 - h
            groups.append(f"g{g}")
            human.append(h)
            judge.append(j)
    return human, judge, groups


def test_bootstrap_unusable_below_ten_groups():
    groups = [f"g{i % 8}" for i in range(80)]
    human = [1] * 40 + [0] * 40
    judge = [1] * 35 + [0] * 5 + [1] * 10 + [0] * 30
    a = agreement(human, judge, groups=groups, bootstrap=500)
    assert a.interval.usable is False
    assert "n_groups=8" in a.interval.note


def test_bootstrap_usable_with_many_groups_and_brackets_kappa():
    human, judge, groups = _mixed_sample(n_groups=12, per_group=10)
    a = agreement(human, judge, groups=groups, bootstrap=1000, seed=7)
    assert a.interval.usable is True
    assert a.interval.low <= a.kappa <= a.interval.high


def test_bootstrap_deterministic_by_seed():
    human, judge, groups = _mixed_sample(n_groups=12, per_group=10)
    a1 = agreement(human, judge, groups=groups, bootstrap=1000, seed=42)
    a2 = agreement(human, judge, groups=groups, bootstrap=1000, seed=42)
    assert a1.interval.low == a2.interval.low
    assert a1.interval.high == a2.interval.high

    a3 = agreement(human, judge, groups=groups, bootstrap=1000, seed=999)
    assert a3.interval.low <= a3.kappa <= a3.interval.high


def test_grouping_widens_the_interval():
    # Pins the reason the bootstrap resamples groups rather than items: 500 items really only
    # carry 10 independent units of evidence (bimodal group base rates, low within-group
    # noise). Treating every item as its own group understates the true uncertainty.
    import random

    rng = random.Random(0)
    groups, human, judge = [], [], []
    for g in range(10):
        base = rng.choice([0.1, 0.9])
        for _ in range(50):
            h = 1 if rng.random() < base else 0
            j = h if rng.random() > 0.1 else 1 - h
            groups.append(f"g{g}")
            human.append(h)
            judge.append(j)

    grouped = agreement(human, judge, groups=groups, bootstrap=2000, seed=42)
    ungrouped = agreement(
        human, judge, groups=[f"i{i}" for i in range(len(human))], bootstrap=2000, seed=42
    )
    grouped_width = grouped.interval.high - grouped.interval.low
    ungrouped_width = ungrouped.interval.high - ungrouped.interval.low
    assert grouped_width > ungrouped_width


# ---------------------------------------------------------------- per-task breakdown


def test_per_task_breakdown_counts_and_small_stratum_note():
    human = [1] * 30 + [0] * 30 + [1] * 10 + [0] * 5
    judge = [1] * 28 + [0] * 2 + [1] * 4 + [0] * 26 + [1] * 9 + [0] * 1 + [1] * 2 + [0] * 3
    tasks = ["big"] * 60 + ["small"] * 15
    a = agreement(human, judge, tasks=tasks, bootstrap=0)
    assert a.per_task["big"]["n"] == 60
    assert a.per_task["small"]["n"] == 15
    assert "indicative only" in a.per_task["small"]["reason"]
    assert a.per_task["big"]["reason"] == ""

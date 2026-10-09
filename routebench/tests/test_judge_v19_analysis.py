"""Tests for the judge-v19 Phase 1 analysis definitions (synthetic data only; no provider, no GPU, no labels)."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from evalops import judge_v19_analysis as an
from evalops import matched_v16_v18 as mv


def mean_threshold_vote(scores, threshold=0.5):
    """The definition the review ruled out for A1: average raw scores and test the mean. Kept here only to prove that it differs."""
    return int(sum(scores) / len(scores) >= threshold)


# -------------------------------------------------------------------------------------------------------------- A0 / A1 / A2 definitions
def test_the_original_pass_threshold_is_unchanged():
    assert an.PASS_THRESHOLD == mv.PASS_THRESHOLD == 1.0


def test_five_rejections_at_0_8_are_rejected_by_a1_but_would_pass_a_mean_threshold():
    scores = [0.8] * 5
    assert [an.verdict(s) for s in scores] == [0] * 5
    assert an.a1_majority(scores) == 0
    assert mean_threshold_vote(scores) == 1  # the loosened rubric the review warned about
    assert an.a1_majority(scores) != mean_threshold_vote(scores)


def test_a1_is_a_majority_of_original_verdicts_not_of_scores():
    assert an.a1_majority([1.0, 1.0, 1.0, 0.0, 0.0]) == 1
    assert an.a1_majority([1.0, 1.0, 0.9, 0.9, 0.9]) == 0  # two passes: reject, though the mean score is 0.94
    assert mean_threshold_vote([1.0, 1.0, 0.9, 0.9, 0.9]) == 1


def test_a1_and_a_mean_threshold_disagree_on_a_whole_grid_of_cases_and_never_in_the_loosening_direction_for_a1():
    levels = (0.0, 0.5, 0.8, 1.0)
    disagreements = 0
    for scores in itertools.product(levels, repeat=an.K):
        a1, m = an.a1_majority(scores), mean_threshold_vote(scores)
        if a1 != m:
            disagreements += 1
        assert not (a1 == 1 and m == 0)  # majority-of-passes implies a mean of at least 0.6
    assert disagreements > 100


def test_a1_requires_exactly_five_valid_scores():
    for bad in ([1.0] * 4, [1.0] * 6):
        with pytest.raises(an.AnalysisError):
            an.a1_majority(bad)
    for bad_score in (float("nan"), 1.5, -0.1, True, None):
        with pytest.raises(an.AnalysisError):
            an.verdict(bad_score)


def test_a2_keeps_raw_scores_as_features_so_it_differs_from_a1():
    row = an.feature_row([0.8] * 5, 0.5, [0.5], "The council met on 3 May.", "The council met on Monday.")
    assert row[0] == pytest.approx(0.8) and row[1] == 0.0 and row[2] == pytest.approx(0.0)  # raw mean, all-pass fraction, sd
    varied = an.feature_row([1.0, 0.2, 0.4, 1.0, 0.6], 0.5, [0.5], "x", "y")
    assert varied[0] == pytest.approx(0.64) and varied[1] == pytest.approx(0.4) and varied[2] > 0.3


def test_feature_schema_has_nine_named_features_and_the_row_matches():
    row = an.feature_row([1.0] * 5, 0.9, [0.9, 0.2], "It did not rise by 12%.", "source text")
    assert len(an.FEATURES) == len(row) == 9
    assert row[4] == pytest.approx(math.log(0.2 / 0.8)) and row[7] == 2.0 and row[8] == 1.0
    assert an.feature_row([1.0] * 5, 0.9, [0.9], "It rose.", "s")[8] == 0.0
    with pytest.raises(an.AnalysisError):
        an.feature_row([1.0] * 4, 0.9, [0.9], "x", "y")


# -------------------------------------------------------------------------------------------------------------- nested CV
def synthetic(n_groups=40, per=6, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    groups, y, rows = [], [], []
    for g in range(n_groups):
        for i in range(per):
            label = int((g * per + i) % 3 != 0)
            groups.append(f"g{g}")
            y.append(label)
            rows.append(rng.normal(size=len(an.FEATURES)) + signal * (label - 0.5) * np.eye(len(an.FEATURES))[0])
    return np.array(rows), np.array(y), groups


def test_fold_assignment_is_deterministic_and_depends_only_on_the_group():
    assert an.fold_of("outer", "g1", 5) == an.fold_of("outer", "g1", 5)
    assert {an.fold_of("outer", f"g{i}", 5) for i in range(200)} == set(range(5))
    assert an.fold_of("outer", "g1", 5) in range(5) and an.fold_of("inner|0", "g1", 4) in range(4)


def test_nested_oof_never_scores_a_row_with_a_model_that_saw_its_group(monkeypatch):
    X, y, groups = synthetic()
    seen = []
    real = an._fit

    def spy(Xt, yt, c):
        seen.append((len(Xt), c))
        return real(Xt, yt, c)

    monkeypatch.setattr(an, "_fit", spy)
    out = an.nested_oof(X, y, groups)
    outer = np.array([an.fold_of("outer", g, an.OUTER_FOLDS) for g in groups])
    assert len(out["oof"]) == len(y) and not np.isnan(out["oof"]).any()
    assert all(c in an.C_GRID for c in out["chosen_c_per_outer_fold"].values()) and sum(out["outer_fold_sizes"]) == len(y)
    # every group sits in exactly one outer fold
    for g in set(groups):
        assert len({outer[i] for i, gg in enumerate(groups) if gg == g}) == 1
    assert max(n for n, _c in seen) < len(y)  # no fit ever used every row


def test_a_signal_gives_out_of_fold_probabilities_that_separate_the_classes():
    X, y, groups = synthetic(signal=3.0)
    oof = an.nested_oof(X, y, groups)["oof"]
    assert oof[y == 1].mean() > oof[y == 0].mean() + 0.2


def test_an_inner_fold_missing_a_class_stops_the_procedure():
    X = np.zeros((8, len(an.FEATURES)))
    y = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    with pytest.raises(an.AnalysisError):
        an.choose_c(X, y, [f"g{i}" for i in range(8)], "inner|0", 4)


# -------------------------------------------------------------------------------------------------------------- bands and uncertain handling
def test_bands_are_class_conditional_and_always_feasible():
    rng = np.random.default_rng(1)
    y = np.array([0] * 300 + [1] * 300)
    oof = np.concatenate([rng.beta(2, 5, 300), rng.beta(5, 2, 300)])
    b = an.bands(oof, y)
    assert b["false_pass_rate_among_unsupported"] <= an.ALPHA_PASS and b["false_reject_rate_among_supported"] <= an.ALPHA_REJECT
    assert b["t_hi"] >= 0.5 >= b["t_lo"]
    # an uninformative score (labels unrelated to it): the bands cover little and the abstain claim is dropped
    flat = an.bands(rng.permutation(np.linspace(0.3, 0.7, 600)), y)
    assert flat["confident_coverage"] < an.MIN_CONFIDENT_COVERAGE and flat["abstain_claim_allowed"] is False


def test_an_impossible_pass_band_becomes_empty_rather_than_infeasible():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    oof = np.array([0.99, 0.98, 0.97, 0.96, 0.6, 0.7, 0.8, 0.9])  # every unsupported item scores above every supported one
    b = an.bands(oof, y)
    assert math.isinf(b["t_hi"]) and b["false_pass_rate_among_unsupported"] == 0.0


def test_uncertain_items_keep_their_best_guess_verdict_and_are_never_rejected_by_default():
    X, y, groups = synthetic(signal=2.0)
    nested = an.nested_oof(X, y, groups)
    model = an.final_model(X, y, groups, nested["oof"])
    out = an.predict(model, X)
    middle = out["uncertain"]
    assert middle.any()
    assert np.array_equal(out["binary"][middle], (out["p"][middle] >= 0.5).astype(int))
    assert out["binary"][middle].sum() > 0  # some uncertain items pass: abstention is not an automatic rejection
    assert len(out["binary"]) == len(y)  # every item stays in every denominator


def test_the_model_hash_covers_its_content():
    X, y, groups = synthetic()
    nested = an.nested_oof(X, y, groups)
    m = an.final_model(X, y, groups, nested["oof"])
    body = {k: v for k, v in m.items() if k != "model_sha256"}
    import hashlib
    import json

    assert hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == m["model_sha256"]
    body["coef"][0] += 1e-9
    assert hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != m["model_sha256"]


# -------------------------------------------------------------------------------------------------------------- screening gate
def gate(y, a0, a2, pops=None):
    pops = pops or (["train"] * (len(y) // 2) + ["dev"] * (len(y) - len(y) // 2))
    return an.screen_gate(pops, y, a0, a2)


def test_the_gate_passes_only_when_all_four_conditions_hold():
    y = [1, 0] * 60
    a0 = [1, 1] * 60  # all pass: 60 false passes
    a2 = [1, 0] * 60  # perfect
    g = gate(y, a0, a2)
    assert g["pass"] and g["s1_pooled_kappa_gain"] and g["s2_each_subset_positive"] and g["s3_error_classes"] and g["s4_flip_precision"]


def test_the_gate_fails_when_no_kappa_gain_or_a_subset_does_not_improve():
    y = [1, 0, 1, 0] * 30
    assert not gate(y, list(y), list(y))["pass"]
    a0 = [1 - v if i % 4 == 0 else v for i, v in enumerate(y)]  # errors only in the TRAIN half? build explicitly instead
    pops = ["train"] * 60 + ["dev"] * 60
    a0 = y[:60][:] + [1 - v for v in y[60:]]
    a2 = [1 - v for v in y[:60]] + y[60:]  # fixes dev, breaks TRAIN
    g = gate(y, a0, a2, pops)
    assert g["s2_each_subset_positive"] is False and not g["pass"]


def test_a_rise_in_one_error_class_above_one_percent_fails_even_with_a_kappa_gain():
    y = [1] * 60 + [0] * 60
    a0 = [0] * 10 + [1] * 50 + [0] * 60  # 10 false rejections, 0 false passes
    a2 = [1] * 60 + [1] * 10 + [0] * 50  # fixes all 10 false rejections but adds 10 false passes
    g = gate(y, a0, a2)
    assert g["false_pass_change"] == 10 and g["s3_error_classes"] is False and not g["pass"]


def test_flip_precision_must_exceed_one_half():
    y = [1, 0] * 60
    a0 = list(y)
    a2 = [1 - v if i < 4 else v for i, v in enumerate(y)]  # four flips, all creating errors
    g = gate(y, a0, a2)
    assert g["flips_creating_error"] == 4 and g["flips_fixing_error"] == 0 and g["s4_flip_precision"] is False
    assert gate(y, list(y), list(y))["s4_flip_precision"] is False  # no flips: no evidence


def test_gate_inputs_must_align():
    with pytest.raises(an.AnalysisError):
        an.screen_gate(["dev"], [1, 0], [1, 0], [1, 0])


def test_paired_bootstrap_is_deterministic_document_grouped_and_never_part_of_the_gate():
    y = [1, 0] * 60
    groups = [f"d{i // 3}" for i in range(120)]
    a0 = [1] * 120
    a2 = list(y)
    ci = an.paired_bootstrap(groups, y, a0, a2, draws=400)
    assert ci == an.paired_bootstrap(groups, y, a0, a2, draws=400) and ci[0] > 0
    assert "interval" not in an.screen_gate(["train"] * 60 + ["dev"] * 60, y, a0, a2)

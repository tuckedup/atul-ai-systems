"""Tests for the judge combiners.

The cross-check against sklearn is the important one. `evalops` keeps no numpy/sklearn dependency
on the production path -- `metrics.py` hand-rolls kappa for the same reason -- so the hand-rolled
IRLS fit has to be pinned against the reference implementation it claims to match, or it can drift
silently and nobody would know which of the two was right.
"""
from __future__ import annotations

import random

import pytest
from evalops.combiners import (
    FEATURE_FAMILIES,
    Combinable,
    CombinerError,
    Contender,
    FeatureMatrix,
    FittedLogistic,
    LogisticSpec,
    MajorityVote,
    MeanScore,
    _fit_logistic,
    align,
    build_features,
    compare,
    fit_logistic,
    require_coverage,
)
from evalops.dataset import Judgment


def _j(case_id, variant_id, score, *, criteria=None, status="ok", location=None, unverified=0):
    return Judgment(
        case_id=case_id, variant_id=variant_id, model_requested="m", config_hash="h",
        raw_score=score, status=status,
        criteria=criteria or [],
        evidence_location=location or {},
        support_unverified=unverified,
    )


# ---------------------------------------------------------------- alignment and coverage


def test_align_reports_the_cases_every_component_judged():
    js = [_j("c1", "a", 1.0), _j("c1", "b", 0.0), _j("c2", "a", 1.0)]
    c, index = align(js, ["a", "b"], case_ids=["c1", "c2"])
    assert c.case_ids == ("c1",)
    assert c.missing == {"c2": ("b",)}
    assert c.coverage == pytest.approx(0.5)
    assert index[("a", "c1")].raw_score == 1.0


def test_align_counts_a_case_no_component_judged_against_coverage():
    # Without the case universe, a case nobody answered would vanish from the denominator and the
    # ensemble would report perfect coverage while having answered two thirds of the split.
    js = [_j("c1", "a", 1.0), _j("c1", "b", 1.0)]
    c, _ = align(js, ["a", "b"], case_ids=["c1", "c2", "c3"])
    assert c.n_offered == 3
    assert c.coverage == pytest.approx(1 / 3)
    assert c.missing == {"c2": ("a", "b"), "c3": ("a", "b")}


def test_align_excludes_errored_judgments():
    js = [_j("c1", "a", 1.0), _j("c1", "b", None, status="provider_error")]
    c, _ = align(js, ["a", "b"], case_ids=["c1"])
    assert c.case_ids == ()
    assert c.missing == {"c1": ("b",)}


def test_coverage_is_multiplicative_across_independent_components():
    # The structural obstacle the module docstring names: eight components each missing a
    # different 10% of cases leave far less than 90% jointly covered.
    rng = random.Random(11)
    ids = [f"c{i}" for i in range(400)]
    comps = [f"v{k}" for k in range(8)]
    js = [_j(cid, v, 0.5) for cid in ids for v in comps if rng.random() > 0.10]
    c, _ = align(js, comps, case_ids=ids)
    assert c.coverage < 0.55, f"expected compounding loss, got {c.coverage:.3f}"


def test_require_coverage_refuses_a_thin_ensemble():
    js = [_j("c1", "a", 1.0), _j("c1", "b", 1.0), _j("c2", "a", 1.0)]
    c, _ = align(js, ["a", "b"], case_ids=["c1", "c2"])
    with pytest.raises(CombinerError, match="below the required"):
        require_coverage(c, 0.98)


def test_require_coverage_accepts_a_complete_ensemble():
    js = [_j("c1", "a", 1.0), _j("c1", "b", 1.0)]
    c, _ = align(js, ["a", "b"], case_ids=["c1"])
    require_coverage(c, 0.98)  # does not raise


def test_align_rejects_an_empty_component_list():
    with pytest.raises(CombinerError, match="at least one component"):
        align([], [])


# ---------------------------------------------------------------- features


def _two_component_index(n=40):
    js = []
    for i in range(n):
        for v in ("a", "b"):
            js.append(_j(
                f"c{i}", v, 1.0 if i % 2 else 0.0,
                criteria=[{"id": "shared", "verdict": i % 2 == 1}],
                location={"total": 2, "in_source": 1, "in_candidate": 1, "unlocated": 0,
                          "too_short": 0},
            ))
    return align(js, ["a", "b"], case_ids=[f"c{i}" for i in range(n)])


def test_build_features_names_every_column():
    c, index = _two_component_index()
    m = build_features(c, index)
    assert "a::score" in m.names and "b::score" in m.names
    assert "a::crit::shared" in m.names
    assert "a::ev_source_rate" in m.names
    assert "a::unverified_support_rate" in m.names
    assert len(m) == len(c.case_ids)
    assert all(len(r) == len(m.names) for r in m.rows)


def test_build_features_families_can_be_ablated():
    c, index = _two_component_index()
    only_scores = build_features(c, index, families=("score",))
    assert only_scores.names == ("a::score", "b::score")
    no_evidence = build_features(c, index, families=("score", "criteria"))
    assert not any("ev_" in n for n in no_evidence.names)


def test_build_features_rejects_an_unknown_family():
    c, index = _two_component_index()
    with pytest.raises(CombinerError, match="unknown feature families"):
        build_features(c, index, families=("score", "vibes"))


def test_build_features_drops_per_case_criterion_ids():
    # Decompose-mode judgments name their criteria fact_0, fact_1, ... Those are per-case ids and
    # carry no cross-case meaning; keeping them fills the matrix with near-empty columns.
    js = []
    for i in range(50):
        js.append(_j(f"c{i}", "a", 0.5, criteria=[
            {"id": f"fact_{i}", "verdict": True},
            {"id": "stable", "verdict": False},
        ]))
    c, index = align(js, ["a"], case_ids=[f"c{i}" for i in range(50)])
    m = build_features(c, index, families=("criteria",))
    assert m.names == ("a::crit::stable",)


def test_build_features_keeps_a_criterion_present_on_nearly_every_case():
    js = [_j(f"c{i}", "a", 0.5, criteria=[{"id": "nearly", "verdict": True}])
          for i in range(99)]
    js.append(_j("c99", "a", 0.5, criteria=[{"id": "other", "verdict": True}]))
    c, index = align(js, ["a"], case_ids=[f"c{i}" for i in range(100)])
    m = build_features(c, index, families=("criteria",))
    assert m.names == ("a::crit::nearly",)


def test_evidence_features_are_rates_not_counts():
    # A long response cites more spans. If the feature were a count, the combiner would learn
    # response length instead of evidence quality.
    js = [
        _j("c1", "a", 1.0, criteria=[{"id": "k", "verdict": True}],
           location={"total": 2, "in_source": 1, "in_candidate": 0, "unlocated": 1, "too_short": 0}),
        _j("c2", "a", 1.0, criteria=[{"id": "k", "verdict": True}],
           location={"total": 20, "in_source": 10, "in_candidate": 0, "unlocated": 10,
                     "too_short": 0}),
    ]
    c, index = align(js, ["a"], case_ids=["c1", "c2"])
    m = build_features(c, index, families=("evidence",))
    col = m.names.index("a::ev_source_rate")
    assert m.rows[0][col] == pytest.approx(0.5)
    assert m.rows[1][col] == pytest.approx(0.5)


def test_too_short_quotes_leave_the_source_rate_defined():
    js = [_j("c1", "a", 1.0, location={"total": 3, "in_source": 1, "in_candidate": 0,
                                       "unlocated": 0, "too_short": 2})]
    c, index = align(js, ["a"], case_ids=["c1"])
    m = build_features(c, index, families=("evidence",))
    assert m.rows[0][m.names.index("a::ev_source_rate")] == pytest.approx(1.0)


def test_build_features_on_no_cases_is_empty_not_an_error():
    c = Combinable((), {"c1": ("a",)}, ("a",))
    assert build_features(c, {}).rows == ()


# ---------------------------------------------------------------- unweighted combiners


def test_mean_score_averages_components():
    js = [_j("c1", "a", 1.0), _j("c1", "b", 0.0)]
    c, index = align(js, ["a", "b"], case_ids=["c1"])
    assert MeanScore(("a", "b")).scores(c, index) == [pytest.approx(0.5)]


def test_majority_vote_returns_the_supporting_fraction():
    js = [_j("c1", "a", 0.9), _j("c1", "b", 0.9), _j("c1", "d", 0.1)]
    c, index = align(js, ["a", "b", "d"], case_ids=["c1"])
    assert MajorityVote(("a", "b", "d")).scores(c, index) == [pytest.approx(2 / 3)]


def test_majority_vote_member_threshold_changes_the_votes():
    js = [_j("c1", "a", 0.6), _j("c1", "b", 0.4)]
    c, index = align(js, ["a", "b"], case_ids=["c1"])
    assert MajorityVote(("a", "b"), member_threshold=0.5).scores(c, index) == [pytest.approx(0.5)]
    assert MajorityVote(("a", "b"), member_threshold=0.7).scores(c, index) == [pytest.approx(0.0)]


# ---------------------------------------------------------------- the logistic fit


def _synthetic(n=300, d=4, seed=3):
    rng = random.Random(seed)
    true_w = [1.8, -1.2, 0.7, 0.0][:d]
    rows, labels = [], []
    for _ in range(n):
        x = [rng.gauss(0, 1) for _ in range(d)]
        z = 0.4 + sum(w * xi for w, xi in zip(true_w, x, strict=True))
        labels.append(1 if rng.random() < 1 / (1 + pow(2.718281828459045, -z)) else 0)
        rows.append(x)
    return rows, labels


@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_fit_matches_sklearn_logistic_regression(C):
    # sklearn is a TEST dependency only. This is the cross-check that stops the hand-rolled IRLS
    # from drifting away from the objective its docstring claims to minimise.
    sklearn_lm = pytest.importorskip("sklearn.linear_model")
    rows, labels = _synthetic()
    coefs, intercept, converged, _ = _fit_logistic(rows, labels, C=C)
    assert converged
    # L2 is sklearn's default; naming it explicitly is deprecated as of 1.8.
    ref = sklearn_lm.LogisticRegression(C=C, max_iter=5000, tol=1e-10)
    ref.fit(rows, labels)
    for mine, theirs in zip(coefs, ref.coef_[0], strict=True):
        assert mine == pytest.approx(float(theirs), abs=1e-3)
    assert intercept == pytest.approx(float(ref.intercept_[0]), abs=1e-3)


def test_stronger_regularisation_shrinks_the_coefficients():
    rows, labels = _synthetic()
    weak, _, _, _ = _fit_logistic(rows, labels, C=100.0)
    strong, _, _, _ = _fit_logistic(rows, labels, C=0.01)
    assert sum(abs(c) for c in strong) < sum(abs(c) for c in weak)


def test_fit_is_deterministic():
    rows, labels = _synthetic()
    a = _fit_logistic(rows, labels, C=1.0)
    b = _fit_logistic(rows, labels, C=1.0)
    assert a == b


def test_fit_refuses_single_class_training_labels():
    rows, _ = _synthetic(n=50)
    with pytest.raises(CombinerError, match="single-class"):
        _fit_logistic(rows, [1] * 50, C=1.0)


def test_fit_refuses_a_ragged_matrix():
    with pytest.raises(CombinerError, match="ragged"):
        _fit_logistic([[1.0, 2.0], [3.0]], [0, 1], C=1.0)


def test_fit_refuses_mismatched_label_count():
    with pytest.raises(CombinerError, match="labels"):
        _fit_logistic([[1.0], [2.0]], [0], C=1.0)


def test_duplicate_feature_columns_are_survivable_under_ridge():
    # Two identical columns make X'X singular; the ridge term is what keeps the solve well posed.
    rows, labels = _synthetic(d=2)
    doubled = [[r[0], r[0], r[1]] for r in rows]
    coefs, _, converged, _ = _fit_logistic(doubled, labels, C=1.0)
    assert converged
    assert coefs[0] == pytest.approx(coefs[1], abs=1e-6), "ridge should split a tied pair evenly"


# ---------------------------------------------------------------- fit_logistic / FittedLogistic


def _fitted(n=120):
    # `positive` keys off the case id, not off enumeration order: `align` returns case ids in
    # SORTED order, so "c10" comes before "c2". Labels have to be read off `m.case_ids` rather
    # than built from a parallel `range(n)`, or every row is paired with the wrong label. This is
    # the usage the matrix's row order requires, so the helper models it.
    def positive(case_id: str) -> int:
        return int(case_id.removeprefix("c")) % 2

    js = []
    for i in range(n):
        cid = f"c{i}"
        for v in ("a", "b"):
            js.append(_j(cid, v, 0.9 if positive(cid) else 0.1,
                         criteria=[{"id": "shared", "verdict": bool(positive(cid))}]))
    c, index = align(js, ["a", "b"], case_ids=[f"c{i}" for i in range(n)])
    m = build_features(c, index, families=("score", "criteria"))
    labels = [positive(cid) for cid in m.case_ids]
    groups = [f"g{k // 2}" for k in range(len(m.case_ids))]
    spec = LogisticSpec("cmb-test", ("a", "b"), families=("score", "criteria"), C=1.0,
                        task_class="code")
    return fit_logistic(spec, m, labels, groups), m, labels


def test_fit_logistic_records_provenance():
    fitted, m, _ = _fitted()
    assert fitted.fitted_on == "train"
    assert fitted.n_train == len(m)
    assert fitted.n_train_groups == len(m) // 2
    assert fitted.feature_names == list(m.names)
    assert fitted.feature_spec_hash == m.spec_hash
    assert fitted.task_class == "code"


def test_fit_logistic_refuses_to_fit_on_test():
    _, m, labels = _fitted()
    spec = LogisticSpec("cmb-test", ("a", "b"), C=1.0)
    with pytest.raises(CombinerError, match="test split"):
        fit_logistic(spec, m, labels, ["g"] * len(labels), fitted_on="test")


def test_fitted_combiner_separates_a_separable_set():
    fitted, m, labels = _fitted()
    predicted = [1 if s >= 0.5 else 0 for s in fitted.scores(m)]
    assert predicted == labels


def test_fitted_combiner_refuses_a_different_feature_set():
    fitted, _, _ = _fitted()
    wrong = FeatureMatrix(("c0",), ("something::else",), ((1.0,),))
    with pytest.raises(CombinerError, match="different feature set"):
        fitted.scores(wrong)


def test_fitted_combiner_refuses_a_row_of_the_wrong_width():
    fitted, _, _ = _fitted()
    with pytest.raises(CombinerError, match="expects"):
        fitted.score_row([1.0])


def test_fitted_combiner_round_trips_through_dict():
    fitted, m, _ = _fitted()
    again = FittedLogistic.from_dict(fitted.as_dict())
    assert again.scores(m) == fitted.scores(m)


def test_from_dict_ignores_unknown_fields():
    fitted, _, _ = _fitted()
    raw = fitted.as_dict() | {"something_a_later_version_added": 1}
    assert FittedLogistic.from_dict(raw).combiner_id == fitted.combiner_id


def test_default_feature_families_are_the_declared_ones():
    assert LogisticSpec("x", ("a",)).families == FEATURE_FAMILIES


# ---------------------------------------------------------------- comparison against the baseline


def test_compare_names_the_strongest_single_component():
    c = compare([
        Contender("v5", 0.50, 0.85, 200, "single"),
        Contender("v4", 0.47, 0.85, 200, "single"),
        Contender("mean", 0.48, 0.50, 200, "unweighted"),
    ])
    assert c.best_single.name == "v5"
    assert c.best_overall.name == "v5"
    assert c.combiner_helps is False
    assert c.margin == pytest.approx(0.0)


def test_compare_reports_a_genuine_combiner_win():
    c = compare([
        Contender("v5", 0.50, 0.85, 200, "single"),
        Contender("learned", 0.61, 0.50, 200, "learned"),
    ])
    assert c.best_overall.name == "learned"
    assert c.combiner_helps is True
    assert c.margin == pytest.approx(0.11)


def test_a_tie_does_not_count_as_a_combiner_win():
    # Equal agreement for k times the cost and k times the failure surface is a loss.
    c = compare([
        Contender("v5", 0.50, 0.85, 200, "single"),
        Contender("mean", 0.50, 0.50, 200, "unweighted"),
    ])
    assert c.best_overall.kind == "single"
    assert c.combiner_helps is False


def test_compare_requires_a_single_component_baseline():
    with pytest.raises(CombinerError, match="strongest individual judge"):
        compare([Contender("mean", 0.7, 0.5, 100, "unweighted")])


def test_compare_ignores_undefined_kappa_contenders():
    c = compare([
        Contender("v5", 0.50, 0.85, 200, "single"),
        Contender("broken", None, 0.5, 200, "learned"),
    ])
    assert c.best_overall.name == "v5"


def test_compare_raises_when_nothing_is_defined():
    with pytest.raises(CombinerError, match="no contender"):
        compare([Contender("broken", None, 0.5, 200, "single")])


def test_comparison_as_dict_carries_the_baseline():
    c = compare([
        Contender("v5", 0.50, 0.85, 200, "single"),
        Contender("learned", 0.61, 0.50, 200, "learned"),
    ])
    d = c.as_dict()
    assert d["best_single"]["name"] == "v5"
    assert d["margin_over_best_single"] == pytest.approx(0.11)
    assert len(d["contenders"]) == 2

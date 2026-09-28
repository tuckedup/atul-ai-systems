import random

import pytest
from evalops.calibrate import (
    ROUTEBENCH_MIN_KAPPA,
    CalibrationBundle,
    CalibrationError,
    Paired,
    expected_bundle_id,
    pair,
    select_threshold,
    validate_bundle,
)
from evalops.dataset import Annotation, CaseRecord, Corpus, Judgment, LabelProvenance
from evalops.rubrics import bundle_hash


def _case(case_id, group_id=None, task_class="code"):
    return CaseRecord(
        case_id=case_id, group_id=group_id or f"g-{case_id}", task_class=task_class,
        task_input=f"input-{case_id}",
    )


def _annotation(case_id, label, provenance=LabelProvenance.HUMAN_LOCAL, **kw):
    return Annotation(case_id=case_id, annotator_id="r1", label=label, provenance=provenance, **kw)


def _judgment(case_id, *, status="ok", raw_score=None):
    return Judgment(
        case_id=case_id, variant_id="v1", model_requested="m", status=status, raw_score=raw_score,
    )


# ---------------------------------------------------------------- pair()


def test_pair_raises_on_fixture_provenance_label():
    # Regression test for the original defect: fixture labels are the artefact this rewrite
    # exists to remove, and pair() is the ONLY place labels meet judgments, so it must refuse
    # them outright rather than silently admitting them into the metric.
    case = _case("c1")
    ann = _annotation("c1", 1, provenance=LabelProvenance.FIXTURE)
    j = _judgment("c1", raw_score=0.9)
    with pytest.raises(CalibrationError, match="synthetic_fixture"):
        pair([case], {"c1": ann}, [j])


def test_pair_excludes_non_ok_judgments_and_counts_them_as_errors():
    cases = [_case(f"c{i}") for i in range(5)]
    labels = {c.case_id: _annotation(c.case_id, 1) for c in cases}
    judgments = [
        _judgment("c0", raw_score=0.9),
        _judgment("c1", raw_score=0.1),
        _judgment("c2", raw_score=0.7),
        _judgment("c3", status="provider_error", raw_score=None),
        _judgment("c4", status="parse_error", raw_score=None),
    ]
    out = pair(cases, labels, judgments)
    assert sorted(out.case_ids) == ["c0", "c1", "c2"]
    assert sorted(out.unjudged) == ["c3", "c4"]
    assert out.errors == {"provider_error": 1, "parse_error": 1}
    # Errors do not enter the metric as score 0: they are missing, not a fail -- only the
    # three genuinely-ok judgments contribute a judge_score at all.
    assert len(out.judge_scores) == 3
    assert out.completion == pytest.approx(3 / 5)


# ---------------------------------------------------------------- Corpus.resolved_labels


def test_resolved_labels_excludes_unadjudicated_disagreement():
    c1, c2 = _case("c1"), _case("c2")
    a1 = _annotation("c1", 1)
    a2 = Annotation(case_id="c1", annotator_id="r2", label=0, provenance=LabelProvenance.HUMAN_LOCAL)
    corpus = Corpus(cases=[c1, c2], annotations=[a1, a2])
    resolved = corpus.resolved_labels()
    assert "c1" not in resolved


def test_resolved_labels_prefers_adjudicated_record():
    c1 = _case("c1")
    a1 = _annotation("c1", 1)
    a2 = Annotation(case_id="c1", annotator_id="r2", label=0, provenance=LabelProvenance.HUMAN_LOCAL)
    adjudicated = Annotation(
        case_id="c1", annotator_id="adjudicator", label=1,
        provenance=LabelProvenance.HUMAN_LOCAL, adjudicated=True,
    )
    corpus = Corpus(cases=[c1], annotations=[a1, a2, adjudicated])
    resolved = corpus.resolved_labels()
    assert resolved["c1"].adjudicated is True
    assert resolved["c1"].annotator_id == "adjudicator"


# ---------------------------------------------------------------- select_threshold


def _paired_from(human, judge_scores, *, tasks=None, prov=None):
    n = len(human)
    return Paired(
        case_ids=[f"c{i}" for i in range(n)],
        human=human,
        judge_scores=judge_scores,
        groups=[f"g{i}" for i in range(n)],
        tasks=tasks or ["code"] * n,
        provenance=prov or [LabelProvenance.HUMAN_LOCAL.value] * n,
    )


def test_select_threshold_raises_below_thirty_paired_items():
    p = _paired_from([1] * 15 + [0] * 14, [0.9] * 15 + [0.1] * 14)
    with pytest.raises(CalibrationError, match="usable dev set"):
        select_threshold(p)


def test_select_threshold_raises_when_no_threshold_yields_defined_kappa():
    # Human labels are constant AND the judge score (1.0) is >= every grid threshold, so every
    # binarization is also constant-and-identical: kappa is undefined at every grid point.
    p = _paired_from([1] * 40, [1.0] * 40)
    with pytest.raises(CalibrationError, match="no threshold produced a defined kappa"):
        select_threshold(p)


def _separable_dev_set():
    rng = random.Random(11)
    n = 80
    human = [1 if i < 40 else 0 for i in range(n)]
    scores = []
    for h in human:
        base = 0.65 if h == 1 else 0.35
        scores.append(min(1.0, max(0.0, base + rng.uniform(-0.4, 0.4))))
    return _paired_from(human, scores)


def test_select_threshold_deterministic():
    p = _separable_dev_set()
    tc1 = select_threshold(p)
    tc2 = select_threshold(p)
    assert tc1.threshold == tc2.threshold
    assert tc1.dev_kappa == tc2.dev_kappa


def test_select_threshold_maximises_dev_kappa_over_the_grid():
    p = _separable_dev_set()
    tc = select_threshold(p)
    usable = [r for r in tc.grid if r["defined"]]
    best_kappa = max(r["kappa"] for r in usable)
    assert tc.dev_kappa == pytest.approx(best_kappa)
    assert tc.threshold == pytest.approx(0.6)
    assert tc.dev_kappa == pytest.approx(0.55)


def test_threshold_invariance_of_the_human_marginal():
    # Defect-1 regression test: binarize() applies only to judge scores. The human marginal
    # (tp + fn) must be identical across the whole threshold grid, because the target must
    # never move while the predictor's threshold is being tuned.
    p = _separable_dev_set()
    tc = select_threshold(p)
    marginals = {row["confusion"]["tp"] + row["confusion"]["fn"] for row in tc.grid}
    assert marginals == {sum(p.human)}


# ---------------------------------------------------------------- validate_bundle


def _bundle(**overrides):
    base = {
        "bundle_id": "", "created_at": "now", "variant_id": "v1", "judge_config": {},
        "threshold": 0.5, "min_kappa": ROUTEBENCH_MIN_KAPPA,
        "rubric_bundle_hash": bundle_hash(), "dataset_hash": "dh",
        "split_seed": 1, "dev_kappa": 0.8, "dev_n": 100, "selection_rule": "r",
        "threshold_grid": [], "label_provenance_counts": {}, "test_result": None,
        # A real artifact binds the labels and the split membership, not just the case contents.
        "annotations_hash": "ann-1", "split_membership_hash": "mem-1",
    }
    base.update(overrides)
    bundle = CalibrationBundle(**base)
    # The gate recomputes `bundle_id` from the identity fields, so it has to be derived here
    # rather than hardcoded -- unless a test is deliberately overriding it to check that a
    # tampered artifact is rejected.
    if "bundle_id" not in overrides:
        bundle.bundle_id = expected_bundle_id(bundle)
    return bundle


def _track(kappa, *, defined=True, insufficient=False, n=50, reason="", track_completion=1.0):
    # `track_completion` and `n_labelled` became required when the gate was tightened: overall
    # completion can be high while the PRIMARY track is badly covered, which is the case that
    # would let a kappa be computed on a convenient subset of the labelled test items.
    return {
        "insufficient": insufficient, "defined": defined, "kappa": kappa, "n": n,
        "reason": reason, "note": reason,
        "track_completion": track_completion,
        "n_labelled": n if track_completion in (None, 0) else round(n / track_completion),
    }


def _passing_test_result(kappa=0.8, completion_rate=1.0, track_completion=1.0):
    # completion_rate defaults to 1.0, not 0.99: the acceptance contract requires an automated
    # decision for EVERY labelled test item, and the earlier 0.98 gate silently relaxed that.
    paired = 200
    unjudged = round(paired * (1 - completion_rate) / completion_rate) if completion_rate else 0
    return {
        "tracks": {"human_judgment_of_response": _track(kappa, track_completion=track_completion)},
        "completion": {
            "completion_rate": completion_rate, "paired": paired,
            "unjudged": unjudged, "judge_errors": {},
        },
    }


def test_validate_bundle_missing_file(tmp_path):
    ok, reasons = validate_bundle(tmp_path / "nope.json")
    assert ok is False
    assert "no calibration artifact" in reasons[0]


def test_validate_bundle_unreadable_json(tmp_path):
    p = tmp_path / "corrupt.json"
    p.write_text("not json {{{", encoding="utf-8")
    ok, reasons = validate_bundle(p)
    assert ok is False
    assert "unreadable" in reasons[0]


def test_validate_bundle_no_test_result(tmp_path):
    b = _bundle(test_result=None)
    p = tmp_path / "b.json"
    b.save(p)
    ok, reasons = validate_bundle(p)
    assert ok is False
    assert any("never measured on held-out data" in r for r in reasons)


def test_validate_bundle_rubric_hash_mismatch(tmp_path):
    b = _bundle(rubric_bundle_hash="deadbeefdeadbeef", test_result=_passing_test_result())
    p = tmp_path / "b.json"
    b.save(p)
    ok, reasons = validate_bundle(p)
    assert ok is False
    assert any("rubric hash mismatch" in r for r in reasons)


def test_validate_bundle_kappa_below_policy(tmp_path):
    b = _bundle(test_result=_passing_test_result(kappa=0.5))
    p = tmp_path / "b.json"
    b.save(p)
    ok, reasons = validate_bundle(p)
    assert ok is False
    assert any("below the 0.74 policy" in r for r in reasons)


def test_validate_bundle_undefined_kappa(tmp_path):
    tr = {
        "tracks": {"human_judgment_of_response": _track(None, defined=False, reason="single class")},
        "completion": {"completion_rate": 1.0, "paired": 200, "unjudged": 0, "judge_errors": {}},
    }
    b = _bundle(test_result=tr)
    p = tmp_path / "b.json"
    b.save(p)
    ok, reasons = validate_bundle(p)
    assert ok is False
    assert any("kappa is undefined" in r for r in reasons)


def test_validate_bundle_incomplete_coverage_rejects(tmp_path):
    b = _bundle(test_result=_passing_test_result(completion_rate=0.9))
    p = tmp_path / "b.json"
    b.save(p)
    ok, reasons = validate_bundle(p)
    assert ok is False
    assert any("automated decision" in r for r in reasons)


def test_validate_bundle_min_kappa_lower_than_policy(tmp_path):
    b = _bundle(min_kappa=0.5, test_result=_passing_test_result())
    p = tmp_path / "b.json"
    b.save(p)
    ok, reasons = validate_bundle(p)
    assert ok is False
    assert any("below policy" in r for r in reasons)


def test_validate_bundle_true_for_well_formed_passing_bundle(tmp_path):
    b = _bundle(test_result=_passing_test_result())
    p = tmp_path / "b.json"
    b.save(p)
    ok, reasons = validate_bundle(p)
    assert ok is True
    assert reasons == []


def test_validate_bundle_kappa_exactly_at_policy_passes(tmp_path):
    b = _bundle(min_kappa=0.74, test_result=_passing_test_result(kappa=0.74))
    p = tmp_path / "b.json"
    b.save(p)
    ok, _reasons = validate_bundle(p, min_kappa=0.74)
    assert ok is True


def test_validate_bundle_kappa_just_below_policy_fails(tmp_path):
    b = _bundle(min_kappa=0.74, test_result=_passing_test_result(kappa=0.7399))
    p = tmp_path / "b.json"
    b.save(p)
    ok, reasons = validate_bundle(p, min_kappa=0.74)
    assert ok is False
    assert any("0.7399" in r for r in reasons)

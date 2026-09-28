"""Tests for four freeze-integrity holes: the protocol was documented but not enforced.

`run_calibration`'s docstring said the dev -> freeze -> test ordering "is enforced by the CLI,
not by documentation". It was in fact enforced by neither type nor assertion -- only by the
order in which `cmd_dev` / `cmd_freeze` / `cmd_test` happened to call things. Anything else
calling `select_threshold(test_paired)` or `evaluate(bundle, dev_paired)` got a number with no
complaint, and the number looks exactly like a held-out one.

Hole 1 -- `select_threshold` could be handed the test split.
Hole 2 -- `evaluate` could be handed the dev split.
Hole 3 -- `evaluate` never checked that the judgments it scored came from the variant and
          `config_hash` the bundle names, so a frozen bundle could be measured with a different
          judge's scores, or with two variants' scores mixed together.
Hole 4 -- `bundle_id` was computed at freeze and never recomputed, so editing `threshold`,
          `variant_id`, `judge_config`, `dataset_hash` or `split_seed` in the JSON left a stale
          id that nothing compared against anything.
"""
from __future__ import annotations

import json

import pytest
from evalops.calibrate import (
    ROUTEBENCH_MIN_KAPPA,
    CalibrationBundle,
    CalibrationError,
    Paired,
    evaluate,
    expected_bundle_id,
    freeze,
    identity_hash,
    pair,
    select_threshold,
    validate_bundle,
)
from evalops.dataset import Annotation, CaseRecord, Judgment, LabelProvenance
from evalops.rubrics import bundle_hash

CONFIG = {"model": "gpt-4.1-mini", "config_hash": "cfg-frozen"}


def _paired(n: int = 60, *, split: str = "dev", judges: dict | None = None) -> Paired:
    """A separable paired set: judge score tracks the label, so kappa is defined and high."""
    human = [i % 2 for i in range(n)]
    scores = [0.9 if h else 0.1 for h in human]
    p = Paired(
        case_ids=[f"c{i}" for i in range(n)],
        human=human,
        judge_scores=scores,
        groups=[f"g{i // 2}" for i in range(n)],
        tasks=["code"] * n,
        provenance=[LabelProvenance.HUMAN_EXPERT.value] * n,
        split=split,
    )
    p.judges = judges if judges is not None else {("v-frozen", "cfg-frozen"): n}
    return p


def _frozen_bundle(dev: Paired | None = None) -> CalibrationBundle:
    dev = dev or _paired()
    choice = select_threshold(dev)
    return freeze(
        choice, variant_id="v-frozen", judge_config=CONFIG,
        dataset_hash="ds-1", split_seed=7, dev=dev,
        annotations_hash="ann-1", split_membership_hash="mem-1",
    )


# ---------------------------------------------------------------- Hole 1: selecting on test


def test_select_threshold_refuses_the_test_split():
    with pytest.raises(CalibrationError, match="TEST split"):
        select_threshold(_paired(split="test"))


def test_select_threshold_still_accepts_dev_and_train():
    for split in ("dev", "train", ""):
        assert select_threshold(_paired(split=split)).threshold > 0


# ---------------------------------------------------------------- Hole 2: measuring on dev


def test_evaluate_refuses_a_dev_paired_set():
    bundle = _frozen_bundle()
    with pytest.raises(CalibrationError, match="'dev' split"):
        evaluate(bundle, _paired(split="dev"), bootstrap=0)


def test_evaluate_accepts_the_test_split():
    bundle = _frozen_bundle()
    result = evaluate(bundle, _paired(split="test"), bootstrap=0)
    assert result["tracks"]["all_admissible_labels"]["kappa"] == pytest.approx(1.0)


def test_evaluate_accepts_an_untagged_paired_set_for_backwards_compatibility():
    # An empty split means "the caller did not say". Refusing it would break every caller that
    # builds a Paired by hand; the guard is for a set that says it is the WRONG split.
    bundle = _frozen_bundle()
    assert evaluate(bundle, _paired(split=""), bootstrap=0)["tracks"]


# ---------------------------------------------------------------- Hole 3: judge binding


def test_evaluate_refuses_judgments_from_another_variant():
    bundle = _frozen_bundle()
    other = _paired(split="test", judges={("v-some-other-judge", "cfg-frozen"): 60})
    with pytest.raises(CalibrationError, match="v-some-other-judge"):
        evaluate(bundle, other, bootstrap=0)


def test_evaluate_refuses_two_variants_mixed_together():
    bundle = _frozen_bundle()
    mixed = _paired(split="test", judges={
        ("v-frozen", "cfg-frozen"): 30, ("v-other", "cfg-frozen"): 30,
    })
    with pytest.raises(CalibrationError, match="v-other"):
        evaluate(bundle, mixed, bootstrap=0)


def test_evaluate_refuses_a_config_hash_that_drifted_after_the_freeze():
    bundle = _frozen_bundle()
    drifted = _paired(split="test", judges={("v-frozen", "cfg-EDITED-RUBRIC"): 60})
    with pytest.raises(CalibrationError, match="cfg-EDITED-RUBRIC"):
        evaluate(bundle, drifted, bootstrap=0)


def test_pair_records_the_variant_and_config_hash_of_every_judgment():
    cases = [CaseRecord(case_id="c1", group_id="g1", task_class="code", task_input="i",
                        context="", reference="r", candidate_output="o")]
    labels = {"c1": Annotation(case_id="c1", annotator_id="a", label=1,
                               provenance=LabelProvenance.HUMAN_EXPERT)}
    judgments = [Judgment(case_id="c1", variant_id="v9", model_requested="m",
                          config_hash="h9", raw_score=0.8, status="ok")]
    p = pair(cases, labels, judgments, split="test")
    assert p.split == "test"
    assert p.judges == {("v9", "h9"): 1}


def test_restrict_preserves_the_split_and_judge_binding():
    p = _paired(split="test")
    r = p.restrict(frozenset({LabelProvenance.HUMAN_EXPERT}))
    assert r.split == "test"
    assert r.judges == p.judges


# ---------------------------------------------------------------- Hole 4: bundle_id integrity


def _measured(tmp_path, **overrides):
    """A bundle that has faced test and passes the gate, written to disk."""
    dev = _paired()
    bundle = _frozen_bundle(dev)
    bundle.test_result = evaluate(bundle, _paired(split="test"), bootstrap=0)
    for k, v in overrides.items():
        setattr(bundle, k, v)
    path = tmp_path / "calibration_bundle.json"
    bundle.save(path)
    return path


def test_freeze_writes_a_bundle_id_that_recomputes(tmp_path):
    ok, reasons = validate_bundle(_measured(tmp_path), min_kappa=ROUTEBENCH_MIN_KAPPA)
    assert ok, reasons


@pytest.mark.parametrize("field,value", [
    ("threshold", 0.05),
    ("variant_id", "v-swapped"),
    ("dataset_hash", "ds-a-smaller-easier-corpus"),
    ("split_seed", 99),
    ("judge_config", {"model": "gpt-4o-mini", "config_hash": "cfg-frozen"}),
])
def test_editing_an_identity_field_after_the_freeze_is_rejected(tmp_path, field, value):
    # Each of these is a field someone might "fix" by hand in the JSON. Before the gate
    # recomputed `bundle_id`, every one of them passed.
    path = _measured(tmp_path)
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    ok, reasons = validate_bundle(path)
    assert ok is False
    assert any("bundle_id mismatch" in r for r in reasons), reasons


def test_lowering_the_threshold_by_hand_cannot_be_hidden_by_rewriting_the_id(tmp_path):
    # Recomputing a CONSISTENT id for the edited threshold gets past the integrity check -- and
    # must then be caught by the kappa the artifact actually reports, not waved through.
    path = _measured(tmp_path)
    payload = json.loads(path.read_text())
    payload["threshold"] = 0.05
    payload["bundle_id"] = identity_hash(
        variant_id=payload["variant_id"], threshold=payload["threshold"],
        judge_config=payload["judge_config"], rubric_bundle_hash=payload["rubric_bundle_hash"],
        dataset_hash=payload["dataset_hash"], split_seed=payload["split_seed"],
        annotations_hash=payload["annotations_hash"],
        split_membership_hash=payload["split_membership_hash"],
    )
    path.write_text(json.dumps(payload))
    ok, reasons = validate_bundle(path)
    # The id is now self-consistent, so integrity passes; what remains is that the reported
    # test_result was measured at the ORIGINAL threshold. That is the honest limit of a hash:
    # it proves the fields were not edited, not that the measurement used them.
    assert not any("bundle_id mismatch" in r for r in reasons), reasons
    assert ok, reasons


def test_expected_bundle_id_is_stable_across_reload(tmp_path):
    path = _measured(tmp_path)
    loaded = CalibrationBundle.load(path)
    assert loaded.bundle_id == expected_bundle_id(loaded)


# ---------------------------------------------------------------- dataset binding


def test_dataset_hash_mismatch_is_rejected(tmp_path):
    path = _measured(tmp_path)
    ok, reasons = validate_bundle(path, expected_dataset_hash="ds-2")
    assert ok is False
    assert any("dataset hash mismatch" in r for r in reasons), reasons


def test_dataset_hash_match_is_accepted(tmp_path):
    ok, reasons = validate_bundle(_measured(tmp_path), expected_dataset_hash="ds-1")
    assert ok, reasons


def test_dataset_binding_is_skipped_when_no_corpus_hash_is_supplied(tmp_path):
    # Validating an artifact on a machine without the corpus is legitimate; it is reported as
    # unverified by the CLI rather than failing the gate.
    ok, reasons = validate_bundle(_measured(tmp_path), expected_dataset_hash=None)
    assert ok, reasons


def test_rubric_drift_is_still_caught_independently(tmp_path):
    path = _measured(tmp_path, rubric_bundle_hash="not-the-working-tree")
    ok, reasons = validate_bundle(path)
    assert ok is False
    # The rubric hash is part of the identity payload, so editing it via `freeze`'s output breaks
    # BOTH checks. Both reasons are useful: one says the file was altered, the other says what.
    assert any("rubric hash mismatch" in r for r in reasons), reasons


def test_bundle_hash_is_the_working_tree_hash_at_freeze(tmp_path):
    bundle = CalibrationBundle.load(_measured(tmp_path))
    assert bundle.rubric_bundle_hash == bundle_hash()


# ---------------------------------------------------------------- Holes 5-7, from the second review
#
# Hole 5 -- `dataset_hash` digests case CONTENTS only, so a bundle was bound to the questions and
#           not the answers. Flipping a human label left it byte-identical, and "editing human
#           labels after seeing judge output" is the first thing docs/KAPPA_DESIGN.md §8 forbids.
# Hole 6 -- the bundle recorded `split_seed` but no membership digest. Since `of_case` reads
#           `assignment` directly, a hand-edited splits.json keeping the seed could move a group
#           from test to train undetected.
# Hole 7 -- `cmd_test` refused to re-measure a frozen bundle, but nothing stopped `cmd_freeze`
#           REPLACING the file. Freezing again and measuring the new bundle was the same loophole
#           one step earlier.

from evalops.calibrate import corpus_bindings


def test_a_relabelled_corpus_is_rejected(tmp_path):
    path = _measured(tmp_path)
    ok, reasons = validate_bundle(path, expected_annotations_hash="labels-were-edited")
    assert ok is False
    assert any("annotation hash mismatch" in r for r in reasons), reasons


def test_matching_annotation_hash_is_accepted(tmp_path):
    ok, reasons = validate_bundle(_measured(tmp_path), expected_annotations_hash="ann-1")
    assert ok, reasons


def test_moved_split_membership_is_rejected(tmp_path):
    path = _measured(tmp_path)
    ok, reasons = validate_bundle(path, expected_split_membership_hash="cases-were-moved")
    assert ok is False
    assert any("split membership mismatch" in r for r in reasons), reasons


def test_matching_split_membership_is_accepted(tmp_path):
    ok, reasons = validate_bundle(_measured(tmp_path), expected_split_membership_hash="mem-1")
    assert ok, reasons


def test_a_bundle_with_no_label_binding_is_rejected(tmp_path):
    path = _measured(tmp_path, annotations_hash="")
    ok, reasons = validate_bundle(path)
    assert ok is False
    assert any("records no annotations_hash" in r for r in reasons), reasons


def test_a_bundle_with_no_split_binding_is_rejected(tmp_path):
    path = _measured(tmp_path, split_membership_hash="")
    ok, reasons = validate_bundle(path)
    assert ok is False
    assert any("records no split_membership_hash" in r for r in reasons), reasons


def test_a_pre_binding_artifact_can_be_accepted_only_deliberately(tmp_path):
    """An escape hatch that has to be asked for by name.

    Built the way a real pre-binding artifact was: frozen WITHOUT the digests, so its `bundle_id`
    is internally consistent and only the missing bindings are at issue. `identity_hash` omits an
    empty digest from the payload for exactly this reason — adding the bindings must not
    retroactively invalidate the artifacts that predate them.
    """
    dev = _paired()
    bundle = freeze(select_threshold(dev), variant_id="v-frozen", judge_config=CONFIG,
                    dataset_hash="ds-1", split_seed=7, dev=dev)
    bundle.test_result = evaluate(bundle, _paired(split="test"), bootstrap=0)
    path = tmp_path / "calibration_bundle.json"
    bundle.save(path)

    ok, reasons = validate_bundle(path)
    assert ok is False
    assert not any("bundle_id mismatch" in r for r in reasons), (
        "a genuine pre-binding artifact must still verify its own identity", reasons)
    assert any("records no annotations_hash" in r for r in reasons), reasons

    ok, reasons = validate_bundle(path, require_data_bindings=False)
    assert ok, reasons


def test_the_two_new_digests_are_part_of_the_bundle_identity(tmp_path):
    # Otherwise they could be edited out of an artifact without breaking bundle_id.
    import json as _json

    path = _measured(tmp_path)
    for field in ("annotations_hash", "split_membership_hash"):
        payload = _json.loads(path.read_text())
        payload[field] = "swapped"
        p = tmp_path / f"{field}.json"
        p.write_text(_json.dumps(payload))
        ok, reasons = validate_bundle(p)
        assert ok is False
        assert any("bundle_id mismatch" in r for r in reasons), (field, reasons)


def test_annotations_hash_moves_when_a_label_flips():
    from evalops.dataset import Annotation, Corpus, LabelProvenance

    cases = [CaseRecord(case_id="c1", group_id="g1", task_class="code", task_input="i",
                        context="", reference="r", candidate_output="o")]
    def corpus(label):
        return Corpus(cases=cases, annotations=[Annotation(
            case_id="c1", annotator_id="a", label=label,
            provenance=LabelProvenance.HUMAN_EXPERT)])
    a, b = corpus(1), corpus(0)
    assert a.cases_hash == b.cases_hash, "the questions are identical"
    assert a.annotations_hash != b.annotations_hash, "the answers are not"


def test_annotations_hash_ignores_commentary_fields():
    # A typo fix in `reason` must not look like tampering; the label, rater, provenance and
    # adjudication flag are what the headline track claims.
    from evalops.dataset import Annotation, Corpus, LabelProvenance

    cases = [CaseRecord(case_id="c1", group_id="g1", task_class="code", task_input="i",
                        context="", reference="r", candidate_output="o")]
    def corpus(reason, timestamp):
        return Corpus(cases=cases, annotations=[Annotation(
            case_id="c1", annotator_id="a", label=1, reason=reason, timestamp=timestamp,
            provenance=LabelProvenance.HUMAN_EXPERT)])
    assert corpus("typo", "t1").annotations_hash == corpus("fixed", "t2").annotations_hash


def test_split_membership_hash_moves_when_a_group_changes_split():
    from evalops.splits import SplitPlan

    a = SplitPlan(assignment={"g1": "train", "g2": "test"}, weights={}, seed=1)
    b = SplitPlan(assignment={"g1": "train", "g2": "train"}, weights={}, seed=1)
    assert a.seed == b.seed, "the seed is unchanged, which is the point"
    assert a.membership_hash != b.membership_hash


def test_split_membership_hash_is_order_independent():
    from evalops.splits import SplitPlan

    a = SplitPlan(assignment={"g1": "train", "g2": "test"}, weights={}, seed=1)
    b = SplitPlan(assignment={"g2": "test", "g1": "train"}, weights={}, seed=1)
    assert a.membership_hash == b.membership_hash


def test_corpus_bindings_reports_all_three_for_the_committed_corpus():
    from pathlib import Path as _Path

    data = _Path(__file__).parent.parent / "evalops" / "data"
    if not (data / "cases.jsonl").exists():
        pytest.skip("no corpus in this checkout")
    b = corpus_bindings(data)
    assert set(b) == {"dataset", "annotations", "split_membership"}
    assert all(v for v in b.values()), b


def test_corpus_bindings_returns_nones_without_a_corpus(tmp_path):
    b = corpus_bindings(tmp_path)
    assert b["dataset"] is None
    assert b["annotations"] is None
    assert b["split_membership"] is None
    # And it says WHY, rather than swallowing the reason: an unverified binding on a machine
    # without the corpus is a legitimate state, but a silent one is indistinguishable from a bug.
    assert b["dataset_error"]
    assert b["split_membership_error"]


def test_corpus_bindings_records_no_error_when_the_corpus_loads():
    from pathlib import Path as _Path

    data = _Path(__file__).parent.parent / "evalops" / "data"
    if not (data / "cases.jsonl").exists():
        pytest.skip("no corpus in this checkout")
    b = corpus_bindings(data)
    assert "dataset_error" not in b
    assert "split_membership_error" not in b

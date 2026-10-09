"""Held-out readiness: offline, synthetic, no inference, no held-out labels.

Unit properties of `evalops.readiness`, then the real command path: `freeze -> readiness ->
test` through `run_calibration`, with the corpus, split and inference monkeypatched.
"""
from __future__ import annotations

import json
import random
from types import SimpleNamespace
from typing import Any

import pytest
from evalops import readiness as rd
from evalops import run_calibration as driver
from evalops.calibrate import (
    CalibrationBundle,
    CalibrationError,
    Paired,
    select_threshold,
)
from evalops.dataset import Annotation, CaseRecord, Corpus, Judgment, LabelProvenance
from evalops.experiments import JudgmentCache, _cache_key
from evalops.judge import JudgeConfig
from evalops.metrics import Confusion, confusion_of, kappa_of
from evalops.probe_combiners import group_folds
from evalops.splits import SplitPlan
from sklearn.metrics import cohen_kappa_score

FAST = rd.ReadinessRule(replicates=400)


# ---------------------------------------------------------------- arithmetic


def test_kappa_from_cells_matches_the_closed_form_on_integer_tables():
    rng = random.Random(7)
    tables = [(75, 16, 4, 33), (10, 0, 0, 10), (3, 9, 8, 1), (0, 5, 5, 0)]
    tables += [tuple(rng.randrange(0, 40) for _ in range(4)) for _ in range(40)]
    for tp, fp, fn, tn in tables:
        expected, *_ = kappa_of(Confusion(tp, fp, fn, tn))
        got = rd.kappa_from_cells(tp, fp, fn, tn)
        if expected is None:
            assert got is None
        else:
            assert got == pytest.approx(expected, abs=1e-12)


def test_kappa_from_cells_agrees_with_sklearn():
    human = [1] * 75 + [0] * 16 + [1] * 4 + [0] * 33
    judge = [1] * 75 + [1] * 16 + [0] * 4 + [0] * 33
    assert rd.kappa_from_cells(75, 16, 4, 33) == pytest.approx(cohen_kappa_score(human, judge))


@pytest.mark.parametrize("cells", [(10, 0, 0, 0), (0, 0, 0, 7), (0, 0, 0, 0)])
def test_kappa_from_cells_is_undefined_not_nan(cells):
    assert rd.kappa_from_cells(*cells) is None


def test_reweighting_reweights_the_table_not_the_kappas():
    cells = {"a": [40, 5, 5, 50], "b": [80, 8, 2, 10]}
    weights = {"a": 0.25, "b": 0.75}
    mixed = [0.25 * a / 100 + 0.75 * b / 100 for a, b in zip(cells["a"], cells["b"])]
    assert rd.reweighted_kappa(cells, weights) == pytest.approx(rd.kappa_from_cells(*mixed))
    naive = 0.25 * rd.kappa_from_cells(*cells["a"]) + 0.75 * rd.kappa_from_cells(*cells["b"])
    assert rd.reweighted_kappa(cells, weights) != pytest.approx(naive, abs=1e-3)


def test_reweighting_needs_evidence_for_every_weighted_stratum():
    assert rd.reweighted_kappa({"a": [5, 1, 1, 5]}, {"a": 0.5, "b": 0.5}) is None
    assert rd.reweighted_kappa({"a": [5, 1, 1, 5], "b": [0, 0, 0, 0]}, {"a": 0.5, "b": 0.5}) is None


# ---------------------------------------------------------------- the held-out shape


def _case(i: int, group: str, subset: str | None = "A", task: str = "summarize") -> CaseRecord:
    return CaseRecord(case_id=f"c{i:03d}", group_id=group, task_class=task,
                      task_input=f"summarize {i}", context="Source.", candidate_output=f"Claim {i}.",
                      meta={"subset": subset} if subset else {})


def test_stratum_prefers_the_published_subset_and_falls_back_to_task_class():
    assert rd.stratum_of(_case(1, "g", "TofuEval-MeetB")) == "TofuEval-MeetB"
    assert rd.stratum_of(_case(2, "g", None, task="sql")) == "sql"
    assert rd.stratum_of(_case(3, "g", "  ")) == "summarize"


def test_heldout_composition_counts_documents_and_items_per_stratum():
    cases = [_case(i, f"g{i // 4}", "M" if i < 12 else "C") for i in range(20)]
    comp = rd.heldout_composition(cases)
    assert comp.group_sizes == {"M": (4, 4, 4), "C": (4, 4)}
    assert comp.n == 20
    assert comp.weights() == {"M": pytest.approx(0.6), "C": pytest.approx(0.4)}
    assert comp.as_dict()["M"] == {"items": 12, "groups": 3, "weight": pytest.approx(0.6),
                                   "group_sizes": [4, 4, 4]}


def test_an_empty_heldout_population_is_an_error():
    with pytest.raises(rd.ReadinessError, match="empty"):
        rd.heldout_composition([])


def test_the_committed_heldout_headline_population_is_read_without_labels():
    """Pins the real composition the gate will describe, reading provenance only."""
    data = driver.DATA
    corpus = Corpus.load(data)
    sp = SplitPlan.load(data / "splits.json")
    test = [c for c in corpus.cases if sp.of_case(c) == "test"]
    prov = driver._annotation_provenance(corpus, {c.case_id for c in test})
    values = {p.value for p in (LabelProvenance.HUMAN_EXPERT, LabelProvenance.HUMAN_LOCAL)}
    population = [c for c in test if values & set(prov.get(c.case_id, []))]
    comp = rd.heldout_composition(population).as_dict()
    assert {s: (row["items"], row["groups"]) for s, row in comp.items()} == {
        "AggreFact-CNN": (40, 36), "TofuEval-MediaS": (80, 5), "TofuEval-MeetB": (143, 9),
    }


# ---------------------------------------------------------------- the honest estimate


def _noise(seed: int, n: int = 120, k: int = 12) -> tuple[list[int], list[str], list[rd.Candidate]]:
    rng = random.Random(seed)
    human = [rng.randrange(2) for _ in range(n)]
    groups = [f"g{i // 3}" for i in range(n)]
    cands = [rd.Candidate(f"v{j}", tuple(rng.random() for _ in range(n))) for j in range(k)]
    return human, groups, cands


def test_each_fold_is_predicted_without_its_own_labels():
    human, groups, cands = _noise(11)
    base, _ = rd.nested_predictions(human, groups, cands, folds=5)
    for fold in [f for f in group_folds(groups, 5) if f]:
        flipped = list(human)
        for i in fold:
            flipped[i] = 1 - flipped[i]
        again, _ = rd.nested_predictions(flipped, groups, cands, folds=5)
        assert [again[i] for i in fold] == [base[i] for i in fold]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_nested_estimate_removes_best_of_k_selection_optimism(seed):
    human, groups, cands = _noise(seed)
    everything = list(range(len(human)))
    in_sample_best = max(
        select_threshold(Paired(
            case_ids=[str(i) for i in everything], human=human, judge_scores=list(c.scores),
            groups=groups, tasks=[""] * len(human), provenance=[""] * len(human), split="dev",
        )).dev_kappa for c in cands)
    honest, log = rd.nested_predictions(human, groups, cands, folds=5)
    honest_kappa, *_ = kappa_of(confusion_of(human, honest))
    assert in_sample_best > 0.15               # pure noise looks like signal when selected
    assert honest_kappa is not None and honest_kappa < in_sample_best - 0.1
    assert len(log) == 5 and all(row["n_train"] >= 30 for row in log)


def test_nested_predictions_fail_closed_on_non_finite_scores():
    human, groups, cands = _noise(4)
    bad = list(cands[0].scores)
    bad[5] = float("nan")
    with pytest.raises(rd.ReadinessError, match="finite"):
        rd.nested_predictions(human, groups, [rd.Candidate("bad", tuple(bad))], folds=5)


def test_nested_predictions_need_two_document_folds():
    human, _, cands = _noise(5)
    with pytest.raises(rd.ReadinessError, match="two non-empty"):
        rd.nested_predictions(human, ["one-document"] * len(human), cands[:1], folds=5)


def test_nested_predictions_refuse_ragged_candidates():
    human, groups, cands = _noise(6)
    with pytest.raises(rd.ReadinessError, match="scores for"):
        rd.nested_predictions(human, groups, [rd.Candidate("short", cands[0].scores[:-1])], folds=5)


# ---------------------------------------------------------------- assessment


def _evidence(error_rate: float, *, seed: int = 3, strata: tuple[str, ...] = ("A", "B"),
              groups_per_stratum: int = 12, size: int = 6, honest: bool = True,
              unjudged: int = 0) -> rd.DevEvidence:
    rng = random.Random(seed)
    ids, human, groups, labels_strata, judge = [], [], [], [], []
    for s in strata:
        for g in range(groups_per_stratum):
            for k in range(size):
                h = rng.randrange(2)
                ids.append(f"{s}{g}-{k}")
                human.append(h)
                groups.append(f"{s}-{g}")
                labels_strata.append(s)
                judge.append(1 - h if rng.random() < error_rate else h)
    return rd.DevEvidence(case_ids=ids, human=human, groups=groups, strata=labels_strata,
                          in_sample=judge, honest=list(judge) if honest else None,
                          unjudged=unjudged, honest_note="" if honest else "not computed here")


def _composition(**sizes: tuple[int, ...]) -> rd.HeldoutComposition:
    return rd.HeldoutComposition(dict(sizes))


def test_strong_evidence_is_ready():
    out = rd.assess(_evidence(0.02), _composition(A=(10,) * 8, B=(6,) * 10), min_kappa=0.74,
                    rule=FAST)
    assert out["verdict"] == rd.READY, out["reasons"]
    assert out["estimates"]["p_pass_honest"] > 0.9
    assert out["estimates"]["reweighted_kappa_honest"] > 0.9


def test_weak_evidence_names_every_failed_condition():
    evidence = _evidence(0.22, strata=("A", "B", "C"), groups_per_stratum=12, unjudged=2)
    evidence.strata = ["B" if s == "C" and int(g.split("-")[1]) >= 3 else s
                       for s, g in zip(evidence.strata, evidence.groups)]  # C keeps 3 documents
    comp = _composition(A=(10,) * 6, B=(5,) * 4, C=(23, 23, 19, 18, 15, 13, 12, 11, 9), D=(4,) * 5)
    out = rd.assess(evidence, comp, min_kappa=0.74, rule=FAST)
    text = " | ".join(out["reasons"])
    assert out["verdict"] == rd.NOT_READY
    assert "dev coverage" in text
    assert "'D'" in text and "no dev evidence" in text
    assert "'C'" in text and "3 dev document(s)" in text
    assert "reweighted to the held-out mix is undefined" in text  # D has no evidence
    assert out["estimates"]["p_pass_honest"] is None


def test_a_light_stratum_with_few_documents_is_reported_not_blocking():
    evidence = _evidence(0.02, strata=("A", "B"), groups_per_stratum=12)
    evidence.strata = ["B" if s == "B" and int(g.split("-")[1]) < 2 else
                       ("A" if s == "B" else s) for s, g in zip(evidence.strata, evidence.groups)]
    out = rd.assess(evidence, _composition(A=(10,) * 20, B=(4,)), min_kappa=0.74, rule=FAST)
    assert out["verdict"] == rd.READY, out["reasons"]
    assert any("'B'" in note and "reported only" in note for note in out["notes"])


def test_without_an_honest_estimate_the_verdict_is_never_ready():
    out = rd.assess(_evidence(0.0, honest=False), _composition(A=(6,) * 10, B=(6,) * 10),
                    min_kappa=0.74, rule=FAST)
    assert out["verdict"] == rd.NOT_READY
    assert any("no honest" in r for r in out["reasons"])


def test_pass_probability_tracks_the_agreement_behind_it():
    comp = _composition(A=(10,) * 10, B=(10,) * 10)
    high = rd.assess(_evidence(0.03), comp, min_kappa=0.74, rule=FAST)["estimates"]
    low = rd.assess(_evidence(0.25), comp, min_kappa=0.74, rule=FAST)["estimates"]
    assert high["p_pass_honest"] > 0.9
    assert low["p_pass_honest"] < 0.05
    assert low["reweighted_kappa_honest"] < 0.74 < high["reweighted_kappa_honest"]


def test_assessment_is_deterministic():
    comp = _composition(A=(7,) * 9, B=(3,) * 9)
    assert rd.assess(_evidence(0.1), comp, min_kappa=0.74, rule=FAST) == \
        rd.assess(_evidence(0.1), comp, min_kappa=0.74, rule=FAST)


@pytest.mark.parametrize("breakage", ["ragged", "non-binary", "empty"])
def test_malformed_evidence_raises_instead_of_producing_a_verdict(breakage):
    evidence = _evidence(0.0)
    if breakage == "ragged":
        evidence.in_sample = evidence.in_sample[:-1]
    elif breakage == "non-binary":
        evidence.in_sample[0] = 2
    else:
        evidence = rd.DevEvidence([], [], [], [], [])
    with pytest.raises(rd.ReadinessError):
        rd.assess(evidence, _composition(A=(5,) * 5), min_kappa=0.74, rule=FAST)


# ---------------------------------------------------------------- the gate


BINDINGS = {"bundle_id": "b1", "variant_id": "v", "threshold": 0.5, "config_hash": "h",
            "dataset_hash": "d", "annotations_hash": "a", "split_membership_hash": "s",
            "rubric_bundle_hash": "r", "track": "headline"}


def _report(error_rate: float) -> dict[str, Any]:
    core = rd.assess(_evidence(error_rate), _composition(A=(10,) * 8, B=(6,) * 10),
                     min_kappa=0.74)
    return rd.finalize_report(core, bindings=BINDINGS, computed_at="t", software={"x": "y"})


def test_the_gate_accepts_a_consistent_ready_report():
    report = _report(0.02)
    assert report["verdict"] == rd.READY
    assert rd.gate_problems(report, bindings=BINDINGS, min_kappa=0.74) == []


def test_commentary_fields_do_not_change_the_report_identity():
    report = _report(0.02)
    report["computed_at"], report["software"] = "later", {"host": "elsewhere"}
    assert rd.gate_problems(report, bindings=BINDINGS, min_kappa=0.74) == []


def test_the_gate_rejects_a_not_ready_report_and_says_why():
    problems = rd.gate_problems(_report(0.3), bindings=BINDINGS, min_kappa=0.74)
    assert any("NOT_READY" in p and "below min_kappa" in p for p in problems)


def test_the_gate_rejects_a_report_edited_to_ready():
    report = _report(0.3)
    report["verdict"], report["reasons"] = rd.READY, []
    problems = rd.gate_problems(report, bindings=BINDINGS, min_kappa=0.74)
    assert any("edited after" in p for p in problems)
    assert any("self-inconsistent" in p for p in problems)


@pytest.mark.parametrize("key", sorted(BINDINGS))
def test_the_gate_rejects_a_report_about_another_bundle_or_corpus(key):
    report = _report(0.02)
    changed = dict(BINDINGS, **{key: "something else"})
    problems = rd.gate_problems(report, bindings=changed, min_kappa=0.74)
    assert any(key in p for p in problems)


def test_the_gate_rejects_a_weakened_rule_and_a_lower_min_kappa():
    core = rd.assess(_evidence(0.02), _composition(A=(10,) * 8, B=(6,) * 10), min_kappa=0.5)
    weak = rd.ReadinessRule(min_pass_probability=0.01)
    report = rd.finalize_report(core, bindings=BINDINGS, rule=weak)
    problems = rd.gate_problems(report, bindings=BINDINGS, min_kappa=0.74)
    assert any("different readiness rule" in p for p in problems)
    assert any("min_kappa" in p for p in problems)


@pytest.mark.parametrize("report", [None, [], "READY", {"verdict": "READY"}])
def test_the_gate_rejects_structural_nonsense(report):
    assert rd.gate_problems(report, bindings=BINDINGS, min_kappa=0.74)


@pytest.mark.parametrize("edit, expected", [
    (lambda r: r["dev_evidence"].__setitem__("unjudged", 3), "unjudged=3"),
    (lambda r: r["dev_evidence"]["by_stratum"]["A"].__setitem__("groups", 2), "'A'"),
    (lambda r: r["dev_evidence"]["by_stratum"].pop("B"), "'B'"),
    (lambda r: r.__setitem__("heldout_composition", {}), "no held-out composition"),
])
def test_the_gate_rederives_ready_from_the_recorded_evidence(edit, expected):
    """A recomputed report_id must not launder an edit: READY has to follow from the evidence."""
    report = _report(0.02)
    edit(report)
    report["report_id"] = rd.report_digest(report)
    problems = rd.gate_problems(report, bindings=BINDINGS, min_kappa=0.74)
    assert any(expected in p for p in problems), problems


# ---------------------------------------------------------------- the command path


def _experiment(tmp_path, monkeypatch, *, frozen_wrong, split_aware: bool = False,
                digest: str | None = None) -> SimpleNamespace:
    """120 summarize cases in 60 two-item documents; the first 60 items carry human labels."""
    if digest is not None:
        monkeypatch.setattr(Corpus, "annotations_hash", property(lambda self: digest))
    cases = [_case(i, f"g{i // 2:02d}", "A" if (i // 2) % 3 else "B") for i in range(120)]
    annotations = [Annotation(case_id=c.case_id, annotator_id="h", label=i % 2,
                              provenance=LabelProvenance.HUMAN_EXPERT if i < 60
                              else LabelProvenance.GOLD_ORACLE) for i, c in enumerate(cases)]
    corpus = Corpus(cases, annotations)
    plan = SimpleNamespace(seed=17, membership_hash="members")

    def split_of(case: CaseRecord) -> str:
        return "dev" if int(case.group_id[1:]) % 2 == 0 else "test"

    def fake_select(pool, _plan, split):
        return [c for c in pool if split_of(c) == split] if split_aware else list(pool)

    monkeypatch.setattr(driver, "DATA", tmp_path)
    monkeypatch.setattr(driver, "_load", lambda: (corpus, plan, corpus.cases_hash))
    monkeypatch.setattr(driver, "select", fake_select)
    frozen = JudgeConfig(variant_id="frozen", model="gpt-4o-mini", include_reference=False)
    rival = JudgeConfig(variant_id="rival", model="gpt-4o-mini", include_reference=True)
    cache = JudgmentCache(tmp_path / "judgment_cache.jsonl")
    for config, wrong in ((frozen, frozen_wrong), (rival, lambda i: i < 40)):
        for i, c in enumerate(cases):
            label = i % 2
            cache.put(_cache_key(c, config), Judgment(
                case_id=c.case_id, variant_id=config.variant_id, model_requested=config.model,
                config_hash=config.config_hash, raw_score=float(1 - label if wrong(i) else label)))
    record = {"dataset_hash": corpus.cases_hash, "annotations_hash": corpus.annotations_hash,
              "split_seed": plan.seed, "split_membership_hash": plan.membership_hash,
              "variants": [{"variant_id": c.variant_id, "judge_config": c.as_dict(),
                            "completion": 1.0} for c in (rival, frozen)]}
    (tmp_path / "dev_experiments.json").write_text(json.dumps(record), encoding="utf-8")
    freeze_args = SimpleNamespace(variant=None, replace=False, select_on="headline",
                                  min_kappa=0.74)
    return SimpleNamespace(path=tmp_path, corpus=corpus, cache=cache, frozen=frozen,
                           freeze_args=freeze_args, split_of=split_of)


def _test_args(**overrides: Any) -> SimpleNamespace:
    base = {"remeasure": False, "budget": 1, "concurrency": 1, "bootstrap": 50,
            "override_readiness": None}
    base.update(overrides)
    return SimpleNamespace(**base)


def _complete_inference(ex: SimpleNamespace) -> Any:
    def run(cases, config, **kwargs):
        return ([ex.cache.get(_cache_key(c, config)) for c in cases],
                SimpleNamespace(errors={}, n_ok=len(cases), as_dict=dict, models_served={}))
    return run


def _no_inference(*args, **kwargs):
    pytest.fail("the sealed split was sent for inference")


def test_ready_bundle_is_measured_and_the_clearance_is_recorded(tmp_path, monkeypatch):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    assert driver.cmd_readiness(SimpleNamespace()) == 0
    report = json.loads((ex.path / driver.READINESS_REPORT).read_text())
    assert report["verdict"] == rd.READY
    assert report["selection"]["candidates"] == ["rival", "frozen"]
    assert report["heldout_composition"]["A"]["items"] + \
        report["heldout_composition"]["B"]["items"] == 60
    monkeypatch.setattr(driver, "run_variant", _complete_inference(ex))
    assert driver.cmd_test(_test_args()) == 0
    result = CalibrationBundle.load(ex.path / "calibration_bundle.json").test_result
    assert result["readiness"]["override"] is False
    assert result["readiness"]["report_id"] == report["report_id"]


def test_weak_bundle_is_not_ready_and_the_split_stays_sealed(tmp_path, monkeypatch, capsys):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: i % 5 == 0)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    assert driver.cmd_readiness(SimpleNamespace()) == 1
    report = json.loads((ex.path / driver.READINESS_REPORT).read_text())
    assert report["verdict"] == rd.NOT_READY and report["reasons"]
    monkeypatch.setattr(driver, "run_variant", _no_inference)
    assert driver.cmd_test(_test_args()) == 1
    assert "stays sealed" in capsys.readouterr().err
    assert not CalibrationBundle.load(ex.path / "calibration_bundle.json").frozen


def test_readiness_holds_the_release_policy_even_if_freeze_lowered_the_bar(tmp_path, monkeypatch):
    """A bundle frozen at --min-kappa 0.3 still faces 0.74 in validate_bundle, so readiness too."""
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: i % 5 == 0)
    ex.freeze_args.min_kappa = 0.3
    assert driver.cmd_freeze(ex.freeze_args) == 0
    assert CalibrationBundle.load(ex.path / "calibration_bundle.json").min_kappa == 0.3
    assert driver.cmd_readiness(SimpleNamespace()) == 1
    report = json.loads((ex.path / driver.READINESS_REPORT).read_text())
    assert report["min_kappa"] == driver.ROUTEBENCH_MIN_KAPPA
    assert report["verdict"] == rd.NOT_READY
    monkeypatch.setattr(driver, "run_variant", _no_inference)
    assert driver.cmd_test(_test_args()) == 1


def test_test_refuses_without_a_readiness_report(tmp_path, monkeypatch):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    monkeypatch.setattr(driver, "run_variant", _no_inference)
    assert driver.cmd_test(_test_args()) == 1


def test_a_valid_report_for_another_bundle_clears_nothing(tmp_path, monkeypatch):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    assert driver.cmd_readiness(SimpleNamespace()) == 0
    path = ex.path / driver.READINESS_REPORT
    report = json.loads(path.read_text())
    report["bindings"]["bundle_id"] = "some-other-bundle"
    report["report_id"] = rd.report_digest(report)  # internally consistent, wrong bundle
    path.write_text(json.dumps(report))
    monkeypatch.setattr(driver, "run_variant", _no_inference)
    assert driver.cmd_test(_test_args()) == 1


def test_a_report_edited_to_ready_is_refused(tmp_path, monkeypatch):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: i % 5 == 0)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    assert driver.cmd_readiness(SimpleNamespace()) == 1
    path = ex.path / driver.READINESS_REPORT
    report = json.loads(path.read_text())
    report["verdict"], report["reasons"] = rd.READY, []
    path.write_text(json.dumps(report))
    monkeypatch.setattr(driver, "run_variant", _no_inference)
    assert driver.cmd_test(_test_args()) == 1


def test_a_human_override_is_measured_and_written_into_the_result(tmp_path, monkeypatch):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: i % 5 == 0)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    assert driver.cmd_readiness(SimpleNamespace()) == 1
    reason = "owner accepts a one-shot measurement to document the shortfall"
    monkeypatch.setattr(driver, "run_variant", _complete_inference(ex))
    assert driver.cmd_test(_test_args(override_readiness=reason)) == 0
    record = CalibrationBundle.load(ex.path / "calibration_bundle.json").test_result["readiness"]
    assert record["override"] is True and record["reason"] == reason
    assert any("NOT_READY" in p for p in record["problems"])


@pytest.mark.parametrize("reason", ["", "   ", "yes do it"])
def test_an_override_without_a_real_reason_is_refused(tmp_path, monkeypatch, reason):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: i % 5 == 0)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    assert driver.cmd_readiness(SimpleNamespace()) == 1
    monkeypatch.setattr(driver, "run_variant", _no_inference)
    assert driver.cmd_test(_test_args(override_readiness=reason)) == 1


def test_a_failed_readiness_run_leaves_no_report_that_could_clear_the_split(tmp_path, monkeypatch):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    assert driver.cmd_readiness(SimpleNamespace()) == 0
    first = ex.corpus.cases[0]
    ex.cache.put(_cache_key(first, ex.frozen), Judgment(
        case_id=first.case_id, variant_id="frozen", model_requested="gpt-4o-mini",
        config_hash=ex.frozen.config_hash, raw_score=float("nan")))
    with pytest.raises(CalibrationError, match="finite"):
        driver.cmd_readiness(SimpleNamespace())
    assert not (ex.path / driver.READINESS_REPORT).exists()
    assert list(ex.path.glob("readiness_report.previous.*.json"))
    monkeypatch.setattr(driver, "run_variant", _no_inference)
    assert driver.cmd_test(_test_args()) == 1


def test_readiness_refuses_a_bundle_that_was_already_measured(tmp_path, monkeypatch):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    path = ex.path / "calibration_bundle.json"
    bundle = CalibrationBundle.load(path)
    bundle.test_result = {"tracks": {}}
    bundle.save(path)
    assert driver.cmd_readiness(SimpleNamespace()) == 1
    assert not (ex.path / driver.READINESS_REPORT).exists()


class _SealedAnnotation:
    """A held-out annotation whose label value must never be read."""

    def __init__(self, a: Annotation) -> None:
        self.case_id, self.provenance = a.case_id, a.provenance
        self.annotator_id, self.adjudicated = a.annotator_id, a.adjudicated

    @property
    def label(self) -> int:
        raise AssertionError(f"held-out label of {self.case_id} was read")


def test_readiness_never_reads_a_heldout_label(tmp_path, monkeypatch):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False, split_aware=True,
                     digest="labels-digest")
    assert driver.cmd_freeze(ex.freeze_args) == 0
    by_id = {c.case_id: c for c in ex.corpus.cases}
    ex.corpus.annotations[:] = [
        _SealedAnnotation(a) if ex.split_of(by_id[a.case_id]) == "test" else a
        for a in ex.corpus.annotations
    ]
    driver.cmd_readiness(SimpleNamespace())  # verdict irrelevant; reading a sealed label raises
    report = json.loads((ex.path / driver.READINESS_REPORT).read_text())
    assert sum(row["items"] for row in report["heldout_composition"].values()) == 30


def test_main_wires_the_readiness_command(tmp_path, monkeypatch):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    assert driver.main(["readiness"]) == 0
    monkeypatch.setattr(driver, "run_variant", _complete_inference(ex))
    assert driver.main(["test", "--bootstrap", "50"]) == 0


def test_self_consistent_forgery_is_rejected_against_source_evidence(tmp_path, monkeypatch, capsys):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: i % 5 == 0)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    assert driver.cmd_readiness(SimpleNamespace()) == 1
    path = ex.path / driver.READINESS_REPORT
    report = json.loads(path.read_text())
    report.update(verdict=rd.READY, reasons=[])
    for key in ("reweighted_kappa_in_sample", "reweighted_kappa_honest", "p_pass_honest"):
        report["estimates"][key] = 1.0
    report["report_id"] = rd.report_digest(report)
    assert not rd.gate_problems(report, bindings=report["bindings"], min_kappa=0.74)
    path.write_text(json.dumps(report))
    monkeypatch.setattr(driver, "run_variant", _no_inference)
    assert driver.cmd_test(_test_args()) == 1
    assert "does not reproduce" in capsys.readouterr().err


def test_freeze_binds_the_actual_override_and_selection_track(tmp_path, monkeypatch):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False)
    ex.freeze_args.variant = "frozen"
    assert driver.cmd_freeze(ex.freeze_args) == 0
    bundle = CalibrationBundle.load(ex.path / "calibration_bundle.json")
    assert bundle.selection_record["track"] == "headline"
    assert bundle.selection_record["variant_override"] == "frozen"
    assert driver.cmd_readiness(SimpleNamespace()) == 0
    report = json.loads((ex.path / driver.READINESS_REPORT).read_text())
    assert report["selection"]["candidates"] == ["frozen"]
    bundle.selection_record["track"] = "pooled"
    bundle.save(ex.path / "calibration_bundle.json")
    assert driver.cmd_readiness(SimpleNamespace()) == 1  # identity detects the edit


def test_readiness_does_not_claim_to_replay_pooled_selection(tmp_path, monkeypatch):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False)
    ex.freeze_args.select_on = "pooled"
    assert driver.cmd_freeze(ex.freeze_args) == 0
    assert driver.cmd_readiness(SimpleNamespace()) == 1
    report = json.loads((ex.path / driver.READINESS_REPORT).read_text())
    assert report["estimates"]["reweighted_kappa_honest"] is None


def test_missing_bound_candidate_cannot_silently_reduce_selection_set(tmp_path, monkeypatch):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    path = ex.path / "judgment_cache.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("\n".join(json.dumps(e) for e in entries if e["judgment"]["variant_id"] == "frozen") + "\n")
    assert driver.cmd_readiness(SimpleNamespace()) == 1
    report = json.loads((ex.path / driver.READINESS_REPORT).read_text())
    assert report["selection"]["excluded"]
    assert report["estimates"]["reweighted_kappa_honest"] is None


def test_unresolved_test_label_cannot_reduce_denominator_even_with_override(tmp_path, monkeypatch, capsys):
    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False, split_aware=True)
    cid = next(c.case_id for c in ex.corpus.cases if ex.split_of(c) == "test")
    original = next(a for a in ex.corpus.annotations if a.case_id == cid)
    ex.corpus.annotations.append(Annotation(case_id=cid, annotator_id="reviewer", label=1-original.label,
                                             provenance=LabelProvenance.HUMAN_LOCAL))
    record_path = ex.path / "dev_experiments.json"
    record = json.loads(record_path.read_text())
    record["annotations_hash"] = ex.corpus.annotations_hash  # synthetic pre-freeze conflict
    record_path.write_text(json.dumps(record))
    assert driver.cmd_freeze(ex.freeze_args) == 0
    monkeypatch.setattr(driver, "run_variant", _no_inference)
    assert driver.cmd_test(_test_args(override_readiness="Owner requests a diagnostic measurement only")) == 1
    assert "acceptance population cannot silently shrink" in capsys.readouterr().err


def test_cli_freeze_defaults_to_headline(monkeypatch):
    received = []
    monkeypatch.setattr(driver, "cmd_freeze", lambda args: received.append(args.select_on) or 0)
    assert driver.main(["freeze"]) == 0
    assert received == ["headline"]


@pytest.mark.parametrize("lineage", [None, [], "headline", {"track": "headline", "candidate_configs": "bad"},
    {"track": "headline", "rule": "wrong-rule", "candidate_configs": [{}]},
    {"track": "headline", "rule": "max-dev-kappa; candidate-order ties; select_threshold-v1", "candidate_configs": [None]}])
def test_malformed_selection_lineage_cannot_clear(tmp_path, monkeypatch, lineage):
    from evalops.calibrate import expected_bundle_id

    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    path = ex.path / "calibration_bundle.json"
    bundle = CalibrationBundle.load(path)
    bundle.selection_record = lineage
    bundle.bundle_id = expected_bundle_id(bundle)
    bundle.save(path)
    assert driver.cmd_readiness(SimpleNamespace()) == 1


def test_missing_selection_lineage_keeps_historical_identity_but_not_readiness(tmp_path, monkeypatch):
    from evalops.calibrate import expected_bundle_id

    ex = _experiment(tmp_path, monkeypatch, frozen_wrong=lambda i: False)
    assert driver.cmd_freeze(ex.freeze_args) == 0
    path = ex.path / "calibration_bundle.json"
    bundle = CalibrationBundle.load(path)
    bundle.selection_record = {}
    historical_id = expected_bundle_id(bundle)
    bundle.bundle_id = historical_id
    bundle.save(path)
    raw = json.loads(path.read_text())
    raw.pop("selection_record")
    path.write_text(json.dumps(raw))
    assert expected_bundle_id(CalibrationBundle.load(path)) == historical_id
    assert driver.cmd_readiness(SimpleNamespace()) == 1

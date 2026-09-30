"""Offline command-path regressions; no inference, network or held-out corpus access."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from evalops import run_calibration as driver
from evalops.calibrate import (
    CalibrationBundle,
    CalibrationError,
    Paired,
    select_threshold,
)
from evalops.checkers import DEFAULT_SPECS, FakeSupportChecker, check_case, load_checker
from evalops.dataset import Annotation, CaseRecord, Corpus, Judgment, LabelProvenance
from evalops.experiments import JudgmentCache, SpendMeter, _cache_key
from evalops.judge import JudgeConfig


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    cases = [CaseRecord(case_id=f"c{i}", group_id=f"g{i}", task_class="summarize",
                        task_input=f"summarize {i}", context="Source.",
                        candidate_output="First claim. Second claim.") for i in range(120)]
    annotations = [Annotation(case_id=c.case_id, annotator_id="human", label=i % 2,
                              provenance=LabelProvenance.HUMAN_EXPERT if i < 40
                              else LabelProvenance.GOLD_ORACLE) for i, c in enumerate(cases)]
    corpus = Corpus(cases, annotations)
    plan = SimpleNamespace(seed=17, membership_hash="members")
    monkeypatch.setattr(driver, "DATA", tmp_path)
    monkeypatch.setattr(driver, "_load", lambda: (corpus, plan, corpus.cases_hash))
    monkeypatch.setattr(driver, "select", lambda cases, plan, split: cases)
    configs = [JudgeConfig(variant_id=n, model="gpt-4o-mini", include_reference=(i == 0))
               for i, n in enumerate(("pooled", "headline"))]
    cache = JudgmentCache(tmp_path / "judgment_cache.jsonl")
    for config in configs:
        for i, c in enumerate(cases):
            label = i % 2
            wrong = (i < 8 if config.variant_id == "pooled" else i >= 40)
            cache.put(_cache_key(c, config), Judgment(
                case_id=c.case_id, variant_id=config.variant_id, model_requested=config.model,
                config_hash=config.config_hash, raw_score=float(1 - label if wrong else label)))
    record = {"dataset_hash": corpus.cases_hash, "annotations_hash": corpus.annotations_hash,
              "split_seed": plan.seed, "split_membership_hash": plan.membership_hash,
              "variants": [{"variant_id": c.variant_id, "judge_config": c.as_dict(),
                            "completion": 1.0, "best_dev_kappa": 0.99 if i == 0 else 0.1}
                           for i, c in enumerate(configs)]}
    (tmp_path / "dev_experiments.json").write_text(json.dumps(record), encoding="utf-8")
    args = SimpleNamespace(variant=None, replace=False, select_on="headline", min_kappa=0.74)
    return tmp_path, corpus, plan, configs, args


def test_freeze_selects_variant_and_threshold_on_requested_track(experiment):
    path, _, _, _, args = experiment
    assert driver.cmd_freeze(args) == 0
    bundle = CalibrationBundle.load(path / "calibration_bundle.json")
    assert bundle.variant_id == "headline"
    assert bundle.dev_n == 40
    assert bundle.dev_kappa == pytest.approx(1.0)


def test_freeze_pooled_still_selects_pooled_winner(experiment):
    path, _, _, _, args = experiment
    args.select_on = "pooled"
    assert driver.cmd_freeze(args) == 0
    assert CalibrationBundle.load(path / "calibration_bundle.json").variant_id == "pooled"


@pytest.mark.parametrize("field", ["annotations_hash", "split_membership_hash", "dataset_hash"])
def test_freeze_refuses_stale_record(experiment, field):
    path, _, _, _, args = experiment
    record_path = path / "dev_experiments.json"
    record = json.loads(record_path.read_text())
    record[field] = "changed"
    record_path.write_text(json.dumps(record))
    with pytest.raises(CalibrationError, match=field):
        driver.cmd_freeze(args)
    assert not (path / "calibration_bundle.json").exists()


def test_freeze_checks_overwrite_before_loading_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "DATA", tmp_path)
    monkeypatch.setattr(driver, "overwrite_refusal", lambda *a, **k: "do not overwrite")
    monkeypatch.setattr(driver, "_load", lambda: pytest.fail("loaded corpus before refusal"))
    assert driver.cmd_freeze(SimpleNamespace(replace=False)) == 1


@pytest.mark.parametrize("field", ["annotations_hash", "split_membership_hash", "threshold"])
def test_test_refuses_changed_bindings_before_inference(experiment, monkeypatch, field):
    path, _, _, _, args = experiment
    assert driver.cmd_freeze(args) == 0
    bundle_path = path / "calibration_bundle.json"
    bundle = CalibrationBundle.load(bundle_path)
    setattr(bundle, field, 0.9 if field == "threshold" else "changed")
    bundle.save(bundle_path)
    monkeypatch.setattr(driver, "run_variant", lambda *a, **k: pytest.fail("inference called"))
    assert driver.cmd_test(SimpleNamespace()) == 1


@pytest.mark.parametrize("change", ["labels", "membership"])
def test_actual_data_drift_is_rejected_with_untampered_bundle(experiment, monkeypatch, change):
    _, corpus, plan, _, args = experiment
    assert driver.cmd_freeze(args) == 0
    if change == "labels":
        corpus.annotations[0].label = 1 - corpus.annotations[0].label
    else:
        plan.membership_hash = "different membership"
    monkeypatch.setattr(driver, "run_variant", lambda *a, **k: pytest.fail("inference called"))
    assert driver.cmd_test(SimpleNamespace()) == 1


def test_targeted_dev_archives_record_and_keeps_only_requested_population(experiment, monkeypatch):
    path, _, _, configs, _ = experiment
    monkeypatch.setattr(driver, "GRID", configs)
    seen = []

    def fake_run(cases, config, **kwargs):
        seen.append((config.variant_id, len(cases)))
        cache = kwargs["cache"]
        return ([cache.get(_cache_key(c, config)) for c in cases],
                SimpleNamespace(models_served={}, errors={}, as_dict=dict))

    monkeypatch.setattr(driver, "run_variant", fake_run)
    before = (path / "dev_experiments.json").read_text()
    assert driver.cmd_dev(SimpleNamespace(select_on="headline", variants=["headline"],
                                          budget=0, concurrency=1)) == 0
    assert seen == [("headline", 40)]
    archived = list(path.glob("dev_experiments.previous.*.json"))
    assert len(archived) == 1
    assert archived[0].read_text() == before
    record = json.loads((path / "dev_experiments.json").read_text())
    assert record["selection_track"] == "headline"
    assert len(record["variants"]) == 1


def test_dev_quota_stops_later_variants_and_reports_failure(experiment, monkeypatch):
    path, _, _, configs, _ = experiment
    monkeypatch.setattr(driver, "GRID", configs)
    calls = []

    def exhausted(cases, config, **kwargs):
        calls.append(config.variant_id)
        return [], SimpleNamespace(errors={"quota_exhausted": 1}, models_served={}, as_dict=dict)

    monkeypatch.setattr(driver, "run_variant", exhausted)
    assert driver.cmd_dev(SimpleNamespace(select_on="headline", variants=None,
                                          budget=1, concurrency=1)) == 3
    assert calls == [configs[0].variant_id]
    record = json.loads((path / "dev_experiments.json").read_text())
    assert record["blocked"] == "quota_exhausted"


def test_test_quota_preserves_frozen_bundle_and_existing_report(experiment, monkeypatch):
    path, _, _, _, args = experiment
    assert driver.cmd_freeze(args) == 0
    bundle_path = path / "calibration_bundle.json"
    before = bundle_path.read_bytes()
    report = path / "calibration_report.json"
    report.write_text("existing-report")
    monkeypatch.setattr(driver, "run_variant", lambda *a, **k: (
        [], SimpleNamespace(errors={"quota_exhausted": 1})))
    assert driver.cmd_test(SimpleNamespace(remeasure=False, budget=1, concurrency=1)) == 3
    assert bundle_path.read_bytes() == before
    assert report.read_text() == "existing-report"


@pytest.mark.parametrize("summary", [
    SimpleNamespace(errors={"provider_error": 2}, n_ok=38),
    SimpleNamespace(errors={}, n_ok=10),  # budget stop: fewer judged than in scope, no error rows
])
def test_test_incomplete_run_never_publishes_terminal_result(experiment, monkeypatch, summary):
    path, _, _, _, args = experiment
    assert driver.cmd_freeze(args) == 0
    bundle_path = path / "calibration_bundle.json"
    before = bundle_path.read_bytes()
    report = path / "calibration_report.json"
    report.write_text("existing-report")
    monkeypatch.setattr(driver, "run_variant", lambda *a, **k: ([], summary))
    assert driver.cmd_test(SimpleNamespace(remeasure=False, budget=1, concurrency=1)) == 3
    assert bundle_path.read_bytes() == before
    assert report.read_text() == "existing-report"


def test_plan_uses_only_selected_variant_and_headline_cases(experiment, monkeypatch, capsys):
    _, _, _, configs, _ = experiment
    monkeypatch.setattr(driver, "GRID", configs)
    monkeypatch.setattr(driver, "run_variant", lambda *a, **k: pytest.fail("inference called"))
    assert driver.cmd_plan(SimpleNamespace(select_on="headline", variants=["headline"])) == 0
    output = json.loads(capsys.readouterr().out)
    assert len(output["variants"]) == 1
    assert output["variants"][0]["cases"] == 40
    assert output["variants"][0]["missing_current_judgments"] == 0


def test_unknown_variant_fails_closed():
    with pytest.raises(CalibrationError, match="unknown variants"):
        driver._requested_grid(SimpleNamespace(variants=["typo"]))


def test_strict_all_supported_threshold_is_available():
    p = Paired(case_ids=[f"c{i}" for i in range(40)], human=[0, 1] * 20,
               judge_scores=[0.96, 1.0] * 20, groups=[f"g{i}" for i in range(40)],
               tasks=["summarize"] * 40, provenance=["human_expert_annotation"] * 40,
               split="dev")
    best = select_threshold(p)
    assert best.threshold == 1.0
    assert best.dev_kappa == pytest.approx(1.0)


def test_alignscore_passes_whole_response_to_its_own_segmenter(experiment):
    _, corpus, _, _, _ = experiment
    spec = next(s for s in DEFAULT_SPECS if s.backend == "alignscore")
    checker = FakeSupportChecker(default=0.8)
    result = check_case(corpus.cases[0], spec, checker,
                        splitter=lambda text: pytest.fail("external segmenter called"))
    assert result.usable
    assert checker.calls[0][1] == (corpus.cases[0].candidate_output,)
    assert result.segmentation["splitter"] == "none"


@pytest.mark.parametrize("device", ["cpu", -1])
def test_alignscore_cpu_device_contract(monkeypatch, device):
    import sys
    calls = []
    monkeypatch.setitem(sys.modules, "alignscore", SimpleNamespace(
        AlignScore=lambda **kw: calls.append(kw)))
    spec = replace(next(s for s in DEFAULT_SPECS if s.backend == "alignscore"),
                   weights_licence_checked=True, ckpt_path="checkpoint")
    load_checker(spec, device=device)
    assert calls[0]["device"] == "cpu"


def test_budget_waits_for_inflight_reservations_instead_of_stopping():
    meter = SpendMeter(cap_usd=0.03)
    first = meter.reserve(worst_case_usd=0.02)
    started = threading.Event()

    def next_case():
        started.set()
        with meter.reserve(worst_case_usd=0.02, wait=True) as slot:
            slot.record(0.005, model="test")

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(next_case)
        assert started.wait(2)
        first.record(0.005, model="test")
        future.result(timeout=2)
    assert meter.calls == 2
    assert meter.spent_usd == pytest.approx(0.01)
    assert meter.reserved_usd == pytest.approx(0)

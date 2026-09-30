"""Offline safety regressions for the one-shot attribution experiment."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from evalops import attribution as attr
from evalops.calibrate import (
    CalibrationBundle,
    CalibrationError,
    freeze,
    select_threshold,
)
from evalops.dataset import Annotation, CaseRecord, Corpus, Judgment, LabelProvenance
from evalops.experiments import JudgmentCache, RunSummary, _cache_key
from evalops.judge import JudgeConfig


def _case(i: int, *, context: str = "Source.") -> CaseRecord:
    return CaseRecord(
        case_id=f"c{i}", group_id=f"g{i}", task_class="summarize",
        task_input="verify", context=context, candidate_output=f"claim {i}",
    )


def _experiment(tmp_path, monkeypatch, *, n: int = 40):
    cases = [_case(i) for i in range(n)]
    annotations = [
        Annotation(
            case_id=case.case_id, annotator_id="human", label=i % 2,
            provenance=LabelProvenance.HUMAN_EXPERT,
        )
        for i, case in enumerate(cases)
    ]
    corpus = Corpus(cases, annotations)
    split_plan = SimpleNamespace(seed=17, membership_hash="membership-v1")
    monkeypatch.setattr(attr, "DATA", tmp_path)
    monkeypatch.setattr(attr, "_load", lambda: (corpus, split_plan, corpus.cases_hash))
    monkeypatch.setattr(attr, "select", lambda cases, plan, split: list(cases))
    protocol = tmp_path / "new-protocol.md"
    protocol.write_text("predeclared full-source protocol", encoding="utf-8")
    (tmp_path / "build_manifest.json").write_text(json.dumps({
        "protocol_path": str(protocol),
        "protocol_hash": attr.content_hash(protocol.read_text(encoding="utf-8")),
        "source_policy": "full source retained; no truncation",
    }), encoding="utf-8")
    return corpus, split_plan


def _judgments(cases, config, *, limit=None):
    chosen = cases if limit is None else cases[:limit]
    return [
        Judgment(
            case_id=case.case_id, variant_id=config.variant_id,
            model_requested=config.model, config_hash=config.config_hash,
            raw_score=float(i % 2),
        )
        for i, case in enumerate(chosen)
    ]


def _summary(config, n, *, errors=None):
    return RunSummary(
        variant_id=config.variant_id, config=config.as_dict(), n_cases=n,
        n_ok=n, n_cached=0, errors=errors or {}, spend={}, wall_s=0,
        models_served={},
    )


def _write_unmeasured_bundle(tmp_path, corpus, split_plan, config):
    dev = attr.pair(
        corpus.cases, corpus.resolved_labels(), _judgments(corpus.cases, config), split="dev"
    )
    frozen_config = config.as_dict()
    frozen_config["experiment_protocol"] = attr._protocol_binding()
    bundle = freeze(
        select_threshold(dev), variant_id=config.variant_id,
        judge_config=frozen_config, dataset_hash=corpus.cases_hash,
        split_seed=split_plan.seed, dev=dev,
        annotations_hash=corpus.annotations_hash,
        split_membership_hash=split_plan.membership_hash,
    )
    return bundle.save(tmp_path / "calibration_bundle.json")


def test_build_refuses_any_existing_experiment_artifact_before_work(tmp_path, monkeypatch):
    monkeypatch.setattr(attr, "DATA", tmp_path)
    (tmp_path / "cases.jsonl").write_text("user data", encoding="utf-8")
    monkeypatch.setattr(attr, "build", lambda: pytest.fail("build touched existing experiment"))

    assert attr.cmd_build(SimpleNamespace(protocol="new.md")) == 1
    assert (tmp_path / "cases.jsonl").read_text(encoding="utf-8") == "user data"


def test_fresh_builder_retains_full_sources_and_excludes_exposed_docs(tmp_path, monkeypatch):
    long_source = "supported evidence " * 400
    exposed_source = "already examined source"
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(row) for row in [
        {"subset": "Reveal", "label": 1, "claim": "exposed",
         "doc": exposed_source + " trailing raw text"},
        {"subset": "Reveal", "label": 1, "claim": "safe", "doc": long_source},
        {"subset": "Reveal", "label": 0, "claim": "false", "doc": "other source"},
    ]), encoding="utf-8")
    exposed = tmp_path / "prior.jsonl"
    exposed.write_text(json.dumps({
        "context": exposed_source, "meta": {"subset": "Reveal"},
    }), encoding="utf-8")
    monkeypatch.setattr(attr, "RAW", raw)
    monkeypatch.setattr(attr, "EXPOSED_CASES", (exposed,))
    monkeypatch.setattr(attr, "SUBSETS", ("Reveal",))
    monkeypatch.setattr(attr, "PER_CLASS", 1)

    cases, _, audit = attr.build()

    assert exposed_source not in {case.context for case in cases}
    assert long_source.strip() in {case.context for case in cases}
    assert max(len(case.context) for case in cases) > 6000
    assert audit["excluded_prior_exposure_documents"] == 1
    assert audit["excluded_prior_exposure_rows"] == 1
    assert audit["retained_over_6000_chars"] == 1


def test_fresh_builder_fails_when_exposure_exclusion_breaks_class_quota(tmp_path, monkeypatch):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(row) for row in [
        {"subset": "Reveal", "label": 1, "claim": "only", "doc": "exposed"},
        {"subset": "Reveal", "label": 0, "claim": "false", "doc": "safe"},
    ]), encoding="utf-8")
    exposed = tmp_path / "prior.jsonl"
    exposed.write_text(json.dumps({
        "context": "exposed", "meta": {"subset": "Reveal"},
    }), encoding="utf-8")
    monkeypatch.setattr(attr, "RAW", raw)
    monkeypatch.setattr(attr, "EXPOSED_CASES", (exposed,))
    monkeypatch.setattr(attr, "SUBSETS", ("Reveal",))
    monkeypatch.setattr(attr, "PER_CLASS", 1)

    with pytest.raises(CalibrationError, match="only 0 eligible unique rows"):
        attr.build()


def test_dev_archives_record_records_full_provenance_and_honors_cli_concurrency(
    tmp_path, monkeypatch,
):
    corpus, _ = _experiment(tmp_path, monkeypatch)
    config = JudgeConfig(variant_id="candidate", model="gpt-4o-mini", concurrency=3)
    monkeypatch.setattr(attr, "CANDIDATES", (config.variant_id,))
    monkeypatch.setattr(attr, "GRID", (config,))
    old = '{"old": true}'
    (tmp_path / "dev_experiments.json").write_text(old, encoding="utf-8")
    seen = []

    def fake_run(cases, cfg, **kwargs):
        seen.append(kwargs["concurrency"])
        return _judgments(cases, cfg), _summary(cfg, len(cases))

    monkeypatch.setattr(attr, "run_variant", fake_run)
    assert attr.cmd_dev(SimpleNamespace(budget=0, concurrency=7)) == 0

    assert seen == [7]
    archived = list(tmp_path.glob("dev_experiments.previous.*.json"))
    assert len(archived) == 1 and archived[0].read_text(encoding="utf-8") == old
    record = json.loads((tmp_path / "dev_experiments.json").read_text(encoding="utf-8"))
    row = record["variants"][0]
    assert record["annotations_hash"] == corpus.annotations_hash
    assert record["split_membership_hash"] == "membership-v1"
    assert row["judge_config"]["config_hash"] == config.config_hash
    assert row["prompt_template_hash"]
    assert row["best_threshold"] == 0.5  # shared closest-to-0.5 tie policy
    assert row["threshold_grid"]
    assert record["run_summaries"][0]["config"]["variant_id"] == "candidate"


def test_dev_quota_stops_later_candidates_and_returns_nonzero(tmp_path, monkeypatch):
    _experiment(tmp_path, monkeypatch)
    configs = (
        JudgeConfig(variant_id="one", model="gpt-4o-mini"),
        JudgeConfig(variant_id="two", model="gpt-4o-mini", include_reference=False),
    )
    monkeypatch.setattr(attr, "CANDIDATES", tuple(c.variant_id for c in configs))
    monkeypatch.setattr(attr, "GRID", configs)
    calls = []

    def quota(cases, config, **kwargs):
        calls.append(config.variant_id)
        return [], _summary(config, len(cases), errors={"quota_exhausted": 1})

    monkeypatch.setattr(attr, "run_variant", quota)
    assert attr.cmd_dev(SimpleNamespace(budget=1, concurrency=1)) == 3
    assert calls == ["one"]
    record = json.loads((tmp_path / "dev_experiments.json").read_text(encoding="utf-8"))
    assert record["blocked"] == "quota_exhausted"


def test_freeze_recomputes_from_recorded_config_cache_and_binds_all_data(tmp_path, monkeypatch):
    corpus, split_plan = _experiment(tmp_path, monkeypatch)
    recorded = JudgeConfig(
        variant_id="candidate", model="gpt-4o-mini", include_reference=False,
    )
    live_grid = JudgeConfig(
        variant_id="candidate", model="gpt-4o-mini", include_reference=True,
    )
    monkeypatch.setattr(attr, "CANDIDATES", (recorded.variant_id,))
    monkeypatch.setattr(attr, "GRID", (live_grid,))
    cache = JudgmentCache(tmp_path / "judgment_cache.jsonl")
    for case, judgment in zip(corpus.cases, _judgments(corpus.cases, recorded)):
        cache.put(_cache_key(case, recorded), judgment)
    record = {
        "dataset_hash": corpus.cases_hash,
        "annotations_hash": corpus.annotations_hash,
        "split_seed": split_plan.seed,
        "split_membership_hash": split_plan.membership_hash,
        "protocol": attr._protocol_binding(),
        "blocked": None,
        "variants": [{
            "variant_id": recorded.variant_id,
            "config_hash": recorded.config_hash,
            "judge_config": recorded.as_dict(),
            "prompt_template_hash": attr.prompt_template_hash(recorded.mode),
            # Deliberately stale; freeze must recompute from cached judgments.
            "best_threshold": 0.95,
            "best_dev_kappa": -1,
        }],
    }
    (tmp_path / "dev_experiments.json").write_text(json.dumps(record), encoding="utf-8")

    assert attr.cmd_freeze(SimpleNamespace(min_kappa=0.74)) == 0
    bundle = CalibrationBundle.load(tmp_path / "calibration_bundle.json")
    assert bundle.judge_config["include_reference"] is False
    assert bundle.threshold == 0.5
    assert bundle.dev_kappa == pytest.approx(1.0)
    assert bundle.annotations_hash == corpus.annotations_hash
    assert bundle.split_membership_hash == split_plan.membership_hash


def test_old_incomplete_measurement_refuses_refreeze_and_retest_before_any_work(
    tmp_path, monkeypatch,
):
    corpus, split_plan = _experiment(tmp_path, monkeypatch)
    config = JudgeConfig(variant_id="candidate", model="gpt-4o-mini")
    path = _write_unmeasured_bundle(tmp_path, corpus, split_plan, config)
    bundle = CalibrationBundle.load(path)
    bundle.test_result = {"completion": {"completion_rate": 0.9}, "tracks": {}}
    bundle.save(path)
    monkeypatch.setattr(attr, "_load", lambda: pytest.fail("loaded data after terminal refusal"))
    monkeypatch.setattr(attr, "run_variant", lambda *a, **k: pytest.fail("inference called"))

    assert attr.cmd_freeze(SimpleNamespace(min_kappa=0.74)) == 1
    assert attr.cmd_test(SimpleNamespace(budget=1, concurrency=1, bootstrap=0)) == 1


def test_test_refuses_binding_drift_before_inference(tmp_path, monkeypatch):
    corpus, split_plan = _experiment(tmp_path, monkeypatch)
    config = JudgeConfig(variant_id="candidate", model="gpt-4o-mini")
    path = _write_unmeasured_bundle(tmp_path, corpus, split_plan, config)
    bundle = CalibrationBundle.load(path)
    bundle.annotations_hash = "edited-labels"
    # Recompute the id so this specifically tests the live corpus binding, not tamper detection.
    from evalops.calibrate import expected_bundle_id
    bundle.bundle_id = expected_bundle_id(bundle)
    bundle.save(path)
    monkeypatch.setattr(attr, "run_variant", lambda *a, **k: pytest.fail("inference called"))

    assert attr.cmd_test(SimpleNamespace(budget=1, concurrency=1, bootstrap=0)) == 1


@pytest.mark.parametrize("failure", ["quota", "incomplete"])
def test_failed_fresh_test_retains_cache_but_publishes_no_terminal_artifacts(
    tmp_path, monkeypatch, failure,
):
    corpus, split_plan = _experiment(tmp_path, monkeypatch)
    config = JudgeConfig(variant_id="candidate", model="gpt-4o-mini", concurrency=3)
    bundle_path = _write_unmeasured_bundle(tmp_path, corpus, split_plan, config)
    before = bundle_path.read_bytes()
    seen = []

    def fail_run(cases, cfg, **kwargs):
        seen.append(kwargs["concurrency"])
        if failure == "quota":
            return [], _summary(cfg, len(cases), errors={"quota_exhausted": 1})
        return (
            _judgments(cases, cfg, limit=len(cases) - 1),
            _summary(cfg, len(cases), errors={"provider_error": 1}),
        )

    monkeypatch.setattr(attr, "run_variant", fail_run)
    assert attr.cmd_test(SimpleNamespace(budget=1, concurrency=6, bootstrap=0)) == 3
    assert seen == [6]
    assert bundle_path.read_bytes() == before
    assert not (tmp_path / "calibration_report.json").exists()


def test_complete_test_publishes_only_after_full_coverage(tmp_path, monkeypatch):
    corpus, split_plan = _experiment(tmp_path, monkeypatch)
    config = JudgeConfig(variant_id="candidate", model="gpt-4o-mini")
    _write_unmeasured_bundle(tmp_path, corpus, split_plan, config)

    monkeypatch.setattr(
        attr,
        "run_variant",
        lambda cases, cfg, **kwargs: (_judgments(cases, cfg), _summary(cfg, len(cases))),
    )
    assert attr.cmd_test(SimpleNamespace(budget=1, concurrency=4, bootstrap=0)) == 0
    assert CalibrationBundle.load(tmp_path / "calibration_bundle.json").frozen
    assert (tmp_path / "calibration_report.json").exists()

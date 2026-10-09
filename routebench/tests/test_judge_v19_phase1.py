"""Tests for the judge-v19 Phase 1 screening protocol and runner. Mocked provider and fake MiniCheck checker only: no provider call, no GPU, no download."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from aisys import llm
from evalops import judge_v19_analysis as an
from evalops import judge_v19_minicheck as mcjob
from evalops import judge_v19_phase1 as jv
from evalops import matched_v16_v18 as mv
from evalops import rubrics

CRITERIA = sorted(rubrics.for_task("summarize").criterion_ids)
SMALL_CAPS = {"1a": {"cap_usd": 5.0, "confirm": "RUN-JV19-STAGE-1A"}, "1b": {"cap_usd": 20.0, "confirm": "RUN-JV19-STAGE-1B"}}


class Fake:
    """Deterministic provider double: every third call fails one criterion, so scores vary."""

    def __init__(self, served="o4-mini-2025-04-16", quota_at=None, every=3):
        self.calls, self.served, self.quota_at, self.every = [], served, quota_at, every

    def __call__(self, model, effort, temperature, max_tokens, messages):
        self.calls.append(messages[0]["content"])
        n = len(self.calls)
        if self.quota_at == n:
            raise llm.QuotaExceededError("HTTP 429: insufficient_quota")
        ok = n % self.every != 0
        body = {"criteria": [{"id": c, "verdict": "yes" if ok or i else "no", "evidence": ""} for i, c in enumerate(CRITERIA)], "rationale": "synthetic"}
        return json.dumps(body), {"prompt_tokens": 900, "completion_tokens": 300}, self.served


def forbidden(*args):
    raise AssertionError("no provider call permitted here")


@pytest.fixture(scope="module")
def real_population():
    return jv.build_population()


@pytest.fixture(scope="module")
def real_freeze(tmp_path_factory):
    work = tmp_path_factory.mktemp("jv19real") / "w"
    body = jv.freeze(work)
    return work, body, jv._sha(work / "protocol.json")


def small_protocol(tmp_path, monkeypatch, real_population, rb=36, rt=100, caps=None, projection=500, fake_mc=None):
    pop = dict(real_population)
    by_hash = sorted(pop["rb_items"], key=lambda i: jv._h("t|" + i["case_id"]))
    pop["rb_items"] = by_hash[:rb]
    ids = {i["case_id"] for i in pop["rb_items"]}
    pop["cached"] = {k: v for k, v in pop["cached"].items() if k in ids}
    pop["mc"] = {k: v for k, v in pop["mc"].items() if k in ids}
    pop["rb_labels"] = {k: v for k, v in pop["rb_labels"].items() if k in ids}
    pop["rt_items"] = pop["rt_items"][:rt]
    rids = {i["case_id"] for i in pop["rt_items"]}
    pop["rt_labels"] = {k: v for k, v in pop["rt_labels"].items() if k in rids}
    if fake_mc is not None:
        for cid in ids:
            c = pop["rb_cases"][cid]
            full, sents = mcjob.minicheck_pair(fake_mc, c.context, c.candidate_output, splitter=lambda t: [t])
            pop["mc"][cid] = {"full": full, "sent": sents}
    monkeypatch.setattr(jv, "build_population", lambda: pop)
    caps = caps or SMALL_CAPS
    monkeypatch.setattr(jv, "STAGES", caps)
    monkeypatch.setattr(jv, "PHASE_CAP_USD", sum(c["cap_usd"] for c in caps.values()))
    monkeypatch.setattr(jv, "PROJECTION_AFTER_JOBS", projection)
    work = tmp_path / "w"
    jv.freeze(work)
    return work, jv._sha(work / "protocol.json"), pop


# ---------------------------------------------------------------------------------------------------------------------------- freeze
def test_the_real_freeze_has_the_exact_populations_call_counts_and_caps(real_freeze):
    _w, b, _s = real_freeze
    assert b["routebench"]["cases"] == 257 and b["routebench"]["train"] == 129 and b["routebench"]["dev"] == 128
    assert (b["ragtruth"]["sentences"], b["ragtruth"]["documents"], b["ragtruth"]["unsupported"]) == (1050, 160, 338)
    assert b["ragtruth"]["sample_sha256"] == "7edc80e3ead8e5f6bc308807078e6a3a7a40d9f0c6afcff9d3f7445452d66012" and len(b["ragtruth"]["sample_sha256"]) == 64
    assert b["stages"]["1a"]["expected_calls"] == 1028 and b["stages"]["1b"]["expected_calls"] == 5250 and b["total_expected_calls"] == 6278
    assert b["stages"]["1a"]["cap_usd"] == 10.0 and b["stages"]["1b"]["cap_usd"] == 46.0 and b["phase_cap_usd"] == 56.0
    assert b["stages"]["1a"]["estimated_usd"] == 8.22 and b["stages"]["1b"]["estimated_usd"] == 42.0 and b["total_estimated_usd"] == 50.22
    assert b["stages"]["1a"]["max_attempts_per_job"] == 3 and b["stages"]["1b"]["projection_stop"]["after_jobs"] == 500
    assert b["decision_rule"] == {"pass_iff_score_at_least": 1.0, "thresholds_searched": False}


def test_the_freeze_defines_a1_as_majority_voting_and_keeps_raw_scores_for_a2(real_freeze):
    _w, b, _s = real_freeze
    assert "majority vote" in b["arms"]["A1"] and "NOT a mean-score threshold" in b["arms"]["A1"] and "keeps raw scores" in b["arms"]["A2"]
    assert b["features"][:3] == ["v18_mean_raw_score", "v18_all_pass_fraction", "v18_raw_score_sd"]


def test_the_freeze_binds_code_inputs_and_evaluation_boundaries(real_freeze):
    _w, b, _s = real_freeze
    assert {"judge_v19_phase1.py", "judge_v19_analysis.py", "judge_v19_minicheck.py", "judge_improvement_sample.py", "srcint_eval_v3.py", "checkers.py", "judge.py"} <= set(b["code"])
    assert set(b["input_sha256"]) == set(jv.INPUT_FILES) and "labels_routebench.json" in b["input_sha256"]
    assert any("only by `screen`" in x for x in b["boundary"]) and "70 TofuEval upstream documents" in b["data_policy"]["untouched"]
    assert "evaluation_only" in b["data_policy"] and "never used to fit" in b["data_policy"]["evaluation_only"]


def test_fit_population_contains_no_tofueval_or_aggrefact_row_and_evaluation_population_is_the_original(real_population):
    rt = real_population["rt_cases"]
    assert all(c.source == "RAGTruth/CNN-DM" and c.case_id.startswith("ragtruth:") for c in rt.values())
    rb = real_population["rb_cases"]
    assert len(rb) == 257 and not set(rb) & set(rt)
    assert {i["pop"] for i in real_population["rb_items"]} == {"train", "dev"}  # never the sealed test


def test_a_second_freeze_into_the_same_directory_is_refused(real_freeze):
    with pytest.raises(jv.Phase1Error, match="never overwritten"):
        jv.freeze(real_freeze[0])


def test_verify_refuses_changed_inputs_code_or_protocol(real_freeze, tmp_path):
    work, _b, sha = real_freeze
    dst = tmp_path / "w"
    dst.mkdir()
    for p in work.iterdir():
        if p.is_file():
            (dst / p.name).write_bytes(p.read_bytes())
    jv.verify(dst, sha)
    with pytest.raises(jv.Phase1Error, match="protocol hash"):
        jv.verify(dst, "0" * 64)
    (dst / "labels_ragtruth.json").write_text("{}", encoding="utf8")
    with pytest.raises(jv.Phase1Error, match="changed after the freeze"):
        jv.verify(dst, sha)


def test_replicates_send_byte_identical_prompts_with_distinct_job_ids(real_freeze, real_population):
    _w, _b, _s = real_freeze
    jobs = jv.jobs_for("1b", real_population["rb_items"], real_population["rt_items"])[:5]
    cases = jv.job_cases(jobs, real_population["rt_cases"])
    cfg = mv.cfg_of(jv.VARIANT)
    rub = mv.rubrics.for_task("summarize", cfg.rubric_dir)
    prompts = {mv.judge.build_prompt(c, rub, cfg) for c in cases.values()}
    assert len(prompts) == 1 and len({j["job_id"] for j in jobs}) == 5 and [j["replicate"] for j in jobs] == [1, 2, 3, 4, 5]
    a = jv.jobs_for("1a", real_population["rb_items"], real_population["rt_items"])
    assert len(a) == 1028 and {j["replicate"] for j in a} == {2, 3, 4, 5} and len({j["job_id"] for j in a}) == 1028


# ---------------------------------------------------------------------------------------------------------------------------- staged runner
def test_stage_1a_runs_to_completion_and_stage_1b_is_not_automatic(tmp_path, monkeypatch, real_population):
    work, sha, _pop = small_protocol(tmp_path, monkeypatch, real_population)
    f = Fake()
    t = jv.run_stage(work, "1a", sha, f)
    assert t["status"] == "COMPLETE" and t["jobs"] == 36 * 4 == len(f.calls) and t["committed_usd"] > 0
    with pytest.raises(jv.Phase1Error, match="explicit release-1b"):
        jv.run_stage(work, "1b", sha, forbidden)
    assert not (work / "stage_1b" / "results.jsonl").exists()


def test_release_requires_a_complete_stage_1a_and_the_exact_confirmation(tmp_path, monkeypatch, real_population):
    work, sha, _pop = small_protocol(tmp_path, monkeypatch, real_population)
    with pytest.raises(jv.Phase1Error, match="confirm"):
        jv.release_1b(work, sha, "yes")
    with pytest.raises(jv.Phase1Error, match="COMPLETE"):
        jv.release_1b(work, sha, jv.RELEASE_CONFIRM)
    jv.run_stage(work, "1a", sha, Fake())
    body = jv.release_1b(work, sha, jv.RELEASE_CONFIRM)
    assert body["protocol_sha256"] == sha
    with pytest.raises(jv.Phase1Error, match="exists"):
        jv.release_1b(work, sha, jv.RELEASE_CONFIRM)


def test_stage_1b_runs_after_release_and_scores_have_five_replicates(tmp_path, monkeypatch, real_population):
    work, sha, _pop = small_protocol(tmp_path, monkeypatch, real_population)
    jv.run_stage(work, "1a", sha, Fake())
    jv.release_1b(work, sha, jv.RELEASE_CONFIRM)
    f = Fake()
    t = jv.run_stage(work, "1b", sha, f)
    assert t["status"] == "COMPLETE" and t["jobs"] == 100 * 5 == len(f.calls)
    s1b, s1a = jv.stage_scores(work, "1b", sha), jv.stage_scores(work, "1a", sha)
    assert all(len(v) == 5 for v in s1b.values()) and all(len(v) == 4 for v in s1a.values()) and len(s1b) == 100 and len(s1a) == 36


def test_a_completed_stage_is_final_and_an_aborted_one_needs_an_explicit_resume(tmp_path, monkeypatch, real_population):
    work, sha, _pop = small_protocol(tmp_path, monkeypatch, real_population)
    t = jv.run_stage(work, "1a", sha, Fake(quota_at=5))
    assert t["status"] == "ABORTED" and t["ok"] == 4 and "quota" in t["stopped_on"]
    with pytest.raises(jv.Phase1Error, match="--resume"):
        jv.run_stage(work, "1a", sha, forbidden)
    f = Fake()
    t2 = jv.run_stage(work, "1a", sha, f, resume=True)  # the quota-failed attempt was settled at its bound by the adapter, so the journal is clean
    assert t2["committed_usd"] > t["committed_usd"]
    assert t2["status"] == "COMPLETE" and len(f.calls) == 36 * 4 - 4
    with pytest.raises(jv.Phase1Error, match="final"):
        jv.run_stage(work, "1a", sha, forbidden, resume=True)


def test_the_first_identity_failure_stops_the_stage_and_keeps_the_charge(tmp_path, monkeypatch, real_population):
    work, sha, _pop = small_protocol(tmp_path, monkeypatch, real_population)
    f = Fake(served="")
    t = jv.run_stage(work, "1a", sha, f)
    assert t["status"] == "ABORTED" and len(f.calls) == 1 and "identity" in t["stopped_on"] and t["committed_usd"] > 0


def test_stage_1b_stops_on_the_projection_rule_after_the_declared_number_of_jobs(tmp_path, monkeypatch, real_population):
    caps = {"1a": {"cap_usd": 5.0, "confirm": "RUN-JV19-STAGE-1A"}, "1b": {"cap_usd": 0.08, "confirm": "RUN-JV19-STAGE-1B"}}
    work, sha, _pop = small_protocol(tmp_path, monkeypatch, real_population, rt=20, caps=caps, projection=5)
    jv.run_stage(work, "1a", sha, Fake())
    jv.release_1b(work, sha, jv.RELEASE_CONFIRM)
    f = Fake()
    t = jv.run_stage(work, "1b", sha, f)
    assert t["status"] == "ABORTED" and "projection stop" in t["stopped_on"] and len(f.calls) == 5 and t["committed_usd"] < 0.08


def test_stage_caps_sum_to_the_phase_cap_and_the_estimate_is_below_each_cap():
    assert sum(s["cap_usd"] for s in jv.STAGES.values()) == jv.PHASE_CAP_USD == 56.0
    assert 1028 * jv.EST_USD_PER_CALL < jv.STAGES["1a"]["cap_usd"] and 5250 * jv.EST_USD_PER_CALL < jv.STAGES["1b"]["cap_usd"]


# ---------------------------------------------------------------------------------------------------------------------------- features, fit, screen
class FakeChecker:
    def __init__(self, offset=0.0):
        self.offset = offset

    def score(self, docs, claims):
        return [min(0.98, max(0.01, int(hashlib.sha256((d[:40] + c).encode()).hexdigest()[:6], 16) % 1000 / 1000 + self.offset)) for d, c in zip(docs, claims, strict=True)]


def test_minicheck_job_requires_equivalence_before_scoring_ragtruth(tmp_path, monkeypatch, real_population):
    work, sha, _pop = small_protocol(tmp_path, monkeypatch, real_population, fake_mc=FakeChecker())
    with pytest.raises(mcjob.MiniCheckJobError, match="confirm"):
        mcjob.run(work, sha, "no", FakeChecker(), lambda t: [t])
    with pytest.raises(mcjob.MiniCheckJobError, match="equivalence check failed"):
        mcjob.run(work, sha, mcjob.MINICHECK_CONFIRM, FakeChecker(offset=0.2), lambda t: [t])
    assert not (work / mcjob.OUTPUT).exists()  # RAGTruth was never scored
    out = mcjob.run(work, sha, mcjob.MINICHECK_CONFIRM, FakeChecker(), lambda t: [t])
    assert out["rows"] == 100 and out["equivalence_worst_abs_difference"] == 0.0
    with pytest.raises(mcjob.MiniCheckJobError, match="exists"):
        mcjob.run(work, sha, mcjob.MINICHECK_CONFIRM, FakeChecker(), lambda t: [t])


def test_minicheck_job_refuses_a_tampered_protocol_input(tmp_path, monkeypatch, real_population):
    work, sha, _pop = small_protocol(tmp_path, monkeypatch, real_population, fake_mc=FakeChecker())
    (work / "minicheck_cached_routebench.json").write_text("{}", encoding="utf8")
    with pytest.raises(mcjob.MiniCheckJobError, match="changed after the freeze"):
        mcjob.run(work, sha, mcjob.MINICHECK_CONFIRM, FakeChecker(), lambda t: [t])


def _complete_through_features(tmp_path, monkeypatch, real_population):
    work, sha, pop = small_protocol(tmp_path, monkeypatch, real_population, fake_mc=FakeChecker())
    jv.run_stage(work, "1a", sha, Fake())
    jv.release_1b(work, sha, jv.RELEASE_CONFIRM)
    jv.run_stage(work, "1b", sha, Fake(every=2))
    mcjob.run(work, sha, mcjob.MINICHECK_CONFIRM, FakeChecker(), lambda t: [t])
    return work, sha, pop


def test_fit_never_opens_routebench_labels_and_screen_checks_the_model_before_opening_them(tmp_path, monkeypatch, real_population):
    work, sha, _pop = _complete_through_features(tmp_path, monkeypatch, real_population)
    real = Path.read_text
    opened = []

    def spy(path, *a, **k):
        if path.name == "labels_routebench.json":
            opened.append(path.name)
        return real(path, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy)
    res = jv.fit(work, sha)
    assert opened == [] and (work / "model.json").exists() and len(res["model_sha256"]) == 64
    with pytest.raises(jv.Phase1Error, match="model hash"):
        jv.screen(work, sha, "0" * 64)
    assert opened == []
    rep = jv.screen(work, sha, res["model_sha256"])
    assert opened == ["labels_routebench.json"]
    assert rep["status"] == "COMPLETE" and set(rep["gate_A2_vs_A0"]) >= {"s1_pooled_kappa_gain", "s2_each_subset_positive", "s3_error_classes", "s4_flip_precision", "pass"}
    assert set(rep["kappa"]["v18_single_runs"]) == {f"run{i}" for i in range(1, 6)} and "A1_majority" in rep["kappa"]
    assert "never a validation" in rep["label"] and "A2_minus_A0" in rep["kappa_difference_95_document_bootstrap_reported_not_gating"]
    with pytest.raises(jv.Phase1Error, match="run once"):
        jv.screen(work, sha, res["model_sha256"])
    with pytest.raises(jv.Phase1Error, match="fitted once"):
        jv.fit(work, sha)


def test_fit_refuses_features_not_produced_by_the_frozen_job_or_without_equivalence(tmp_path, monkeypatch, real_population):
    work, sha, _pop = _complete_through_features(tmp_path, monkeypatch, real_population)
    path = work / mcjob.OUTPUT
    good = json.loads(path.read_text(encoding="utf8"))
    path.write_text(json.dumps({**good, "producer_sha256": "0" * 64}), encoding="utf8")
    with pytest.raises(jv.Phase1Error, match="frozen job"):
        jv.fit(work, sha)
    path.write_text(json.dumps({**good, "equivalence_worst_abs_difference": 0.5}), encoding="utf8")
    with pytest.raises(jv.Phase1Error, match="equivalence"):
        jv.fit(work, sha)


def test_screen_needs_a_complete_stage_1a(tmp_path, monkeypatch, real_population):
    work, sha, _pop = small_protocol(tmp_path, monkeypatch, real_population)
    jv.run_stage(work, "1a", sha, Fake(quota_at=3))
    (work / "model.json").write_text(json.dumps({"model_sha256": "x"}), encoding="utf8")
    with pytest.raises(jv.Phase1Error):
        jv.screen(work, sha, "x")


# ---------------------------------------------------------------------------------------------------------------------------- CLI and preservation
def test_cli_refuses_without_hash_confirmation_or_stage(real_freeze, capsys):
    work, _b, sha = real_freeze
    assert jv.main(["run", "--work", str(work), "--stage", "1a"]) == 2
    assert jv.main(["run", "--work", str(work), "--protocol-sha256", sha, "--stage", "1a", "--confirm", "wrong"]) == 2
    assert jv.main(["run", "--work", str(work), "--protocol-sha256", sha, "--confirm", "RUN-JV19-STAGE-1A"]) == 2
    assert jv.main(["screen", "--work", str(work), "--protocol-sha256", sha]) == 2


def test_prior_frozen_artifacts_are_untouched_by_this_protocol():
    v3 = jv.ROOT / ".local" / "srcint-eval-v3" / "protocol.json"
    if v3.exists():
        assert jv._sha(v3) == "5cbfa028be3c2cdd648d99f433af1d5e76dd998843a483381c9c3bfa17747fc9"
    assert jv.WORK != jv.ROOT / ".local" / "srcint-eval-v3" and an.PASS_THRESHOLD == 1.0

"""Tests for the frozen claim-fidelity-eval-v1 protocol (v24 vs five stored v18 runs). Mocked provider only: no provider call, no GPU, no download."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import httpx
import pytest
from aisys import llm
from evalops import claim_fidelity as cf
from evalops import claim_fidelity_eval as ce
from evalops import judge_v19_phase1 as jv

COPY = ("protocol.json", *ce.INPUT_FILES)
SECRET = "sk-" + "synthetic-test-key-" + "x" * 30


class Fake:
    """Scripted provider: call n fails claim_fidelity only when n%9==0, claims_entailed when n%13==0 (both at multiples of 117), otherwise all yes."""

    def __init__(self, served="o4-mini-2025-04-16", raises=None, bad_after=None, drop_fidelity=False):
        self.calls, self.served, self.raises, self.bad_after, self.drop = 0, served, raises or {}, bad_after, drop_fidelity
        self.prompts: list[str] = []

    def __call__(self, model, effort, temperature, max_tokens, messages):
        self.calls += 1
        n = self.calls
        self.prompts.append(messages[0]["content"])
        if n in self.raises:
            raise self.raises[n]
        verdict = {c: "yes" for c in ce.CRITERIA}
        if n % 9 == 0:
            verdict["claim_fidelity"] = "no"
        if n % 13 == 0:
            verdict["claims_entailed"] = "no"
        entries = [{"id": c, "verdict": v, "evidence": ""} for c, v in verdict.items() if not (self.drop and c == "claim_fidelity")]
        return json.dumps({"criteria": entries, "critical_errors": [], "rationale": "scripted"}), {"prompt_tokens": 900, "completion_tokens": 300}, self.served


def forbidden(*args):
    raise AssertionError("no provider call permitted here")


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setattr(llm.settings, "openai_api_key", SECRET)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    work = tmp_path_factory.mktemp("cfe") / "w"
    body = ce.freeze(work)
    return work, body, ce._sha(work / "protocol.json")


def copy_work(frozen, tmp_path, cap=None):
    work, _b, sha = frozen
    dst = tmp_path / "w"
    dst.mkdir()
    for n in COPY:
        shutil.copy(work / n, dst / n)
    if cap is not None:
        p = json.loads((dst / "protocol.json").read_text(encoding="utf8"))
        p["cap_usd"] = cap
        (dst / "protocol.json").write_text(json.dumps(p, indent=2, sort_keys=True), encoding="utf8")
        sha = ce._sha(dst / "protocol.json")
    return dst, sha


@pytest.fixture(scope="module")
def completed(frozen, tmp_path_factory):
    work, _b, sha = frozen
    dst = tmp_path_factory.mktemp("cfe_done") / "w"
    dst.mkdir()
    for n in COPY:
        shutil.copy(work / n, dst / n)
    f = Fake()
    t = ce.run(dst, sha, f)
    return dst, sha, t, f


# ---------------------------------------------------------------------------------------------------------------------------- freeze
def test_freeze_binds_population_variant_baseline_cap_and_code(frozen):
    _w, b, _s = frozen
    assert b["population"]["cases"] == 257 and b["population"]["train"] == 129 and b["population"]["dev"] == 128 and b["population"]["original_inputs"] and b["population"]["removed_cases"] == 0
    assert b["cap_usd"] == 3.0 and b["estimate"]["calls"] == 257 and b["estimate"]["expected_usd"] == pytest.approx(2.11) and b["estimate"]["expected_usd"] < b["cap_usd"]
    j = b["judge"]
    assert j["variant"] == cf.VARIANT_ID and j["criteria"] == list(ce.CRITERIA) and j["config_hash"] == cf.config().config_hash and j["rubric_hash"] == cf.rubric().hash
    assert j["v18_identity"] == cf.V18_IDENTITY and j["identity"]["v18_config_hash"] == "12fc8b550d954359"
    assert b["decision_rule"] == {"pass_iff_score_at_least": 1.0, "thresholds_searched": False, "rubric_edits_after_freeze": False}
    assert b["baseline"]["judge_v19_protocol_sha256"] == ce.JV19_PROTOCOL_SHA256 and set(b["baseline"]["artifact_sha256"]) == set(ce.BASELINE_ARTIFACTS)
    assert {"claim_fidelity_eval.py", "claim_fidelity.py", "rubrics_claim_fidelity_v1/summarize.yaml", "judge_v19_recovery.py", "judge.py", "srcint_eval_v3.py"} <= set(b["code"])
    assert "status" in b and "not run" in b["status"] and any("no automatic follow-up run" in x for x in b["boundary"]) and any("no held-out access" in x for x in b["boundary"])


def test_the_baseline_holds_five_runs_per_case_and_run_one_is_the_cached_judgment(frozen):
    work, _b, _s = frozen
    five = json.loads((work / "baseline_v18_five_runs.json").read_text(encoding="utf8"))
    cached = json.loads((jv.WORK / "cached_v18.json").read_text(encoding="utf8"))
    assert len(five) == 257 and all(len(v) == 5 for v in five.values()) and all(v[0] == cached[c]["score"] for c, v in five.items())


def test_the_population_is_the_original_unrepaired_inputs_and_never_the_sealed_test(frozen):
    work, _b, _s = frozen
    items = json.loads((work / "population.json").read_text(encoding="utf8"))
    assert {i["pop"] for i in items} == {"train", "dev"} and len({i["case_id"] for i in items}) == 257 and all(i["case_id"].startswith("gnd-") and not i["case_id"].startswith("source-integrity") for i in items)


def test_verify_refuses_changed_inputs_protocol_code_or_baseline(frozen, tmp_path, monkeypatch):
    work, sha = copy_work(frozen, tmp_path)
    ce.verify(work, sha)
    with pytest.raises(ce.ClaimFidelityEvalError, match="protocol hash"):
        ce.verify(work, "0" * 64)
    original = (work / "labels.json").read_bytes()
    (work / "labels.json").write_text("{}", encoding="utf8")
    with pytest.raises(ce.ClaimFidelityEvalError, match="changed after the freeze"):
        ce.verify(work, sha)
    (work / "labels.json").write_bytes(original)
    monkeypatch.setattr(ce, "code_hashes", lambda: {"tampered": "x"})
    with pytest.raises(ce.ClaimFidelityEvalError, match="code, dependency"):
        ce.verify(work, sha)
    monkeypatch.undo()
    bad = tmp_path / "bad_baseline.json"
    bad.write_text("{}", encoding="utf8")
    real = ce._abs_baseline
    monkeypatch.setattr(ce, "_abs_baseline", lambda rel: bad if rel == "cached_v18.json" else real(rel))
    monkeypatch.setattr(llm.settings, "openai_api_key", SECRET)
    with pytest.raises(ce.ClaimFidelityEvalError, match="baseline artifact"):
        ce.verify(work, sha)


def test_earlier_frozen_artifacts_are_untouched():
    v3 = ce.ROOT / ".local" / "srcint-eval-v3" / "protocol.json"
    if v3.exists():
        assert ce._sha(v3) == "5cbfa028be3c2cdd648d99f433af1d5e76dd998843a483381c9c3bfa17747fc9"
    p = jv.WORK / "protocol.json"
    if p.exists():
        assert ce._sha(p) == ce.JV19_PROTOCOL_SHA256 and jv.verify(jv.WORK, ce.JV19_PROTOCOL_SHA256)
    assert ce.WORK != jv.WORK and ce.WORK != ce.ROOT / ".local" / "srcint-eval-v3"


# ---------------------------------------------------------------------------------------------------------------------------- key preflight
def test_preflight_reports_key_facts_without_exposing_the_key_or_calling_the_network(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    out = ce.preflight(work, sha)
    assert out["key_loaded"] and out["key_format_ok"] and out["network_call_made"] is False and out["ready_to_run"] and out["cases"] == 257 and out["cap_usd"] == 3.0
    assert SECRET not in json.dumps(out) and out["key_length"] == len(SECRET)


@pytest.mark.parametrize("bad", ["", "not-a-key", "sk-has a space" + "x" * 30, 'sk-"quoted"' + "x" * 30])
def test_a_missing_or_malformed_key_refuses_the_run_with_no_reservation_row_lock_or_charge(frozen, tmp_path, monkeypatch, bad):
    work, sha = copy_work(frozen, tmp_path)
    monkeypatch.setattr(llm.settings, "openai_api_key", bad)
    assert ce.preflight(work, sha)["ready_to_run"] is False
    with pytest.raises(ce.PreflightError):
        ce.run(work, sha, forbidden)
    assert not (work / "results.jsonl").exists() and not (work / "run.lock").exists() and not (work / "terminal.json").exists()
    assert not (work / "spend_journal.jsonl").exists() or (work / "spend_journal.jsonl").read_text(encoding="utf8") == ""


# ---------------------------------------------------------------------------------------------------------------------------- the run
def test_a_complete_run_judges_every_case_once_with_the_v24_prompt_on_original_inputs(completed, frozen):
    work, sha, t, f = completed
    assert t["status"] == "COMPLETE" and t["ok"] == 257 and t["failed"] == [] and f.calls == 257 and t["committed_usd"] == pytest.approx(257 * (900 * 1.1 + 300 * 4.4) / 1e6, abs=1e-4) and t["committed_usd"] < 3.0
    assert all("- id: claim_fidelity" in p for p in f.prompts)
    items = json.loads((work / "population.json").read_text(encoding="utf8"))
    first = json.loads((work / "cases.jsonl").read_text(encoding="utf8").splitlines()[0])
    assert first["case_id"] == items[0]["case_id"] and first["context"] in f.prompts[0]  # the original, unrepaired source text was sent
    with pytest.raises(ce.ClaimFidelityEvalError, match="final"):
        ce.run(work, sha, forbidden, resume=True)


def test_the_cap_is_cumulative_and_a_run_that_reaches_it_aborts_without_completing(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path, cap=0.1)
    t = ce.run(work, sha, Fake())
    assert t["status"] == "ABORTED" and "cap $0.10" in t["stopped_on"] and 0 < t["ok"] < 257 and t["committed_usd"] <= 0.1
    again = ce.run(work, sha, Fake(), resume=True)
    assert again["status"] == "ABORTED" and again["ok"] == t["ok"] and again["committed_usd"] <= 0.1  # no further spend past the cap


def test_a_request_that_was_not_sent_writes_no_row_charges_the_bound_and_leaves_the_job_pending(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    f = Fake(raises={3: httpx.LocalProtocolError("Illegal header value b'Bearer '")})
    t = ce.run(work, sha, f)
    assert t["status"] == "ABORTED" and "could not be sent" in t["stopped_on"] and t["ok"] == 2 and t["failed"] == [] and t["not_attempted"] == 255 and f.calls == 3
    rows = [json.loads(x) for x in (work / "results.jsonl").read_text(encoding="utf8").splitlines()]
    assert len(rows) == 2 and all(r["record"]["status"] == "ok" for r in rows)  # no failed row exists for the unsent job
    assert t["committed_usd"] > 2 * 0.0023 + 0.04  # the third reservation was charged at its bound
    with pytest.raises(ce.ClaimFidelityEvalError, match="--resume"):
        ce.run(work, sha, forbidden)
    f2 = Fake()
    t2 = ce.run(work, sha, f2, resume=True)  # the pending job is judged now: nothing was skipped
    assert t2["status"] == "COMPLETE" and t2["ok"] == 257 and f2.calls == 255


def test_an_interrupted_reservation_must_be_settled_explicitly_and_its_job_is_redone(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    f = Fake(raises={5: KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        ce.run(work, sha, f)
    assert not (work / "run.lock").exists()
    state = ce.sv3.journal_state(work / "spend_journal.jsonl", 3.0)
    assert len(state["unsettled"]) == 1
    with pytest.raises(ce.ClaimFidelityEvalError, match="settle-unsettled"):
        ce.run(work, sha, forbidden, resume=True)
    f2 = Fake()
    t = ce.run(work, sha, f2, resume=True, settle_unsettled=True)
    assert t["status"] == "COMPLETE" and f2.calls == 257 - 4 and t["unsettled"] == [] and t["committed_usd"] > 257 * 0.0023 + 0.04  # the interrupted reservation stays charged at its bound


def test_a_judgment_without_claim_fidelity_is_never_a_pass_and_withholds_kappa_without_opening_labels(frozen, tmp_path, monkeypatch):
    work, sha = copy_work(frozen, tmp_path)
    f = Fake(drop_fidelity=True)
    t = ce.run(work, sha, f)
    assert t["status"] == "ABORTED" and len(t["failed"]) == 1 and f.calls == cf.config().max_attempts and "claim_fidelity" in t["stopped_on"]
    real = Path.read_text

    def guard(path, *a, **k):
        if path.name in ("labels.json", "baseline_v18_five_runs.json"):
            raise AssertionError("labels or baseline were read before completeness")
        return real(path, *a, **k)

    monkeypatch.setattr(Path, "read_text", guard)
    out = ce.compare(work, sha)
    assert out["status"] == "WITHHELD_INCOMPLETE" and "no kappa exists" in out["note"] and not (work / "comparison.json").exists()


def test_a_missing_served_model_identity_stops_the_run_and_keeps_the_charge(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    f = Fake(served="")
    t = ce.run(work, sha, f)
    assert t["status"] == "ABORTED" and f.calls == 1 and "identity" in t["stopped_on"] and t["committed_usd"] > 0


def test_a_tampered_stored_row_is_refused_before_any_label_is_read(completed, tmp_path):
    work, sha, _t, _f = completed
    dst = tmp_path / "w"
    shutil.copytree(work, dst)
    lines = (dst / "results.jsonl").read_text(encoding="utf8").splitlines()
    row = json.loads(lines[0])
    row["record"]["score"] = 0.123
    (dst / "results.jsonl").write_text("\n".join([json.dumps(row), *lines[1:]]) + "\n", encoding="utf8")
    with pytest.raises(ce.ClaimFidelityEvalError, match="differ from the stored raw response"):
        ce.compare(dst, sha)


# ---------------------------------------------------------------------------------------------------------------------------- the comparison
def test_the_comparison_reports_dev_and_train_separately_against_all_five_v18_runs(completed, tmp_path):
    work, sha, _t, _f = completed
    dst = tmp_path / "w"
    shutil.copytree(work, dst)
    r = ce.compare(dst, sha)
    assert r["status"] == "COMPLETE" and r["coverage"] == {"cases": 257, "valid_judgments": 257, "removed_cases": 0} and "not a pass of the 0.74 target" in r["label"]
    assert r["dev"]["cases"] == 128 and r["train"]["cases"] == 129 and r["pooled_supplementary"]["cases"] == 257
    for pop in ("dev", "train", "pooled_supplementary"):
        b = r[pop]
        assert set(b["v18_runs"]) == {f"run{i}" for i in range(1, 6)} and b["v24"]["n"] == b["cases"] and "kappa_95_document_bootstrap" in b["v24"]
        v24_errors = b["v24"]["false_passes"] + b["v24"]["false_rejections"]
        for name, m in b["v18_runs"].items():
            run_errors = m["false_passes"] + m["false_rejections"]
            assert run_errors - v24_errors == m["errors_corrected_by_v24"] - m["new_errors_introduced_by_v24"], (pop, name)
            assert "v24_minus_run_kappa_95_document_bootstrap" in m and m["verdicts_changed"] >= abs(m["errors_corrected_by_v24"] - m["new_errors_introduced_by_v24"])
        assert b["v18_five_run_summary"]["kappa_min"] <= b["v18_five_run_summary"]["kappa_max"]
    assert r["dev"]["v18_runs"]["run1"]["v24_minus_run_kappa_95_document_bootstrap"] is not None
    with pytest.raises(ce.ClaimFidelityEvalError, match="written once"):
        ce.compare(dst, sha)


def test_fidelity_only_failures_cost_and_per_criterion_counts_match_the_scripted_run(completed, tmp_path):
    work, sha, t, f = completed
    dst = tmp_path / "w"
    shutil.copytree(work, dst)
    r = ce.compare(dst, sha)
    crit = r["pooled_supplementary"]["v24_criteria"]
    assert crit["failures_by_criterion"]["claim_fidelity"] == 257 // 9 == 28 and crit["failures_by_criterion"]["claims_entailed"] == 257 // 13 == 19
    assert crit["fidelity_only_failures"] == 28 - 2 and crit["fidelity_only_failures"] == r["dev"]["v24_criteria"]["fidelity_only_failures"] + r["train"]["v24_criteria"]["fidelity_only_failures"]
    assert crit["fidelity_only_failures_published_supported"] + crit["fidelity_only_failures_published_unsupported"] == crit["fidelity_only_failures"]
    assert r["cost"]["committed_usd"] == t["committed_usd"] and r["cost"]["cap_usd"] == 3.0 and r["cost"]["judgments_with_usage"] == 257 and r["cost"]["retried_cases"] == 0
    assert r["cost"]["actual_usage_usd"] == pytest.approx(257 * (900 * 1.1 + 300 * 4.4) / 1e6, rel=1e-3)
    assert r["pooled_supplementary"]["v24"]["false_passes"] + r["pooled_supplementary"]["v24"]["false_rejections"] >= 0 and f.calls == 257


# ---------------------------------------------------------------------------------------------------------------------------- CLI
def test_cli_refuses_without_hash_or_confirmation_and_preflight_prints_no_secret(frozen, tmp_path, capsys):
    work, sha = copy_work(frozen, tmp_path)
    assert ce.main(["run", "--work", str(work)]) == 2
    assert ce.main(["run", "--work", str(work), "--protocol-sha256", sha]) == 2
    assert ce.main(["run", "--work", str(work), "--protocol-sha256", sha, "--confirm", "wrong"]) == 2
    capsys.readouterr()
    assert ce.main(["preflight", "--work", str(work), "--protocol-sha256", sha]) == 0
    out = capsys.readouterr().out
    assert SECRET not in out and '"key_loaded": true' in out and '"network_call_made": false' in out
    assert hashlib.sha256(b"x").hexdigest()  # keep hashlib imported for clarity of intent

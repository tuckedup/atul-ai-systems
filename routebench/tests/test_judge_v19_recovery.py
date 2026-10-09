"""Tests for the audited recovery of one Stage 1a job that failed before any request was sent, and for the report helper on a recovered stage. Mocked provider only."""
from __future__ import annotations

import json

import httpx
import pytest
from aisys import llm
from evalops import judge_v19_phase1 as jv
from evalops import judge_v19_recovery as rec
from evalops import judge_v19_stage1a_report as rp
from evalops import rubrics

CRITERIA = sorted(rubrics.for_task("summarize").criterion_ids)
CAPS = {"1a": {"cap_usd": 5.0, "confirm": "RUN-JV19-STAGE-1A"}, "1b": {"cap_usd": 20.0, "confirm": "RUN-JV19-STAGE-1B"}}


class Fake:
    def __init__(self, fail_first=None, every=3):
        self.calls, self.fail_first, self.every = 0, fail_first, every

    def __call__(self, model, effort, temperature, max_tokens, messages):
        self.calls += 1
        if self.fail_first is not None and self.calls <= self.fail_first:
            raise httpx.LocalProtocolError("Illegal header value b'Bearer '")
        ok = self.calls % self.every != 0
        body = {"criteria": [{"id": c, "verdict": "yes" if ok or i else "no", "evidence": ""} for i, c in enumerate(CRITERIA)], "rationale": "synthetic"}
        return json.dumps(body), {"prompt_tokens": 900, "completion_tokens": 300}, "o4-mini-2025-04-16"


def forbidden(*args):
    raise AssertionError("no provider call permitted here")


@pytest.fixture(scope="module")
def real_population():
    return jv.build_population()


def protocol(tmp_path, monkeypatch, real_population, rb=36):
    pop = dict(real_population)
    pop["rb_items"] = sorted(pop["rb_items"], key=lambda i: jv._h("t|" + i["case_id"]))[:rb]
    ids = {i["case_id"] for i in pop["rb_items"]}
    for k in ("cached", "mc", "rb_labels"):
        pop[k] = {c: v for c, v in pop[k].items() if c in ids}
    pop["rt_items"] = pop["rt_items"][:20]
    pop["rt_labels"] = {i["case_id"]: pop["rt_labels"][i["case_id"]] for i in pop["rt_items"]}
    monkeypatch.setattr(jv, "build_population", lambda: pop)
    monkeypatch.setattr(jv, "STAGES", CAPS)
    monkeypatch.setattr(jv, "PHASE_CAP_USD", 25.0)
    work = tmp_path / "w"
    jv.freeze(work)
    return work, jv._sha(work / "protocol.json")


def failed_stage(tmp_path, monkeypatch, real_population):
    work, sha = protocol(tmp_path, monkeypatch, real_population)
    t = jv.run_stage(work, "1a", sha, Fake(fail_first=3))  # the first job fails all three attempts locally
    assert t["status"] == "ABORTED" and len(t["failed"]) == 1
    return work, sha, t


def test_recovery_reruns_only_the_failed_job_and_preserves_everything_else(tmp_path, monkeypatch, real_population):
    work, sha, t = failed_stage(tmp_path, monkeypatch, real_population)
    sd = work / "stage_1a"
    protocol_before, results_before, journal_before = (work / "protocol.json").read_bytes(), (sd / "results.jsonl").read_bytes(), (sd / "spend_journal.jsonl").read_bytes()
    with pytest.raises(rec.RecoveryError, match="confirm"):
        rec.recover(work, sha, "yes", forbidden)
    f = Fake()
    out = rec.recover(work, sha, rec.RECOVERY_CONFIRM, f)
    assert out["status"] == "ok" and f.calls == 1 and out["job"] == t["failed"][0] and out["committed_usd_after"] > t["committed_usd"] and out["committed_usd_after"] <= 5.0
    assert (work / "protocol.json").read_bytes() == protocol_before and (sd / "results.jsonl").read_bytes() == results_before  # nothing frozen or recorded was edited
    assert (sd / "spend_journal.jsonl").read_bytes().startswith(journal_before)  # append-only
    note = json.loads((work / rec.NOTE).read_text(encoding="utf8"))
    assert note["job"] == out["job"] and note["journal_prefix_bytes"] == len(journal_before) and note["protocol_sha256"] == sha
    with pytest.raises(rec.RecoveryError, match="already been recorded"):
        rec.recover(work, sha, rec.RECOVERY_CONFIRM, forbidden)


def test_after_recovery_the_frozen_runner_resumes_the_rest_and_ends_incomplete_with_the_one_failed_row_and_scores_assemble(tmp_path, monkeypatch, real_population):
    work, sha, t = failed_stage(tmp_path, monkeypatch, real_population)
    rec.recover(work, sha, rec.RECOVERY_CONFIRM, Fake())
    f = Fake()
    t2 = jv.run_stage(work, "1a", sha, f, resume=True)
    assert f.calls == 36 * 4 - 1 and t2["status"] == "INCOMPLETE" and t2["failed"] == t["failed"] and t2["not_attempted"] == 0
    scores, info = rec.assemble_scores(work, sha)
    assert len(scores) == 36 and all(len(v) == 4 for v in scores.values()) and info["recovered_job"] == t["failed"][0] and info["committed_usd"] <= 5.0
    with pytest.raises(jv.Phase1Error):
        jv.stage_scores(work, "1a", sha)  # the frozen accessor still (correctly) refuses an INCOMPLETE stage
    with pytest.raises(jv.Phase1Error):
        jv.release_1b(work, sha, jv.RELEASE_CONFIRM)  # and Stage 1b stays locked


def test_assemble_refuses_if_the_original_failed_row_or_the_journal_prefix_changed(tmp_path, monkeypatch, real_population):
    work, sha, _t = failed_stage(tmp_path, monkeypatch, real_population)
    rec.recover(work, sha, rec.RECOVERY_CONFIRM, Fake())
    jv.run_stage(work, "1a", sha, Fake(), resume=True)
    sd = work / "stage_1a"
    original = (sd / "results.jsonl").read_bytes()
    lines = original.splitlines()
    row = json.loads(lines[0])
    row["record"]["reason"] = row["record"]["reason"] + " "
    (sd / "results.jsonl").write_bytes(b"\n".join([json.dumps(row).encode(), *lines[1:]]) + b"\n")
    with pytest.raises(rec.RecoveryError, match="byte-identical"):
        rec.assemble_scores(work, sha)
    (sd / "results.jsonl").write_bytes(original)
    journal = (sd / "spend_journal.jsonl").read_bytes()
    (sd / "spend_journal.jsonl").write_bytes(journal.replace(b'"charged"', b'"charged "', 1))
    with pytest.raises((rec.RecoveryError, jv.sv3.SrcIntV3Error)):
        rec.assemble_scores(work, sha)


def test_a_genuine_failure_is_never_recovered_by_this_module(tmp_path, monkeypatch, real_population):
    work, sha = protocol(tmp_path, monkeypatch, real_population)

    def bad(*args):
        return "not json", {"prompt_tokens": 500, "completion_tokens": 50}, "o4-mini-2025-04-16"

    t = jv.run_stage(work, "1a", sha, bad)
    assert t["status"] == "ABORTED" and len(t["failed"]) == 1
    with pytest.raises(rec.RecoveryError, match="never retried"):
        rec.recover(work, sha, rec.RECOVERY_CONFIRM, forbidden)
    assert not (work / rec.NOTE).exists()


def test_recovery_refuses_a_complete_stage_and_an_over_cap_journal(tmp_path, monkeypatch, real_population):
    work, sha = protocol(tmp_path, monkeypatch, real_population)
    jv.run_stage(work, "1a", sha, Fake())
    with pytest.raises(rec.RecoveryError, match="ABORTED or INCOMPLETE"):
        rec.recover(work, sha, rec.RECOVERY_CONFIRM, forbidden)


def test_report_helper_uses_the_recovered_stage_and_reports_dev_and_train_separately(tmp_path, monkeypatch, real_population):
    work, sha, _t = failed_stage(tmp_path, monkeypatch, real_population)
    rec.recover(work, sha, rec.RECOVERY_CONFIRM, Fake())
    jv.run_stage(work, "1a", sha, Fake(), resume=True)
    out = rp.report(work, sha)
    assert out["coverage"]["stage_1a_jobs_valid"] == 144 and out["coverage"]["stage_1a_terminal_status"] == "INCOMPLETE" and out["recovery"]["recovered_job"]
    assert out["dev"]["cases"] + out["train"]["cases"] == 36 == out["pooled_supplementary"]["cases"] and out["spend"]["cap_usd"] == 5.0 and out["spend"]["unsettled"] == []
    assert set(out["dev"]["single_runs"]) == {f"run{i}" for i in range(1, 6)} and "A1_majority_vote" in out["train"]
    with pytest.raises(jv.Phase1Error, match="written once"):
        rp.report(work, sha)


def test_the_authorization_header_error_is_the_local_error_the_recovery_recognises():
    assert issubclass(httpx.LocalProtocolError, Exception) and rec.NO_REQUEST_REASON == "LocalProtocolError: Illegal header value"
    assert llm.QuotaExceededError is not None

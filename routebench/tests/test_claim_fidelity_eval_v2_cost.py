"""Regressions for the v2 cost-reporting correction: usage cost and attempt counts come from the SPEND JOURNAL (every attempt), unknown-cost bound charges are reported separately and never as zero, and the unrun v1 freeze
is preserved. Mocked provider only: no provider call, no GPU, no download."""
from __future__ import annotations

import json
import shutil

import httpx
import pytest
from aisys import llm
from evalops import claim_fidelity_eval as v1
from evalops import claim_fidelity_eval_v2 as ce

COPY = ("protocol.json", *ce.INPUT_FILES)
PER_CALL = (900 * 1.1 + 300 * 4.4) / 1e6  # the scripted usage, priced at the frozen o4-mini rates
SECRET = "sk-" + "synthetic-test-key-" + "x" * 30


def reply(drop=()):
    entries = [{"id": c, "verdict": "yes", "evidence": ""} for c in ce.CRITERIA if c not in drop]
    return json.dumps({"criteria": entries, "critical_errors": [], "rationale": "scripted"})


class Scripted:
    """call n -> behaviour: 'drop' = valid JSON missing claim_fidelity (billed parse retry), 'nousage' = valid reply with no usage, 'unsent' = local protocol error, 'kill' = KeyboardInterrupt."""

    def __init__(self, script=None):
        self.script, self.calls = script or {}, 0

    def __call__(self, model, effort, temperature, max_tokens, messages):
        self.calls += 1
        what = self.script.get(self.calls)
        if what == "unsent":
            raise httpx.LocalProtocolError("Illegal header value b'Bearer '")
        if what == "kill":
            raise KeyboardInterrupt
        usage = None if what == "nousage" else {"prompt_tokens": 900, "completion_tokens": 300}
        return reply(drop=("claim_fidelity",) if what == "drop" else ()), usage, "o4-mini-2025-04-16"


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setattr(llm.settings, "openai_api_key", SECRET)
    monkeypatch.setattr(ce, "BOOT_DRAWS", 25)  # keep the comparison cheap; the production constant is unchanged


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    work = tmp_path_factory.mktemp("cfe2cost") / "w"
    ce.freeze(work)
    return work, ce._sha(work / "protocol.json")


def fresh(frozen, tmp_path):
    work, sha = frozen
    dst = tmp_path / "w"
    dst.mkdir()
    for n in COPY:
        shutil.copy(work / n, dst / n)
    return dst, sha


def journal_known_usage(work):
    return sum(e["actual"] for e in (json.loads(x) for x in (work / "spend_journal.jsonl").read_text(encoding="utf8").splitlines()) if e["op"] == "settle" and e["actual"] is not None)


def test_the_reviewers_reproduction_a_billed_parse_retry_is_now_in_the_reported_usage(frozen, tmp_path):
    work, sha = fresh(frozen, tmp_path)
    f = Scripted({1: "drop"})
    t = ce.run(work, sha, f)
    assert t["status"] == "COMPLETE" and f.calls == 258
    c = ce.compare(work, sha)["cost"]
    assert c["known_usage_cost_usd"] == round(journal_known_usage(work), 4) == round(258 * PER_CALL, 4) and c["committed_usd"] == t["committed_usd"] == c["known_usage_cost_usd"]
    assert c["journal_attempts_reserved"] == c["journal_attempts_settled"] == c["attempts_with_known_usage"] == 258 and c["successful_final_judgments"] == 257
    assert c["retry_attempts_within_stored_rows"] == 1 and c["attempts_in_stored_rows"] == 258 and c["attempts_outside_stored_rows"] == 0 and c["cases_with_more_than_one_attempt"] == 1 and c["max_attempts_for_one_case"] == 2
    assert c["unknown_cost_attempts"] == 0 and c["unknown_cost_bound_charges_usd"] == 0 and c["components_reconcile_to_committed"] is True


def test_a_resumed_interruption_is_counted_as_an_attempt_and_its_bound_charge_is_reported_separately(frozen, tmp_path):
    work, sha = fresh(frozen, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        ce.run(work, sha, Scripted({5: "kill"}))
    t = ce.run(work, sha, Scripted(), resume=True, settle_unsettled=True)
    assert t["status"] == "COMPLETE" and t["unsettled"] == []
    c = ce.compare(work, sha)["cost"]
    assert c["journal_attempts_reserved"] == 258 and c["successful_final_judgments"] == 257 and c["attempts_outside_stored_rows"] == 1 and c["retry_attempts_within_stored_rows"] == 0
    assert c["attempts_with_known_usage"] == 257 and c["known_usage_cost_usd"] == pytest.approx(257 * PER_CALL, abs=1e-4)
    assert c["unknown_cost_attempts"] == 1 and 0.03 < c["unknown_cost_bound_charges_usd"] < 0.06  # the interrupted reservation, charged at its bound, never treated as zero
    assert c["committed_usd"] == pytest.approx(c["known_usage_cost_usd"] + c["unknown_cost_bound_charges_usd"], abs=1e-4) and c["unsettled_attempts"] == 0 and c["components_reconcile_to_committed"] is True
    assert c["cases_with_more_than_one_attempt"] == 1  # the interrupted case was redone


def test_missing_usage_is_charged_at_the_bound_reported_separately_and_never_counted_as_zero(frozen, tmp_path):
    work, sha = fresh(frozen, tmp_path)
    t = ce.run(work, sha, Scripted({3: "nousage", 40: "nousage"}))
    assert t["status"] == "COMPLETE" and t["ok"] == 257
    c = ce.compare(work, sha)["cost"]
    assert c["attempts_with_known_usage"] == 255 and c["unknown_cost_attempts"] == 2 and c["unknown_cost_bound_charges_usd"] > 0.08
    assert c["known_usage_cost_usd"] == pytest.approx(255 * PER_CALL, abs=1e-4) and c["known_usage_cost_usd"] < 257 * PER_CALL  # unknown usage is not silently added as known
    assert c["committed_usd"] == pytest.approx(c["known_usage_cost_usd"] + c["unknown_cost_bound_charges_usd"], abs=1e-4) and c["committed_usd"] == t["committed_usd"]


def test_an_unsent_request_counts_as_an_attempt_outside_the_stored_rows_with_a_bound_charge(frozen, tmp_path):
    work, sha = fresh(frozen, tmp_path)
    ce.run(work, sha, Scripted({3: "unsent"}))
    t = ce.run(work, sha, Scripted(), resume=True)
    assert t["status"] == "COMPLETE"
    c = ce.compare(work, sha)["cost"]
    assert c["journal_attempts_reserved"] == 258 and c["attempts_outside_stored_rows"] == 1 and c["unknown_cost_attempts"] == 1 and c["unknown_cost_bound_charges_usd"] > 0.03


def test_combined_retry_interruption_and_missing_usage_still_reconcile_to_the_journal_total(frozen, tmp_path):
    work, sha = fresh(frozen, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        ce.run(work, sha, Scripted({2: "drop", 6: "kill"}))
    t = ce.run(work, sha, Scripted({1: "nousage"}), resume=True, settle_unsettled=True)
    assert t["status"] == "COMPLETE"
    c = ce.compare(work, sha)["cost"]
    state = ce.sv3.journal_state(work / "spend_journal.jsonl", 3.0)
    assert c["committed_usd"] == round(state["committed"], 4) == t["committed_usd"] and c["components_reconcile_to_committed"] is True
    assert c["journal_attempts_reserved"] == c["successful_final_judgments"] + c["retry_attempts_within_stored_rows"] + c["attempts_outside_stored_rows"] == 257 + 1 + 1
    assert c["unknown_cost_attempts"] == 2 and c["cases_with_more_than_one_attempt"] == 2 and c["committed_usd"] <= c["cap_usd"] == 3.0


def test_the_v1_freeze_is_preserved_unrun_and_still_verifies_with_its_own_untouched_module():
    v1_protocol = ce.V1_WORK / "protocol.json"
    assert ce.V1_PROTOCOL_SHA256 == "287d8195f2d2a28c5cccf9aad1d189907f654c1e53497810fe4611ef0d4b6a69"
    if v1_protocol.exists():
        assert ce._sha(v1_protocol) == ce.V1_PROTOCOL_SHA256
        assert not any((ce.V1_WORK / n).exists() for n in ("results.jsonl", "spend_journal.jsonl", "terminal.json"))
        v1.verify(ce.V1_WORK, ce.V1_PROTOCOL_SHA256)  # the v1 module is unedited, so the v1 freeze remains verifiable
    assert ce.WORK != ce.V1_WORK and ce.CONFIRM != v1.CONFIRM


def test_v2_freeze_refuses_if_the_v1_freeze_was_edited_or_has_run_artifacts(frozen, tmp_path, monkeypatch):
    fake_v1 = tmp_path / "v1"
    fake_v1.mkdir()
    (fake_v1 / "protocol.json").write_text("{}", encoding="utf8")
    monkeypatch.setattr(ce, "V1_WORK", fake_v1)
    with pytest.raises(ce.ClaimFidelityEvalError, match="v1 freeze"):
        ce.freeze(tmp_path / "new")

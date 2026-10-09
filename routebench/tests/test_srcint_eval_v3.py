"""Tests for srcint-eval-v3: the v2 safeguards (all 38 v2 tests, adapted) plus the two follow-up fixes - raw-JSON model identity through the REAL client path and numeric-domain
journal validation. Mocked HTTP only: no provider call."""
import hashlib
import json
import re
from pathlib import Path

import pytest
from aisys import llm
from evalops import multi_judge as mj
from evalops import rubrics
from evalops import srcint_eval as v1
from evalops import srcint_eval_v3 as se
from evalops.audit2_v2 import Journal, SpendError

CRITERIA = sorted(rubrics.for_task("summarize").criterion_ids)
JOINS = ("labels.json", "baseline_cached_v18.json")


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    work = tmp_path_factory.mktemp("srcint3") / "w"
    return work, se.freeze(work), se._sha(work / "protocol.json")


def copy_work(frozen, tmp_path):
    work, _b, sha = frozen
    dst = tmp_path / "w"
    dst.mkdir()
    for p in work.iterdir():
        if p.name in {"protocol.json", "cases.jsonl", "population.json", *JOINS}:
            (dst / p.name).write_bytes(p.read_bytes())
    return dst, sha


class Fake:
    def __init__(self, flip_every=0, quota_at=None, bad=False, served="o4-mini-2025-04-16", overspend_at=None):
        self.calls, self.flip_every, self.quota_at, self.bad, self.served, self.overspend_at = [], flip_every, quota_at, bad, served, overspend_at

    def __call__(self, model, effort, temperature, max_tokens, messages):
        self.calls.append({"model": model, "effort": effort, "max_tokens": max_tokens, "prompt": messages[0]["content"]})
        n = len(self.calls)
        if self.quota_at == n:
            raise llm.QuotaExceededError("HTTP 429: insufficient_quota")
        if self.bad:
            return "not json", {"prompt_tokens": 500, "completion_tokens": 50}, self.served
        ok = not (self.flip_every and n % self.flip_every == 0)
        body = {"criteria": [{"id": c, "verdict": "yes" if ok else "no", "evidence": ""} for c in CRITERIA], "rationale": "synthetic"}
        usage = {"prompt_tokens": 900, "completion_tokens": 300}
        if self.overspend_at == n:
            usage = {"prompt_tokens": 900, "completion_tokens": 8000}  # at the bound of the reservation; used with a tiny cap below
        return json.dumps(body), usage, self.served


def run(work, sha, fake, **kw):
    return se.run(work, sha, fake, **kw)


# ---------------------------------------------------------------------------------------------------------------------------------- freeze
def test_freeze_binds_the_runtime_dependencies_and_preserves_v1(frozen):
    _w, b, _s = frozen
    assert {"audit2_v3.py", "multi_judge.py", "dataset.py", "run_calibration.py", "srcint_eval.py", "srcint_eval_v3.py", "matched_v16_v18.py", "judge.py", "aisys/llm.py"} <= set(b["code"])
    assert b["protocol"] == "srcint-eval-v3" and b["population"]["cases"] == 257 and b["population"]["removed_cases"] == 0 and b["cap_usd"] == 3.0
    assert "supersedes" in b and se.V1_PROTOCOL_SHA256 in b["supersedes"]
    v1_protocol = v1.WORK / "protocol.json"
    if v1_protocol.exists():
        assert v1._sha(v1_protocol) == se.V1_PROTOCOL_SHA256
        assert json.loads(v1_protocol.read_text())["population_digest"] == b["population_digest"]


# ---------------------------------------------------------------------------------------------------------------------------------- the review's seven regressions
@pytest.mark.parametrize("score", [float("nan"), 2.0, -0.1, True])
def test_invalid_scores_are_rejected_before_any_label_is_read(frozen, tmp_path, monkeypatch, score):
    work, sha = copy_work(frozen, tmp_path)
    run(work, sha, Fake())
    rows = [json.loads(x) for x in (work / "results.jsonl").read_text(encoding="utf8").splitlines()]
    rows[0]["record"]["score"] = score
    (work / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    real = Path.read_text

    def guard(path, *a, **k):
        if path.name in JOINS:
            raise AssertionError("labels reached before malformed-result rejection")
        return real(path, *a, **k)

    monkeypatch.setattr(Path, "read_text", guard)
    with pytest.raises(se.SrcIntV3Error):
        se.compare(work, sha)


def test_conflicting_or_identical_duplicate_results_are_refused_not_overwritten(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    run(work, sha, Fake())
    lines = (work / "results.jsonl").read_text(encoding="utf8").splitlines()
    dup = json.loads(lines[0])
    dup["record"]["score"] = 0.0
    (work / "results.jsonl").write_text("\n".join([*lines, json.dumps(dup)]) + "\n", encoding="utf8")
    (work / "terminal.json").unlink()

    def forbidden(*args):
        raise AssertionError("no network permitted")

    with pytest.raises(se.SrcIntV3Error, match="duplicate"):
        run(work, sha, forbidden, resume=True)
    with pytest.raises(se.SrcIntV3Error, match="duplicate"):
        se.compare(work, sha)


def test_a_run_already_over_its_cap_can_never_return_complete(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    run(work, sha, Fake())
    (work / "terminal.json").unlink()
    journal = Journal(work / "spend_journal.jsonl", 3.0)
    rid = journal.reserve("synthetic", 1, 0.10)
    with pytest.raises(SpendError):
        journal.settle(rid, 0.10, 3.10)

    def forbidden(*args):
        raise AssertionError("no network permitted")

    with pytest.raises(se.SrcIntV3Error, match="exceeds"):
        run(work, sha, forbidden, resume=True)
    assert json.loads((work / "terminal.json").read_text())["status"] == "REFUSED_OVER_CAP"
    with pytest.raises(se.SrcIntV3Error, match="permanently refused"):
        run(work, sha, forbidden, resume=True)
    assert se.compare(work, sha)["status"] == "WITHHELD_INCOMPLETE"


def test_frozen_dependencies_include_the_usage_validator_and_the_loader(frozen):
    assert {"audit2_v3.py", "multi_judge.py"} <= set(json.loads((frozen[0] / "protocol.json").read_text())["code"])


def test_freeze_loader_skips_heldout_before_annotation_validation(monkeypatch):
    assignment = json.loads((mj.DATA / "splits.json").read_text(encoding="utf8"))["assignment"]
    protected = {json.loads(line)["case_id"] for line in (mj.DATA / "cases.jsonl").read_bytes().splitlines() if assignment[json.loads(line)["group_id"]] == "test"}
    real_ann, real_case = mj.Annotation.model_validate_json, mj.CaseRecord.model_validate_json

    def ann_guard(line, *a, **k):
        if json.loads(line)["case_id"] in protected:
            raise AssertionError("held-out annotation reached typed validation")
        return real_ann(line, *a, **k)

    def case_guard(line, *a, **k):
        if json.loads(line)["case_id"] in protected:
            raise AssertionError("held-out case reached typed validation")
        return real_case(line, *a, **k)

    monkeypatch.setattr(mj.Annotation, "model_validate_json", ann_guard)
    monkeypatch.setattr(mj.CaseRecord, "model_validate_json", case_guard)
    assert len(mj.load_cases(["train", "dev"])) == 257
    with pytest.raises(ValueError, match="sealed"):
        mj.load_cases(["test"])
    with pytest.raises(ValueError):
        mj.load_cases(["train", "test"])


def test_the_bounded_loader_returns_exactly_what_the_old_loader_did():
    items = mj.load_cases(["train", "dev"])
    assert len(items) == 257 and sum(s == "dev" for _c, _y, s in items) == 128 and sum(s == "train" for _c, _y, s in items) == 129
    assert all(re.fullmatch(r"gnd-[0-9a-f]{16}", c.case_id) for c, _y, _s in items)


def test_missing_served_model_identity_makes_the_judgment_unverifiable_stops_the_run_and_keeps_the_charge(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    f = Fake(served="")
    t = run(work, sha, f)
    assert t["status"] == "ABORTED" and len(f.calls) == 1 and len(t["failed"]) == 1 and "identity" in t["stopped_on"]
    row = json.loads((work / "results.jsonl").read_text(encoding="utf8").splitlines()[0])
    assert row["record"]["status"] == "failed" and "unverifiable" in row["record"]["reason"] and row["record"]["identity_failure"]
    assert se.journal_state(work / "spend_journal.jsonl", 3.0)["committed"] > 0  # the attempt's charge was retained


# ---------------------------------------------------------------------------------------------------------------------------------- row validation
def test_a_clean_run_writes_bound_rows_with_raw_responses_and_settled_attempts(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    f = Fake()
    t = run(work, sha, f)
    assert t["status"] == "COMPLETE" and t["ok"] == 257 and t["unsettled"] == [] and t["committed_usd"] <= 3.0 and len(f.calls) == 257
    row = json.loads((work / "results.jsonl").read_text(encoding="utf8").splitlines()[0])
    rec = row["record"]
    assert row["protocol_sha256"] == sha and rec["raw_response"] and rec["attempt_ids"] and rec["config_hash"] == "12fc8b550d954359" and rec["model_served"].startswith("o4-mini")
    assert not (work / "run.lock").exists() and (work / "invocations.jsonl").exists()


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r["record"].update(raw_response=r["record"]["raw_response"].replace("yes", "no")), "differ from the stored raw response"),
    (lambda r: r["record"].update(prompt_sha256="0" * 64), "binding"), (lambda r: r["record"].update(config_hash="0" * 16), "binding"),
    (lambda r: r["record"].update(rubric_hash="0" * 16), "binding"), (lambda r: r["record"].update(model_served="gpt-4o"), "served model"),
    (lambda r: r["record"].update(model_served=""), "served model"), (lambda r: r["record"].update(raw_response=""), "raw response"),
    (lambda r: r["record"].update(attempt_ids=["nope#1#0"]), "not a settled reservation"), (lambda r: r["record"].update(attempt_ids=[]), "attempt ids"),
    (lambda r: r.update(protocol_sha256="0" * 64), "not bound to this protocol"), (lambda r: r.update(case_id="foreign-case"), "foreign"),
    (lambda r: r.update(extra=1), "row shape"), (lambda r: r["record"].update(status="weird"), "unknown status"),
])
def test_tampered_or_unbound_rows_are_refused(frozen, tmp_path, mutate, needle):
    work, sha = copy_work(frozen, tmp_path)
    run(work, sha, Fake())
    rows = [json.loads(x) for x in (work / "results.jsonl").read_text(encoding="utf8").splitlines()]
    mutate(rows[0])
    (work / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    with pytest.raises(se.SrcIntV3Error, match=needle):
        se.validate_rows(work, *se.verify(work, sha), se.journal_state(work / "spend_journal.jsonl", 3.0), sha)


def test_an_attempt_id_cannot_be_borrowed_from_another_case(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    run(work, sha, Fake())
    rows = [json.loads(x) for x in (work / "results.jsonl").read_text(encoding="utf8").splitlines()]
    rows[1]["record"]["attempt_ids"] = rows[0]["record"]["attempt_ids"]
    (work / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    with pytest.raises(se.SrcIntV3Error, match="not a settled reservation of this case"):
        se.validate_rows(work, *se.verify(work, sha), se.journal_state(work / "spend_journal.jsonl", 3.0), sha)


# ---------------------------------------------------------------------------------------------------------------------------------- journal / terminal / lock
def test_unsettled_reservations_block_resume_until_settled_at_their_bound(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    run(work, sha, Fake(quota_at=6))
    j = Journal(work / "spend_journal.jsonl", 3.0)
    j.reserve(f"{se.VARIANT}:crashed", 1, 0.05)  # a crash between reserve and settle
    before = se.journal_state(work / "spend_journal.jsonl", 3.0)["committed"]
    with pytest.raises(se.SrcIntV3Error, match="unsettled"):
        run(work, sha, Fake(), resume=True)
    assert not (work / "run.lock").exists()
    t = run(work, sha, Fake(), resume=True, settle_unsettled=True)
    assert t["status"] == "COMPLETE" and t["unsettled"] == []
    assert se.journal_state(work / "spend_journal.jsonl", 3.0)["committed"] >= before  # the charge was kept, never discarded


def test_an_aborted_run_needs_an_explicit_resume_and_a_complete_run_is_final(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    t1 = run(work, sha, Fake(quota_at=11))
    assert t1["status"] == "ABORTED" and t1["ok"] == 10
    with pytest.raises(se.SrcIntV3Error, match="explicit --resume"):
        run(work, sha, Fake())
    f = Fake()
    t2 = run(work, sha, f, resume=True)
    assert t2["status"] == "COMPLETE" and len(f.calls) == 247
    with pytest.raises(se.SrcIntV3Error, match="final"):
        run(work, sha, Fake(), resume=True)


def test_a_second_runner_is_refused_while_the_lock_exists(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    (work / "run.lock").write_text("{}")
    f = Fake()
    with pytest.raises(se.SrcIntV3Error, match="another runner"):
        run(work, sha, f)
    assert f.calls == [] and (work / "run.lock").exists()  # the foreign lock is not removed


def test_terminal_status_is_written_atomically_and_failed_cases_are_retained(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    t = run(work, sha, Fake(bad=True))
    assert t["status"] == "INCOMPLETE" and len(t["failed"]) == 257 and not (work / "terminal.json.tmp").exists()
    out = se.compare(work, sha)
    assert out["status"] == "WITHHELD_INCOMPLETE" and any("terminal" in p for p in out["problems"])


def test_comparison_gates_run_before_labels_are_read(frozen, tmp_path, monkeypatch):
    work, sha = copy_work(frozen, tmp_path)
    run(work, sha, Fake(quota_at=5))
    real, touched = Path.read_text, []

    def guard(path, *a, **k):
        if path.name in JOINS:
            touched.append(path.name)
        return real(path, *a, **k)

    monkeypatch.setattr(Path, "read_text", guard)
    assert se.compare(work, sha)["status"] == "WITHHELD_INCOMPLETE" and touched == []


# ---------------------------------------------------------------------------------------------------------------------------------- reporting
def test_complete_run_reports_dev_first_with_descriptive_flip_counts_and_no_p_value(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    run(work, sha, Fake(flip_every=7))
    out = se.compare(work, sha)
    text = json.dumps(out)
    assert out["status"] == "COMPLETE" and out["dev_primary"]["n"] == 128 and out["train_diagnostic"]["n"] == 129 and out["removed_cases"] == 0
    assert "fisher" not in text.lower() and "p_value" not in text and "noise_control" not in text
    d = out["descriptive_verdict_flips"]
    assert d["changed_inputs"]["n"] == 63 and d["unchanged_inputs"]["n"] == 194 and "COUNTS ONLY" in d["caveat"] and "does not isolate" in out["interpretation"]
    assert out["dev_primary"]["kappa_difference_fresh_minus_cached_95_document_bootstrap"]
    with pytest.raises(se.SrcIntV3Error, match="written once"):
        se.compare(work, sha)


@pytest.mark.parametrize("name", ["cases.jsonl", "population.json", *JOINS])
def test_altered_inputs_after_the_freeze_are_refused_before_any_call(frozen, tmp_path, name):
    work, sha = copy_work(frozen, tmp_path)
    (work / name).write_text((work / name).read_text(encoding="utf8") + " ", encoding="utf8")
    f = Fake()
    with pytest.raises(se.SrcIntV3Error):
        run(work, sha, f)
    assert f.calls == [] and not (work / "run.lock").exists()


def test_changed_code_or_wrong_hash_or_missing_confirmation_is_refused(frozen, tmp_path, monkeypatch, capsys):
    work, sha = copy_work(frozen, tmp_path)
    with pytest.raises(se.SrcIntV3Error, match="freeze time"):
        run(work, "0" * 64, Fake())
    monkeypatch.setattr(se, "code_hashes", lambda: {"x": "y"})
    with pytest.raises(se.SrcIntV3Error, match="code"):
        run(work, sha, Fake())
    assert se.main(["run", "--protocol-sha256", "0" * 64]) == 2 and "--confirm" in capsys.readouterr().err


def test_v1_protocol_and_original_files_are_untouched():
    manifest = json.loads((se.SRC / "manifest.json").read_text(encoding="utf8"))
    for path, digest in manifest["inputs"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


# ================================================================================================================ follow-up fix 1: identity from the RAW provider JSON, real client path
import httpx
from evalops.dataset import CaseRecord


def _answer(ok=True):
    return {"criteria": [{"id": c, "verdict": "yes" if ok else "no", "evidence": ""} for c in CRITERIA], "rationale": "synthetic"}


@pytest.fixture
def http(monkeypatch):
    """Replace only the HTTP transport of the real aisys client; returns a setter for the raw provider payload."""
    state = {"payload": None, "posts": 0}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            state["posts"] += 1
            return httpx.Response(200, json=state["payload"], request=httpx.Request("POST", "https://unit-test.invalid/chat/completions"))

    monkeypatch.setattr(llm.httpx, "Client", FakeClient)
    monkeypatch.setattr(llm.settings, "openai_api_key", "synthetic-test-key")
    return state


def payload(model="__absent__", usage=None):
    p = {"choices": [{"message": {"content": json.dumps(_answer())}}], "usage": usage if usage is not None else {"prompt_tokens": 900, "completion_tokens": 300}}
    if model != "__absent__":
        p["model"] = model
    return p


CASE = CaseRecord(case_id="synthetic", group_id="synthetic-group", task_class="summarize", task_input="Summarize faithfully.", context="The council met on Monday.",
                  candidate_output="The council met on Monday.")


def judge_with_production_adapter(tmp_path):
    import inspect
    default = inspect.signature(se.run).parameters["request"].default
    assert default is se.http_request_raw  # the default production adapter is the one under test
    journal = se.Journal(tmp_path / "journal.jsonl", 3.0)
    return se.judge_case(CASE, journal, default, "synthetic-protocol"), journal


def test_a_response_without_a_model_field_is_not_credited_to_the_requested_model(tmp_path, http):
    http["payload"] = payload()
    row, _unused = judge_with_production_adapter(tmp_path)
    assert row["status"] == "failed" and row["identity_failure"] and "unverifiable" in row["reason"] and row["attempt_ids"]
    st = se.journal_state(tmp_path / "journal.jsonl", 3.0)
    assert st["committed"] > 0 and not st["unsettled"]  # usage accounting preserved: the billed attempt is charged
    assert st["committed"] == pytest.approx((900 * 1.10 + 300 * 4.40) / 1e6, rel=1e-6)


@pytest.mark.parametrize("model", ["", "   ", None, 123, ["o4-mini"], {"name": "o4-mini"}, "gpt-4o-2024", "claude-opus-5-5"])
def test_blank_malformed_or_wrong_family_model_metadata_is_unverifiable_with_the_charge_retained(tmp_path, http, model):
    http["payload"] = payload(model=model)
    row, _j = judge_with_production_adapter(tmp_path)
    assert row["status"] == "failed" and row["identity_failure"]
    assert se.journal_state(tmp_path / "journal.jsonl", 3.0)["committed"] > 0


def test_a_valid_provider_model_value_passes_through_the_real_client_path(tmp_path, http):
    http["payload"] = payload(model="o4-mini-2025-04-16")
    row, _j = judge_with_production_adapter(tmp_path)
    assert row["status"] == "ok" and row["model_served"] == "o4-mini-2025-04-16" and row["score"] == 1.0 and row["raw_response"]


def test_the_adapter_returns_the_raw_value_not_the_clients_fallback(http):
    http["payload"] = payload()
    _text, usage, served = se.http_request_raw("o4-mini", "medium", 0.0, 8000, [{"role": "user", "content": "x"}])
    assert served is None and usage == {"prompt_tokens": 900, "completion_tokens": 300}
    http["payload"] = payload(model="o4-mini-2025-04-16")
    assert se.http_request_raw("o4-mini", "medium", 0.0, 8000, [{"role": "user", "content": "x"}])[2] == "o4-mini-2025-04-16"


def test_a_run_through_the_production_adapter_stops_after_the_first_unverifiable_identity(frozen, tmp_path, http):
    work, sha = copy_work(frozen, tmp_path)
    http["payload"] = payload()  # no model field, for every call
    t = se.run(work, sha)  # default request = the production adapter
    assert t["status"] == "ABORTED" and http["posts"] == 1 and len(t["failed"]) == 1 and t["committed_usd"] > 0
    assert se.compare(work, sha)["status"] == "WITHHELD_INCOMPLETE"


# ================================================================================================================ follow-up fix 2: numeric-domain validation in the strict journal
BAD_AMOUNTS = [float("nan"), float("inf"), -float("inf"), -0.01, True, False, "0.01", None]


def _journal(tmp_path, entries):
    p = tmp_path / "journal.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf8")
    return p


@pytest.mark.parametrize("amount", BAD_AMOUNTS)
def test_strict_journal_refuses_invalid_reservation_amounts(tmp_path, amount):
    p = _journal(tmp_path, [{"op": "reserve", "id": "r", "case": "synthetic", "attempt": 1, "usd": amount}, {"op": "settle", "id": "r", "charged": 0.01, "actual": 0.01}])
    before = p.read_bytes()
    with pytest.raises(se.SrcIntV3Error):
        se.journal_state(p, 3.0)
    assert p.read_bytes() == before  # nothing was repaired, zeroed or erased


@pytest.mark.parametrize("amount", BAD_AMOUNTS)
def test_strict_journal_refuses_invalid_settlement_amounts(tmp_path, amount):
    p = _journal(tmp_path, [{"op": "reserve", "id": "r", "case": "synthetic", "attempt": 1, "usd": 0.05}, {"op": "settle", "id": "r", "charged": amount, "actual": 0.01}])
    with pytest.raises(se.SrcIntV3Error):
        se.journal_state(p, 3.0)


@pytest.mark.parametrize("actual", [float("nan"), -0.01, True, "x"])
def test_strict_journal_refuses_invalid_recorded_actual_costs(tmp_path, actual):
    p = _journal(tmp_path, [{"op": "reserve", "id": "r", "case": "synthetic", "attempt": 1, "usd": 0.05}, {"op": "settle", "id": "r", "charged": 0.01, "actual": actual}])
    with pytest.raises(se.SrcIntV3Error):
        se.journal_state(p, 3.0)


def test_valid_journals_including_a_null_actual_are_accepted_and_totals_keep_every_charge(tmp_path):
    p = _journal(tmp_path, [{"op": "reserve", "id": "a", "case": "x", "attempt": 1, "usd": 0.05}, {"op": "settle", "id": "a", "charged": 0.05, "actual": None},
                            {"op": "reserve", "id": "b", "case": "x", "attempt": 2, "usd": 0.02}, {"op": "settle", "id": "b", "charged": 0.01, "actual": 0.01},
                            {"op": "reserve", "id": "c", "case": "x", "attempt": 3, "usd": 0.07}])
    st = se.journal_state(p, 3.0)
    assert st["committed"] == pytest.approx(0.05 + 0.01 + 0.07) and st["unsettled"] == ["c"]
    with pytest.raises(se.SrcIntV3Error, match="not an object"):
        se.journal_state(_journal(tmp_path, [[1, 2]]), 3.0)


def test_a_malformed_journal_blocks_a_run_before_any_call(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    (work / "spend_journal.jsonl").write_text(json.dumps({"op": "reserve", "id": "r", "case": "s", "attempt": 1, "usd": float("nan")}) + "\n", encoding="utf8")
    f = Fake()
    with pytest.raises(se.SrcIntV3Error):
        run(work, sha, f)
    assert f.calls == [] and not (work / "run.lock").exists()


# ================================================================================================================ preservation
def test_v1_and_v2_freezes_and_code_are_preserved_and_the_v3_freeze_supersedes_them(frozen):
    _w, b, _s = frozen
    assert se.V1_PROTOCOL_SHA256 in b["supersedes"] and se.V2_PROTOCOL_SHA256 in b["supersedes"] and b["protocol"] == "srcint-eval-v3" and "srcint_eval_v3.py" in b["code"]
    v2_protocol = se.ROOT / ".local" / "srcint-eval-v2" / "protocol.json"
    if v2_protocol.exists():
        assert se._sha(v2_protocol) == se.V2_PROTOCOL_SHA256
        assert json.loads(v2_protocol.read_text())["population_digest"] == b["population_digest"]
        assert not (v2_protocol.parent / "results.jsonl").exists() and not (v2_protocol.parent / "spend_journal.jsonl").exists()
    assert se.WORK != se.ROOT / ".local" / "srcint-eval-v2" and se.CONFIRM == "RUN-SRCINT-OPENAI-V3" and b["cap_usd"] == 3.0 and b["population"]["cases"] == 257


def test_the_adapter_sends_the_same_request_body_as_the_project_client(monkeypatch):
    """The recipe must not drift: the evaluation adapter's request equals what aisys.llm._call sends for the same arguments."""
    sent = []

    class Capture:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            sent.append((url, headers, json))
            return httpx.Response(200, json=payload(model="o4-mini-2025-04-16"), request=httpx.Request("POST", "https://unit-test.invalid/x"))

    monkeypatch.setattr(llm.httpx, "Client", Capture)
    monkeypatch.setattr(llm.settings, "openai_api_key", "synthetic-test-key")
    messages = [{"role": "user", "content": "x"}]
    se.http_request_raw("o4-mini", "medium", 0.0, 8000, messages)
    llm._call("o4-mini", messages, None, 0.0, 8000, {"reasoning_effort": "medium"})
    assert len(sent) == 2 and sent[0] == sent[1] and sent[0][2]["max_completion_tokens"] == 8000 and "temperature" not in sent[0][2]


@pytest.mark.parametrize("status,body,exc", [(429, {"error": {"code": "insufficient_quota"}}, llm.QuotaExceededError), (503, {}, llm.ProviderError)])
def test_quota_and_server_errors_keep_their_meaning_in_the_adapter(monkeypatch, status, body, exc):
    class C:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *a, **k):
            return httpx.Response(status, json=body, request=httpx.Request("POST", "https://unit-test.invalid/x"))

    monkeypatch.setattr(llm.httpx, "Client", C)
    with pytest.raises(exc):
        se.http_request_raw("o4-mini", "medium", 0.0, 8000, [{"role": "user", "content": "x"}])

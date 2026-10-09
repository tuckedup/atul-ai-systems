"""Tests for the frozen, resumable continuation protocol. A fake CLI replaces `claude`: no model call, no subscription use."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from evalops import claim_relation as v1
from evalops import claim_relation_claude as cc
from evalops import claim_relation_claude_cont as ct

FIXTURES = v1.synthetic_fixtures()
BY_PROMPT = {v1.sha256(v1.build_prompt(f["source"], f["candidate"])): f for f in FIXTURES}
GOOD_AUTH = {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": "pro"}
N = len(FIXTURES)
BAD = "```json\n{\"assertions\": [ {\"candidate_span\" \"oops\"} ]}\n```"  # malformed JSON


class Fake:
    """Fake `claude`: `script(n, case_id, prompt)` returns a CLI json dict, a raw string, or raises."""

    def __init__(self, script=None):
        self.calls, self.script = [], script or (lambda n, cid, p: None)

    def __call__(self, args, **kw):
        self.calls.append((list(args), kw))
        if args[1:3] == ["auth", "status"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(GOOD_AUTH), stderr="")
        if args[1:] == ["--version"]:
            return SimpleNamespace(returncode=0, stdout="9.9.9 (Claude Code)\n", stderr="")
        f = BY_PROMPT[v1.sha256(kw["input"])]
        n = len(self.judging())
        out = self.script(n, f["case_id"], kw["input"])
        if out is None:
            out = good(f)
        elif isinstance(out, str) and out == "MALFORMED":
            out = {**good(f), "result": BAD}
        return SimpleNamespace(returncode=0, stdout=out if isinstance(out, str) else json.dumps(out), stderr="")

    def judging(self):
        return [c for c in self.calls if "-p" in c[0]]

    def case_ids(self):
        return [BY_PROMPT[v1.sha256(c[1]["input"])]["case_id"] for c in self.judging()]


def good(f):
    return {"is_error": False, "subtype": "success", "result": v1.scripted_response(f), "modelUsage": {cc.MODEL: {}}, "total_cost_usd": 0.1, "num_turns": 1,
            "usage": {"input_tokens": 5, "output_tokens": 1000, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 0}, "session_id": f"sess-{f['case_id']}-{id(f)}"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(cc.shutil, "which", lambda name: "claude")
    for k in cc.FORBIDDEN_ENV:
        monkeypatch.delenv(k, raising=False)


def make_inputs(root):
    root.mkdir(parents=True, exist_ok=True)
    files = {"cases.json": [{"case_id": f["case_id"], "source": f["source"], "candidate": f["candidate"]} for f in FIXTURES],
             "labels.json": {f["case_id"]: f["label"] for f in FIXTURES}, "groups.json": {f["case_id"]: f["document_family"] for f in FIXTURES},
             "baseline_v18.json": {f["case_id"]: 1 for f in FIXTURES}, "splits.json": {f["case_id"]: "dev" for f in FIXTURES}}
    for n, v in files.items():
        (root / n).write_text(json.dumps(v, sort_keys=True), encoding="utf8")
    return root


class Env:
    """An 'original' INCOMPLETE run (malformed at case index `stop_at`) under the original protocol, plus the continuation freeze."""

    def __init__(self, tmp_path, monkeypatch, stop_at=5):
        self.tmp, self.mp = tmp_path, monkeypatch
        self.inputs = make_inputs(tmp_path / "in")
        self.orig_fake = Fake(lambda n, cid, p: "MALFORMED" if n == stop_at else None)
        self.orig_proto = tmp_path / "orig_protocol.json"
        cc.freeze(self.inputs, self.orig_proto, self.orig_fake)
        self.orig_sha = cc._sha_file(self.orig_proto)
        self.orig_run = tmp_path / "orig_run"
        cc.run(self.inputs, self.orig_proto, self.orig_run, cc.make_request(self.orig_fake), expected_sha256=self.orig_sha, runner=self.orig_fake)
        monkeypatch.setattr(ct, "ORIGINAL_PROTOCOL_SHA", self.orig_sha)
        self.work = tmp_path / "work"
        self.fake = Fake()
        ct.freeze(self.inputs, self.work, self.fake, self.orig_run, self.orig_proto)
        self.sha = ct._sha(self.work / "protocol.json")

    def run(self, fake=None, **kw):
        f = fake or self.fake
        return ct.run(self.inputs, self.work, cc.make_request(f), expected_sha256=self.sha, runner=f, original_run=self.orig_run, original_protocol=self.orig_proto, **kw)

    def compare(self, fake=None):
        f = fake or self.fake
        return ct.compare(self.inputs, self.work, expected_sha256=self.sha, runner=f, original_run=self.orig_run, original_protocol=self.orig_proto)


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


# ------------------------------------------------------------------------------------------------ original run and import provenance
def test_the_original_run_is_the_incomplete_one_we_continue_from(env):
    t = json.loads((env.orig_run / "terminal.json").read_text())
    assert t["status"] == "INCOMPLETE" and t["attempted"] == 5 and t["ok"] == 4 and len(t["not_attempted"]) == N - 5


def test_freeze_discloses_the_reuse_and_counts_pending_cases(env):
    p = json.loads((env.work / "protocol.json").read_text())["import_provenance"]
    assert p["valid_judgments_imported"] == 4 and p["pending_cases"] == N - 4 and len(p["malformed_attempt_imported_as_attempt_1"]) == 1
    assert "reused unchanged after identity re-verification" in p["disclosure"]
    rule = json.loads((env.work / "protocol.json").read_text())["retry_rule"]
    assert rule["max_attempts_per_case_for_malformed_output"] == 2 and rule["gold_labels_used"] is False


def _tamper(env, mutate):
    lines = (env.orig_run / "judgments.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    mutate(rec)
    lines[0] = json.dumps(rec)
    (env.orig_run / "judgments.jsonl").write_text("\n".join(lines) + "\n", encoding="utf8")


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(raw_response=r["raw_response"].replace("supported", "unsupported", 1)),
    lambda r: r.update(model="claude-haiku-4-5"), lambda r: r.update(content_hash="0" * 64), lambda r: r.update(prompt_sha256="0" * 64),
    lambda r: r["input"].update(candidate=r["input"]["candidate"] + " edited"), lambda r: r.update(usage=None), lambda r: r.update(variant="claim-relation-v1"),
    lambda r: r["structured"].update(verdict="supported" if r["structured"]["verdict"] != "supported" else "unsupported"),
])
def test_tampered_imported_judgments_are_refused_before_any_model_call(env, mutate):
    _tamper(env, mutate)
    with pytest.raises(v1.ClaimRelationError):
        env.run()
    assert env.fake.judging() == []


def test_changed_original_files_or_protocol_are_refused(env):
    (env.orig_run / "terminal.json").write_text((env.orig_run / "terminal.json").read_text().replace('"INCOMPLETE"', '"COMPLETE"'), encoding="utf8")
    with pytest.raises(v1.ClaimRelationError):
        env.run()
    assert env.fake.judging() == []


def test_the_original_run_files_are_never_modified(env):
    before = {p.name: ct._sha(p) for p in env.orig_run.iterdir()}
    env.run()
    env.compare()
    assert {p.name: ct._sha(p) for p in env.orig_run.iterdir()} == before


# ------------------------------------------------------------------------------------------------ retry rule
def test_only_pending_cases_are_judged_and_valid_imports_are_not_rejudged(env):
    t = env.run()
    judged = env.fake.case_ids()
    imported_valid = {json.loads(x)["case_id"] for x in (env.orig_run / "judgments.jsonl").read_text().splitlines() if json.loads(x)["status"] == "ok"}
    assert not imported_valid & set(judged) and len(judged) == N - 4 and t["status"] == "COMPLETE" and t["imported_valid"] == 4 and t["new_valid"] == N - 4


def test_the_original_malformed_attempt_counts_as_attempt_one_so_that_case_gets_one_retry(env):
    failed_case = json.loads((env.orig_run / "terminal.json").read_text())["stopped_on"]["case_id"]
    t = env.run(Fake(lambda n, cid, p: "MALFORMED" if cid == failed_case else None))
    assert t["status"] == "INCOMPLETE" and t["failed_after_retry"] == [failed_case]
    rows = [r for r in ct.State(env.work / "attempts.jsonl").rows if r["case_id"] == failed_case and r.get("kind", "result") == "result"]
    assert [r["origin"] for r in rows] == ["imported", "new"] and [r["record"]["status"] for r in rows] == ["invalid_structure", "invalid_structure"]  # exactly one retry


def test_malformed_then_valid_is_retried_once_with_the_unchanged_prompt_and_settings(env):
    seen = {}

    def script(n, cid, p):
        seen.setdefault(cid, []).append(p)
        return "MALFORMED" if cid == FIXTURES[10]["case_id"] and len(seen[cid]) == 1 else None

    f = Fake(script)
    t = env.run(f)
    cid = FIXTURES[10]["case_id"]
    assert t["status"] == "COMPLETE" and len(seen[cid]) == 2 and seen[cid][0] == seen[cid][1] == v1.build_prompt(FIXTURES[10]["source"], FIXTURES[10]["candidate"])
    args = [c[0] for c in f.judging()]
    assert all(a[1:1 + len(cc.CLI_FLAGS)] == cc.CLI_FLAGS for a in args)  # identical model and settings on every attempt
    assert t["attempt_counts"]["2"] == 2  # the originally failed case and this one
    rows = [r for r in ct.State(env.work / "attempts.jsonl").rows if r["case_id"] == cid]
    assert [r["kind"] for r in rows] == ["reserve", "result", "reserve", "result"]  # every attempt reserved first, then settled


def test_malformed_twice_retains_the_case_never_a_third_attempt_and_the_run_continues(env):
    bad = FIXTURES[12]["case_id"]
    f = Fake(lambda n, cid, p: "MALFORMED" if cid == bad else None)
    t = env.run(f)
    assert f.case_ids().count(bad) == 2 and t["status"] == "INCOMPLETE" and bad in t["failed_after_retry"]
    assert t["valid_judgments"] == N - len(t["failed_after_retry"])  # every other case was still judged
    again = Fake(lambda n, cid, p: "MALFORMED" if cid == bad else None)
    t2 = env.run(again)
    assert again.judging() == [] and t2["failed_after_retry"] == t["failed_after_retry"]  # restart does not grant extra retries


def test_a_failed_case_withholds_kappa_and_no_label_file_is_parsed(env, monkeypatch):
    bad = FIXTURES[12]["case_id"]
    env.run(Fake(lambda n, cid, p: "MALFORMED" if cid == bad else None))
    real, touched = Path.read_text, []

    def guard(self, *a, **k):
        if self.name in cc.JOINS:
            touched.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", guard)
    out = env.compare()
    assert out["status"] == "WITHHELD_INCOMPLETE" and out["missing_or_failed"] == [bad] and touched == [] and not (env.work / "comparison.json").exists()
    assert "no label, group, baseline or split file was read" in out["note"]


def test_retry_decisions_never_read_labels(env, monkeypatch):
    real, touched = Path.read_text, []

    def guard(self, *a, **k):
        if self.name in cc.JOINS:
            touched.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", guard)
    env.run(Fake(lambda n, cid, p: "MALFORMED" if n == 2 else None))
    assert touched == []


# ------------------------------------------------------------------------------------------------ request errors and restart recovery
def test_request_errors_stop_the_run_do_not_use_the_retry_and_resume_cleanly(env):
    def script(n, cid, p):
        return {"is_error": True, "subtype": "error", "result": "usage limit", "api_error_status": 429, "modelUsage": {}} if n == 3 else None

    f = Fake(script)
    t = env.run(f)
    assert t["status"] == "ABORTED" and len(f.judging()) == 3 and t["valid_judgments"] == 4 + 2
    stopped = t["stopped_on"]["case_id"]
    assert ct.State(env.work / "attempts.jsonl").malformed(stopped) == (1 if stopped == json.loads((env.orig_run / "terminal.json").read_text())["stopped_on"]["case_id"] else 0)
    resume = Fake()
    t2 = env.run(resume)
    assert t2["status"] == "COMPLETE" and stopped in resume.case_ids() and t2["valid_judgments"] == N
    assert len(set(resume.case_ids())) == len(resume.case_ids()) and not set(resume.case_ids()) & {r["case_id"] for r in ct.State(env.work / "attempts.jsonl").rows[:5]}


def test_restart_after_a_crash_mid_write_recovers_from_the_log(env):
    crash = Fake(lambda n, cid, p: (_ for _ in ()).throw(KeyboardInterrupt) if n == 4 else None)
    with pytest.raises(KeyboardInterrupt):
        env.run(crash)
    log = env.work / "attempts.jsonl"
    log.write_text(log.read_text() + '{"case_id": "syn-partial", "origin": "new", "reco', encoding="utf8")  # a torn final line
    resume = Fake()
    t = env.run(resume)
    assert t["status"] == "COMPLETE" and t["truncated_log_line_recovered"] is True and t["valid_judgments"] == N
    assert len(resume.judging()) == N - 4 - 3  # the 3 cases finished before the crash are not re-judged


def test_corruption_before_the_last_line_is_refused(env):
    env.run(Fake())
    log = env.work / "attempts.jsonl"
    lines = log.read_text().split("\n")
    lines[2] = "{broken"
    log.write_text("\n".join(lines), encoding="utf8")
    with pytest.raises(v1.ClaimRelationError, match="corrupt"):
        env.run(Fake())


def test_the_new_call_cap_is_enforced(env, monkeypatch):
    with pytest.raises(v1.ProtocolError, match="max_new_model_calls"):  # the cap is part of the frozen protocol: changing it after the freeze is refused
        monkeypatch.setattr(ct, "MAX_NEW_MODEL_CALLS", 3)
        env.run()
    env.work = env.tmp / "work2"  # a protocol frozen WITH the smaller cap
    ct.freeze(env.inputs, env.work, env.fake, env.orig_run, env.orig_proto)
    env.sha = ct._sha(env.work / "protocol.json")
    f = Fake()
    t = env.run(f)
    assert len(f.judging()) == 3 and t["status"] == "ABORTED" and t["stopped_on"]["status"] == "call_cap"


# ------------------------------------------------------------------------------------------------ full result, freeze integrity, publication
def test_complete_128_of_128_publishes_the_comparison_once_with_provenance(env):
    t = env.run()
    assert t["status"] == "COMPLETE" and t["valid_judgments"] == N and t["models_reported"] == [cc.MODEL]
    out = env.compare()
    assert out["status"] == "COMPLETE" and out["dev_primary"]["n"] == N and out["run"]["imported_valid"] == 4 and out["run"]["coverage"] == f"{N}/{N}"
    with pytest.raises(v1.ClaimRelationError, match="written once"):
        env.compare()


@pytest.mark.parametrize("name", ["cases.json", "labels.json", "groups.json", "baseline_v18.json", "splits.json"])
def test_altered_inputs_after_freeze_are_refused_before_any_call(env, name):
    p = env.inputs / name
    p.write_text(p.read_text() + " ", encoding="utf8")
    with pytest.raises(v1.ProtocolError):
        env.run()
    assert env.fake.judging() == []


def test_wrong_protocol_hash_or_missing_confirmation_is_refused(env, capsys):
    with pytest.raises(v1.ProtocolError, match="freeze time"):
        ct.run(env.inputs, env.work, cc.make_request(env.fake), expected_sha256="0" * 64, runner=env.fake, original_run=env.orig_run, original_protocol=env.orig_proto)
    assert ct.main(["run", "--protocol-sha256", "0" * 64]) == 2 and "--confirm" in capsys.readouterr().err
    assert ct.main(["run"]) == 2


def test_subscription_authentication_is_still_required(env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    with pytest.raises(cc.AuthError):
        env.run()
    assert env.fake.judging() == []


def test_usage_estimate_comes_from_the_original_calls(env):
    est = ct.usage_estimate(env.orig_run, pending=N - 4)
    assert est["hard_bound_on_calls"]["if_no_request_errors"] == 2 * (N - 4) and est["expected_calls_estimate"] >= N - 4
    assert "worst_case" not in json.dumps(est) and "estimate" in est["note"]  # averages are labelled as estimates, never as worst cases
    assert (est["two_attempts_for_every_case_estimate_at_max_observed_cost"]["notional_list_cost_usd"]
            >= est["two_attempts_for_every_case_estimate_at_mean_cost"]["notional_list_cost_usd"] >= est["expected_estimate"]["notional_list_cost_usd"])


def test_real_preserved_run_matches_the_continuation_expectations():
    if not (ct.ORIGINAL_RUN / "terminal.json").exists():
        pytest.skip("preserved original run not present")
    recs = ct.import_original(ct.INPUTS, ct.ORIGINAL_RUN, ct.ORIGINAL_PROTOCOL)
    assert sum(r["status"] == "ok" for r in recs) == 81 and len(recs) == 82 and ct.usage_estimate()["pending_cases"] == 47


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(raw_response=r["raw_response"].replace("supported", "unsupported", 1)),
    lambda r: r.update(model="claude-haiku-4-5"), lambda r: r.update(content_hash="0" * 64), lambda r: r.update(prompt_sha256="0" * 64),
    lambda r: r["input"].update(candidate=r["input"]["candidate"] + " edited"), lambda r: r.update(usage=None), lambda r: r.update(variant="claim-relation-v1"),
    lambda r: r["structured"].update(verdict="unsupported" if r["structured"]["verdict"] != "unsupported" else "supported"), lambda r: r.update(prompt="other"),
    lambda r: r.update(status="aborted"),
])
def test_record_level_identity_verification_rejects_each_tampering_independently_of_file_hashes(env, mutate):
    rec = json.loads((env.orig_run / "judgments.jsonl").read_text().splitlines()[0])
    case = next(c for c in json.loads((env.inputs / "cases.json").read_text()) if c["case_id"] == rec["case_id"])
    ct.verify_imported_record(rec, case)  # the untouched record passes
    mutate(rec)
    with pytest.raises(ct.ImportError_):
        ct.verify_imported_record(rec, case)


# ------------------------------------------------------------------------------------------------ restart-safety repairs
def _failed_case(env):
    return json.loads((env.orig_run / "terminal.json").read_text())["stopped_on"]["case_id"]


def _interrupt_at(k):
    return Fake(lambda n, cid, p: (_ for _ in ()).throw(KeyboardInterrupt) if n == k else None)


def test_reservation_is_durable_before_launch_and_a_lost_result_consumes_the_attempt(env, monkeypatch):
    failed = _failed_case(env)
    first = Fake(lambda n, cid, p: "MALFORMED" if cid == failed else None)
    append = ct.State.append

    def lose_results(self, row):  # the model answered but the result row never reached the disk
        if row.get("kind") == "result" and row["origin"] == "new" and row["case_id"] == failed:
            raise OSError("simulated disk failure after model response")
        return append(self, row)

    with monkeypatch.context() as patch:
        patch.setattr(ct.State, "append", lose_results)
        with pytest.raises(OSError):
            env.run(first)
    st = ct.State(env.work / "attempts.jsonl")
    assert [r["case_id"] for r in st.unsettled()] == [failed]
    resumed = Fake(lambda n, cid, p: "MALFORMED" if cid == failed else None)
    t = env.run(resumed)
    assert 1 + first.case_ids().count(failed) + resumed.case_ids().count(failed) <= 2  # original + lost attempt = 2: no replay, no extra retry
    assert resumed.case_ids().count(failed) == 0 and failed in t["failed_after_retry"] and t["status"] == "INCOMPLETE"
    assert t["unsettled_reservations"] == [st.unsettled()[0]["reservation_id"]]


def test_an_unsettled_reservation_counts_toward_the_call_cap(env):
    with pytest.raises(KeyboardInterrupt):
        env.run(_interrupt_at(1))
    st = ct.State(env.work / "attempts.jsonl")
    assert st.new_calls() == 1
    for _ in range(ct.MAX_NEW_MODEL_CALLS - 1):  # the rest of the cap is used up by earlier unsettled reservations
        st.append({"kind": "reserve", "case_id": "ghost", "origin": "new", "reservation_id": f"ghost#{st.new_calls() + 1}", "attempt_no": 1})
    f = Fake()
    t = env.run(f)
    assert f.judging() == [] and t["stopped_on"]["status"] == "call_cap" and t["new_model_calls_total"] == ct.MAX_NEW_MODEL_CALLS


def test_an_interrupted_launch_is_never_silently_replayed_but_the_case_keeps_its_one_remaining_attempt(env):
    with pytest.raises(KeyboardInterrupt):
        env.run(_interrupt_at(2))
    lost = ct.State(env.work / "attempts.jsonl").unsettled()[0]["case_id"]
    again = Fake()
    t = env.run(again)
    st = ct.State(env.work / "attempts.jsonl")
    assert t["unsettled_reservations"] and again.case_ids().count(lost) == 1 and st.attempts_used(lost) == 1 and st.valid(lost)
    assert t["status"] == "COMPLETE" and t["valid_judgments"] == N


def test_torn_tail_recovery_preserves_the_damaged_bytes_and_survives_further_restarts(env):
    with pytest.raises(KeyboardInterrupt):
        env.run(_interrupt_at(3))
    log = env.work / "attempts.jsonl"
    torn = b'{"case_id": "syn-torn", "origin": "new", "reco'
    log.write_bytes(log.read_bytes() + torn)
    good_rows = len(ct.State(log).rows)
    st = ct.State(log, repair=True)
    assert st.recovered and len(st.rows) == good_rows and st.torn_files == ["attempts.jsonl.torn.1"]
    assert (env.work / "attempts.jsonl.torn.1").read_bytes() == torn  # damaged bytes preserved exactly
    assert log.read_bytes().endswith(b"\n") and torn not in log.read_bytes() and not (env.work / "attempts.jsonl.tmp").exists()
    reread = ct.State(log)  # a LATER restart reads a clean journal
    assert not reread.recovered and len(reread.rows) == good_rows
    t = env.run(Fake())  # appending after recovery and reopening again still works
    assert t["status"] == "COMPLETE" and t["torn_tail_sidecar_files"] == []
    final = ct.State(log)
    assert not final.recovered and len(final.rows) > good_rows


def test_recovery_through_the_run_command_records_the_sidecar_and_a_second_tear_gets_its_own(env):
    with pytest.raises(KeyboardInterrupt):
        env.run(_interrupt_at(2))
    log = env.work / "attempts.jsonl"
    log.write_bytes(log.read_bytes() + b'{"half')
    st = ct.State(log, repair=True)
    assert st.torn_files == ["attempts.jsonl.torn.1"]
    log.write_bytes(log.read_bytes() + b'{"again')
    assert ct.State(log, repair=True).torn_files == ["attempts.jsonl.torn.2"]
    assert (env.work / "attempts.jsonl.torn.1").read_bytes() == b'{"half'


def test_a_complete_final_line_without_a_newline_is_kept_and_normalised(tmp_path):
    log = tmp_path / "attempts.jsonl"
    log.write_bytes(b'{"case_id": "a", "origin": "new", "record": {"status": "ok"}}')
    st = ct.State(log, repair=True)
    assert len(st.rows) == 1 and not st.recovered and log.read_bytes().endswith(b"\n") and st.torn_files == []


def test_read_only_state_never_rewrites_a_torn_journal(tmp_path):
    log = tmp_path / "attempts.jsonl"
    log.write_bytes(b'{"case_id": "a", "origin": "new", "record": {"status": "ok"}}\n{"tor')
    before = log.read_bytes()
    st = ct.State(log)  # compare() and other readers
    assert st.recovered and len(st.rows) == 1 and log.read_bytes() == before and not list(tmp_path.glob("*.torn.*"))


def test_import_seeding_is_atomic_if_interrupted(env, monkeypatch):
    real_replace = ct.os.replace

    def fail(*a, **k):
        raise OSError("simulated crash during seeding")

    monkeypatch.setattr(ct.os, "replace", fail)
    with pytest.raises(OSError):
        env.run()
    monkeypatch.setattr(ct.os, "replace", real_replace)
    log = env.work / "attempts.jsonl"
    assert not log.exists() or log.stat().st_size == 0  # no partially seeded journal
    t = env.run()
    imported = [r for r in ct.State(log).rows if r["origin"] == "imported"]
    assert t["status"] == "COMPLETE" and len(imported) == 5 and len({r["case_id"] for r in imported}) == 5


def test_a_partially_seeded_journal_is_completed_exactly_once(env):
    originals = ct.import_original(env.inputs, env.orig_run, env.orig_proto)
    log = env.work / "attempts.jsonl"
    log.write_text("".join(json.dumps({"case_id": r["case_id"], "origin": "imported", "attempt_note": "original run", "record": r}) + "\n" for r in originals[:2]),
                   encoding="utf8")
    t = env.run()
    imported = [r for r in ct.State(log).rows if r["origin"] == "imported"]
    assert len(imported) == len(originals) and len({r["case_id"] for r in imported}) == len(originals) and t["status"] == "COMPLETE"
    env.run()  # idempotent: a further start changes nothing
    assert len([r for r in ct.State(log).rows if r["origin"] == "imported"]) == len(originals)


def test_a_modified_imported_row_in_the_journal_is_refused(env):
    with pytest.raises(KeyboardInterrupt):
        env.run(_interrupt_at(1))
    log = env.work / "attempts.jsonl"
    lines = log.read_text().splitlines()
    row = json.loads(lines[0])
    row["record"]["raw_response"] += " tampered"
    lines[0] = json.dumps(row)
    log.write_text("\n".join(lines) + "\n", encoding="utf8")
    with pytest.raises(ct.ImportError_):
        env.run(Fake())


def test_the_superseded_continuation_protocol_is_recorded_and_untouched():
    old = ct.ROOT / "continuation" / "protocol.json"
    if old.exists():
        assert ct._sha(old) == ct.SUPERSEDED_CONTINUATION_SHA256
    assert ct.WORK != ct.ROOT / "continuation"


def test_appending_through_a_read_only_state_restores_the_journal_first(tmp_path):
    log = tmp_path / "attempts.jsonl"
    row = {"case_id": "one", "origin": "new", "record": {"status": "ok"}}
    log.write_bytes((json.dumps(row) + "\n").encode() + b'{"case_id": "torn')
    recovered = ct.State(log)
    recovered.append({"case_id": "two", "origin": "new", "record": {"status": "ok"}})
    reopened = ct.State(log)
    assert len(reopened.rows) == 2 and not reopened.recovered and (tmp_path / "attempts.jsonl.torn.1").read_bytes() == b'{"case_id": "torn'


def test_journal_restore_is_atomic_a_crash_before_the_swap_leaves_the_original_bytes_and_a_retry_succeeds(tmp_path, monkeypatch):
    log = tmp_path / "attempts.jsonl"
    row = {"case_id": "one", "origin": "new", "record": {"status": "ok"}}
    original = (json.dumps(row) + "\n").encode() + b'{"case_id": "torn'
    log.write_bytes(original)
    real = ct.os.replace
    monkeypatch.setattr(ct.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("crash before swap")))
    with pytest.raises(OSError):
        ct.State(log, repair=True)
    assert log.read_bytes() == original  # the journal itself was never half-written
    monkeypatch.setattr(ct.os, "replace", real)
    st = ct.State(log, repair=True)
    assert len(st.rows) == 1 and not ct.State(log).recovered and not (tmp_path / "attempts.jsonl.tmp").exists()
    assert all((tmp_path / n).read_bytes() == b'{"case_id": "torn' for n in (p.name for p in tmp_path.glob("attempts.jsonl.torn.*")))  # repeated sidecars hold identical bytes

"""Tests for the Claude-subscription claim-relation experiment. The CLI is replaced by a fake runner: no model call, no subscription use."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from evalops import claim_relation as v1
from evalops import claim_relation_claude as cc

FIXTURES = v1.synthetic_fixtures()
BY_PROMPT = {v1.sha256(v1.build_prompt(f["source"], f["candidate"])): f for f in FIXTURES}
GOOD_AUTH = {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": "pro"}


def make_inputs(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    files = {"cases.json": [{"case_id": f["case_id"], "source": f["source"], "candidate": f["candidate"]} for f in FIXTURES],
             "labels.json": {f["case_id"]: f["label"] for f in FIXTURES}, "groups.json": {f["case_id"]: f["document_family"] for f in FIXTURES},
             "baseline_v18.json": {f["case_id"]: 1 for f in FIXTURES}, "splits.json": {f["case_id"]: "dev" for f in FIXTURES}}
    for n, v in files.items():
        (root / n).write_text(json.dumps(v, sort_keys=True), encoding="utf8")
    return root


class FakeCli:
    """Stands in for subprocess.run: answers `auth status`, `--version` and judging calls; records every call."""

    def __init__(self, auth=None, judge=None):
        self.calls, self.auth, self.judge = [], auth or GOOD_AUTH, judge or self.default_judge

    @staticmethod
    def default_judge(n, prompt):
        f = BY_PROMPT[v1.sha256(prompt)]
        return {"is_error": False, "subtype": "success", "result": v1.scripted_response(f), "modelUsage": {cc.MODEL: {}}, "total_cost_usd": 0.01, "num_turns": 1,
                "usage": {"input_tokens": 600, "output_tokens": 300, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, "session_id": f"s{n}"}

    def __call__(self, args, **kw):
        self.calls.append((list(args), kw))
        if args[1:3] == ["auth", "status"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.auth), stderr="")
        if args[1:] == ["--version"]:
            return SimpleNamespace(returncode=0, stdout="9.9.9 (Claude Code)\n", stderr="")
        n = sum(1 for a, _ in self.calls if "-p" in a)
        out = self.judge(n, kw["input"])
        return SimpleNamespace(returncode=0, stdout=out if isinstance(out, str) else json.dumps(out), stderr="")

    def judging(self):
        return [c for c in self.calls if "-p" in c[0]]


@pytest.fixture(autouse=True)
def _cli_and_env(monkeypatch):
    monkeypatch.setattr(cc.shutil, "which", lambda name: "claude")
    for k in cc.FORBIDDEN_ENV:
        monkeypatch.delenv(k, raising=False)


def frozen(tmp_path, fake):
    inputs = make_inputs(tmp_path / "in")
    proto = tmp_path / "protocol.json"
    cc.freeze(inputs, proto, fake)
    return inputs, proto, cc._sha_file(proto)


# ---------------------------------------------------------------------------------------------------- authentication
@pytest.mark.parametrize("auth", [{**GOOD_AUTH, "loggedIn": False}, {**GOOD_AUTH, "authMethod": "api_key"}, {**GOOD_AUTH, "apiProvider": "bedrock"},
                                  {**GOOD_AUTH, "subscriptionType": None}])
def test_non_subscription_authentication_is_refused_before_any_judging_call(tmp_path, auth):
    good = FakeCli()
    inputs, proto, sha = frozen(tmp_path, good)
    bad = FakeCli(auth=auth)
    with pytest.raises(cc.AuthError):
        cc.run(inputs, proto, tmp_path / "run", cc.make_request(bad), expected_sha256=sha, runner=bad)
    assert bad.judging() == []


@pytest.mark.parametrize("var", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK"])
def test_metered_api_environment_variables_block_the_run_and_are_stripped_from_the_child(tmp_path, monkeypatch, var):
    fake = FakeCli()
    frozen(tmp_path, fake)
    monkeypatch.setenv(var, "x")
    with pytest.raises(cc.AuthError, match="environment variables"):
        cc.verify_subscription(fake)
    assert var not in cc.clean_env()


def test_the_real_auth_status_shape_is_accepted():
    assert cc.verify_subscription(FakeCli()) == {"authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": "pro"}


# ---------------------------------------------------------------------------------------------------- request shape
def test_the_judge_receives_only_the_prompt_with_no_tools_context_or_labels(tmp_path):
    fake = FakeCli()
    inputs, proto, sha = frozen(tmp_path, fake)
    cc.run(inputs, proto, tmp_path / "run", cc.make_request(fake, tmp_path / "cwd"), expected_sha256=sha, runner=fake)
    args, kw = fake.judging()[0]
    assert args[1:1 + len(cc.CLI_FLAGS)] == cc.CLI_FLAGS and "--tools" in args and args[args.index("--tools") + 1] == ""
    assert {"--no-session-persistence", "--disable-slash-commands", "--strict-mcp-config"} <= set(args) and "--bare" not in args
    assert kw["cwd"] == str(tmp_path / "cwd") and list((tmp_path / "cwd").iterdir()) == []  # an empty directory: nothing to read
    prompt = kw["input"]
    assert prompt == v1.build_prompt(FIXTURES[0]["source"], FIXTURES[0]["candidate"])
    assert not any(w in prompt.lower() for w in ("gold", "v18", "label", "expected_verdict", "baseline"))
    assert not any(k in kw["env"] for k in cc.FORBIDDEN_ENV)


def test_labels_groups_baseline_and_splits_are_not_parsed_during_the_run(tmp_path, monkeypatch):
    fake = FakeCli()
    inputs, proto, sha = frozen(tmp_path, fake)
    real_read, real_loads = Path.read_text, json.loads
    seen = []

    def guard(self, *a, **k):
        if self.name in cc.JOINS:
            seen.append(self.name)
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", guard)
    cc.run(inputs, proto, tmp_path / "run", cc.make_request(fake), expected_sha256=sha, runner=fake)
    assert seen == []  # only raw bytes were hashed (read_bytes), never parsed
    monkeypatch.setattr(Path, "read_text", real_read)
    assert real_loads("{}") == {}
    result = cc.compare(inputs, tmp_path / "run", proto, expected_sha256=sha, runner=fake)  # labels are parsed only now
    assert result["status"] == "COMPLETE"


# ---------------------------------------------------------------------------------------------------- complete run and records
def test_a_complete_run_records_the_actual_model_coverage_and_dev_comparison(tmp_path):
    fake = FakeCli()
    inputs, proto, sha = frozen(tmp_path, fake)
    t = cc.run(inputs, proto, tmp_path / "run", cc.make_request(fake), expected_sha256=sha, runner=fake)
    assert t["status"] == "COMPLETE" and t["ok"] == t["cases"] == 19 and t["models_reported"] == [cc.MODEL] and t["auth"]["authMethod"] == "claude.ai"
    assert t["token_totals"]["input_tokens"] == 19 * 600 and t["no_automatic_retry"] and len(fake.judging()) == 19
    rec = json.loads((tmp_path / "run" / "judgments.jsonl").read_text().splitlines()[0])
    assert rec["model"] == cc.MODEL and rec["usage"]["session_id"] and rec["raw_response"] and rec["variant"] == "claim-relation-v2"
    out = cc.compare(inputs, tmp_path / "run", proto, expected_sha256=sha, runner=fake)
    assert out["dev_primary"]["n"] == 19 and out["train_diagnostic_only"] is None and out["run"]["coverage"] == "19/19"
    assert out["run"]["models_reported"] == [cc.MODEL] and out["gate"]["criterion"].startswith("dev errors fixed")
    with pytest.raises(v1.ClaimRelationError, match="written once"):
        cc.compare(inputs, tmp_path / "run", proto, expected_sha256=sha, runner=fake)


# ---------------------------------------------------------------------------------------------------- fail-fast
@pytest.mark.parametrize("label,bad", [
    ("cli error", {"is_error": True, "subtype": "error_during_execution", "result": "rate limit", "modelUsage": {cc.MODEL: {}}}),
    ("rate limit text", {"is_error": True, "subtype": "success", "result": "usage limit reached", "api_error_status": 429, "modelUsage": {}}),
    ("model mismatch", {"is_error": False, "subtype": "success", "result": "{}", "modelUsage": {"claude-haiku-4-5": {}}}),
    ("two models", {"is_error": False, "subtype": "success", "result": "{}", "modelUsage": {cc.MODEL: {}, "claude-haiku-4-5": {}}}),
    ("non-json stdout", "Error: not logged in"),
])
def test_first_anomaly_stops_with_zero_further_calls_and_withholds_comparison(tmp_path, label, bad):
    fake = FakeCli(judge=lambda n, p: bad if n == 2 else FakeCli.default_judge(n, p))
    inputs, proto, sha = frozen(tmp_path, fake)
    t = cc.run(inputs, proto, tmp_path / "run", cc.make_request(fake), expected_sha256=sha, runner=fake)
    assert t["status"] == "ABORTED" and len(fake.judging()) == 2 and t["attempted"] == 2 and len(t["not_attempted"]) == 17, label
    assert cc.compare(inputs, tmp_path / "run", proto, expected_sha256=sha, runner=fake)["status"] == "WITHHELD_INCOMPLETE"


def test_malformed_model_output_is_incomplete_not_guessed_and_stops(tmp_path):
    def judge(n, p):
        return {**FakeCli.default_judge(n, p), "result": "I think it is supported."} if n == 3 else FakeCli.default_judge(n, p)

    fake = FakeCli(judge=judge)
    inputs, proto, sha = frozen(tmp_path, fake)
    t = cc.run(inputs, proto, tmp_path / "run", cc.make_request(fake), expected_sha256=sha, runner=fake)
    assert t["status"] == "INCOMPLETE" and len(fake.judging()) == 3 and t["stopped_on"]["status"] == "invalid_structure"


def test_timeout_aborts_without_retry(tmp_path):
    def judge(n, p):
        raise subprocess.TimeoutExpired("claude", 1)

    fake = FakeCli(judge=judge)
    inputs, proto, sha = frozen(tmp_path, fake)
    t = cc.run(inputs, proto, tmp_path / "run", cc.make_request(fake), expected_sha256=sha, runner=fake)
    assert t["status"] == "ABORTED" and len(fake.judging()) == 1 and "timeout" in t["stopped_on"]["error"]


# ---------------------------------------------------------------------------------------------------- freeze integrity
@pytest.mark.parametrize("name,mutate", [("cases.json", lambda d: d[0].update(candidate=d[0]["candidate"] + " x")), ("labels.json", lambda d: d.update({"syn-01": 1 - d["syn-01"]})),
                                         ("groups.json", lambda d: d.update({"syn-01": "other"})), ("baseline_v18.json", lambda d: d.update({"syn-01": 0})),
                                         ("splits.json", lambda d: d.update({"syn-01": "train"}))])
def test_altered_inputs_after_the_freeze_are_refused_before_any_call(tmp_path, name, mutate):
    fake = FakeCli()
    inputs, proto, sha = frozen(tmp_path, fake)
    d = json.loads((inputs / name).read_text())
    mutate(d)
    (inputs / name).write_text(json.dumps(d, sort_keys=True), encoding="utf8")
    with pytest.raises(v1.ProtocolError):
        cc.run(inputs, proto, tmp_path / "run", cc.make_request(fake), expected_sha256=sha, runner=fake)
    assert fake.judging() == []


def test_changed_cli_version_model_flags_or_manifest_are_refused(tmp_path, monkeypatch):
    fake = FakeCli()
    inputs, proto, sha = frozen(tmp_path, fake)
    other = FakeCli()
    other.__class__ = type("Newer", (FakeCli,), {"__call__": lambda s, args, **kw: SimpleNamespace(returncode=0, stdout="10.0.0\n", stderr="") if args[1:] == ["--version"]
                                                  else FakeCli.__call__(s, args, **kw)})
    with pytest.raises(v1.ProtocolError, match="backend"):
        cc.verify(inputs, proto, sha, other)
    monkeypatch.setattr(cc, "EFFORT", "high")
    monkeypatch.setattr(cc, "CLI_FLAGS", [*cc.CLI_FLAGS])
    with pytest.raises(v1.ProtocolError):
        cc.verify(inputs, proto, sha, fake)
    monkeypatch.undo()
    with pytest.raises(v1.ProtocolError, match="freeze time"):
        cc.verify(inputs, proto, "0" * 64, fake)


def test_dev_only_and_label_free_cases_are_enforced_and_openai_protocol_is_untouched(tmp_path):
    fake = FakeCli()
    inputs = make_inputs(tmp_path / "in")
    s = json.loads((inputs / "splits.json").read_text())
    s["syn-01"] = "train"
    (inputs / "splits.json").write_text(json.dumps(s))
    with pytest.raises(v1.ProtocolError, match="dev-only"):
        cc.freeze(inputs, tmp_path / "p.json", fake)
    c = json.loads((inputs / "cases.json").read_text())
    c[0]["label"] = 1
    (inputs / "cases.json").write_text(json.dumps(c))
    with pytest.raises(v1.ProtocolError, match="exactly case_id"):
        cc._check_inputs(inputs, parse_joins=False)
    frozen_openai = Path(cc.WORK).parent / "claim-relation-ablation-v1" / "protocol.json"
    if frozen_openai.exists():
        assert cc._sha_file(frozen_openai) == "8c32146ae66f0cccd5d42c429b080efbc766f23809c545f7f9a64582d40323ce"


def test_run_command_requires_confirmation_and_the_freeze_hash(tmp_path, capsys):
    assert cc.main(["run", "--inputs", str(tmp_path), "--protocol", str(tmp_path / "p.json")]) == 2
    assert "--protocol-sha256" in capsys.readouterr().err
    assert cc.main(["run", "--inputs", str(tmp_path), "--protocol", str(tmp_path / "p.json"), "--protocol-sha256", "0" * 64]) == 2
    assert "--confirm" in capsys.readouterr().err

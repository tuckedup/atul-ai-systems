"""Claude-backed claim-relation experiment (`claim-relation-v2` post-processing, Claude Code subscription backend). Separate from the frozen OpenAI protocol.

The judge sees ONLY the fixed prompt (policy + source + candidate) through `claude -p` with no tools, no project context, no skills, no MCP and a fixed minimal
system prompt. Reference labels, groups, v18 verdicts and splits live in the same input directory but are hashed as raw bytes during the run and parsed only by
`compare`, after predictions are saved. Authentication is verified before execution: it must be a claude.ai subscription login, with no API-key,
auth-token, proxy or cloud-provider variables present (those are also stripped from the child environment). Usage is notional list-price cost, not an invoice;
subscription rate limits can stop a run, which then ends ABORTED with no further call. Actual model identity comes from the CLI's own `modelUsage` report and
must equal the frozen model.

    python -m evalops.claim_relation_claude (see main): build-inputs -> freeze -> auth -> run -> compare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import claim_relation as v1
from . import claim_relation_v2 as v2

MODEL = "claude-opus-5-5"
EFFORT = "medium"
TIMEOUT_S = 900
SYSTEM_PROMPT = "You are a careful grader. Reply with only the requested JSON."
CLI_FLAGS = ["-p", "--model", MODEL, "--effort", EFFORT, "--output-format", "json", "--tools", "", "--no-session-persistence", "--setting-sources", "project",
             "--disable-slash-commands", "--strict-mcp-config", "--exclude-dynamic-system-prompt-sections", "--system-prompt", SYSTEM_PROMPT]
FORBIDDEN_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_BEDROCK_BASE_URL", "ANTHROPIC_VERTEX_PROJECT_ID",
                 "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_OAUTH_TOKEN")
INPUTS = ("cases.json", "labels.json", "groups.json", "baseline_v18.json", "splits.json")
JOINS = INPUTS[1:]
CONFIRM = "RUN-CLAUDE-SUBSCRIPTION"
WORK = Path(__file__).resolve().parents[2] / ".local" / "claim-relation-claude-v1"


class AuthError(v1.ClaimRelationError):
    pass


def _exe() -> str:
    exe = shutil.which("claude")
    if exe is None:
        raise AuthError("claude CLI not found on PATH")
    return exe


def clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in FORBIDDEN_ENV}


def verify_subscription(runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Raise AuthError unless the CLI is logged in with a claude.ai subscription and nothing could route the call to a metered API."""
    present = [k for k in FORBIDDEN_ENV if os.environ.get(k)]
    if present:
        raise AuthError(f"metered-API environment variables are set: {present}")
    r = runner([_exe(), "auth", "status"], capture_output=True, text=True, timeout=60, encoding="utf8", env=clean_env())
    try:
        status = json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise AuthError("could not read `claude auth status`") from exc
    if not (status.get("loggedIn") is True and status.get("authMethod") == "claude.ai" and status.get("apiProvider") == "firstParty"
            and status.get("subscriptionType") not in (None, "", "none")):
        raise AuthError(f"not a claude.ai subscription login: loggedIn={status.get('loggedIn')} authMethod={status.get('authMethod')} "
                        f"provider={status.get('apiProvider')} subscription={status.get('subscriptionType')}")
    return {"authMethod": status["authMethod"], "apiProvider": status["apiProvider"], "subscriptionType": status["subscriptionType"]}


def cli_version(runner: Callable[..., Any] = subprocess.run) -> str:
    return runner([_exe(), "--version"], capture_output=True, text=True, timeout=60, encoding="utf8", env=clean_env()).stdout.strip()


def make_request(runner: Callable[..., Any] = subprocess.run, cwd: Path | None = None) -> v1.Request:
    """One `claude -p` call per case. ANY failure raises RunAbort so the runner stops with no further call (no retries)."""
    workdir = Path(cwd) if cwd else Path(tempfile.gettempdir()) / "cc-judge-cwd"
    workdir.mkdir(parents=True, exist_ok=True)  # an EMPTY directory: no project files, CLAUDE.md or memory can reach the judge

    def request(prompt: str) -> v1.Response:
        started = time.perf_counter()
        try:
            r = runner([_exe(), *CLI_FLAGS], input=prompt, capture_output=True, text=True, timeout=TIMEOUT_S, encoding="utf8", cwd=str(workdir), env=clean_env())
        except subprocess.TimeoutExpired as exc:
            raise v1.RunAbort(f"timeout after {TIMEOUT_S}s") from exc
        try:
            d = json.loads(r.stdout)
        except json.JSONDecodeError as exc:
            raise v1.RunAbort(f"unreadable CLI output (rc={r.returncode}): {(r.stderr or r.stdout)[:200]!r}") from exc
        if r.returncode != 0 or d.get("is_error") or d.get("subtype") != "success" or not isinstance(d.get("result"), str):
            raise v1.RunAbort(f"CLI error (rc={r.returncode}, subtype={d.get('subtype')}, status={d.get('api_error_status')}): {str(d.get('result'))[:200]!r}")
        models = d.get("modelUsage") or {}
        if set(models) != {MODEL}:
            raise v1.RunAbort(f"model mismatch: CLI reported {sorted(models)}, frozen {MODEL!r}")
        u = d.get("usage") or {}
        usage = {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"), "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
                 "cache_read_input_tokens": u.get("cache_read_input_tokens"), "notional_list_cost_usd": d.get("total_cost_usd"), "latency_s": round(time.perf_counter() - started, 2),
                 "num_turns": d.get("num_turns"), "session_id": d.get("session_id")}
        return v1.Response(text=d["result"], usage=usage, model=MODEL)

    return request


# ---------------------------------------------------------------------------------------------------------------- inputs / freeze
def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_inputs(source_inputs: Path, out_dir: Path) -> dict[str, Any]:
    """Dev-only copy of the frozen OpenAI-experiment inputs (128 cases). Nothing is dropped from dev; TRAIN is excluded by design."""
    src = {n: json.loads((Path(source_inputs) / n).read_text(encoding="utf8")) for n in INPUTS}
    dev = {i for i, s in src["splits.json"].items() if s == "dev"}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [c for c in src["cases.json"] if c["case_id"] in dev]
    data = {"cases.json": cases} | {n: {i: v for i, v in src[n].items() if i in dev} for n in JOINS}
    for name, value in data.items():
        target = out_dir / name
        if target.exists():
            raise v1.ClaimRelationError(f"{name} exists; refusing to overwrite")
        target.write_text(json.dumps(value, sort_keys=True), encoding="utf8")
    return {"cases": len(cases), "source_input_sha256": {n: _sha_file(Path(source_inputs) / n) for n in INPUTS}}


def _check_inputs(inputs_dir: Path, *, parse_joins: bool) -> dict[str, Any]:
    """Structure checks. With parse_joins=False the join files (labels, groups, baseline, splits) are NOT parsed: only their bytes are hashed elsewhere."""
    cases = json.loads((Path(inputs_dir) / "cases.json").read_text(encoding="utf8"))
    ids = [c.get("case_id") for c in cases] if isinstance(cases, list) else []
    if not isinstance(cases, list) or len(set(ids)) != len(ids) or any(not isinstance(i, str) for i in ids):
        raise v1.ProtocolError("cases.json must be a list with unique string case ids")
    if any(set(c) != {"case_id", "source", "candidate"} for c in cases):
        raise v1.ProtocolError("cases.json rows must contain exactly case_id, source, candidate")
    out: dict[str, Any] = {"cases": cases}
    if parse_joins:
        for n in JOINS:
            out[n] = json.loads((Path(inputs_dir) / n).read_text(encoding="utf8"))
            if not isinstance(out[n], dict) or set(out[n]) != set(ids):
                raise v1.ProtocolError(f"{n} case ids differ from cases.json (missing or extra records)")
        if set(out["splits.json"].values()) != {"dev"}:
            raise v1.ProtocolError("this experiment is dev-only: splits.json must contain only 'dev'")
        if set(out["labels.json"].values()) - {0, 1} or set(out["baseline_v18.json"].values()) - {0, 1}:
            raise v1.ProtocolError("labels and baseline must be 0/1")
    return out


def _body(inputs_dir: Path, version: str) -> dict[str, Any]:
    d = Path(inputs_dir)
    cases = _check_inputs(d, parse_joins=False)["cases"]
    ident = v1.check_v18_identity()
    here = Path(__file__)
    return {
        "protocol": "claim-relation-claude-dev-v1", "input_sha256": {n: _sha_file(d / n) for n in INPUTS}, "population": {"cases": len(cases), "split": "dev"},
        "candidate": {"variant": v2.VARIANT_ID, "content_hash": v2.content_hash(), "claim_relation_v2_sha256": _sha_file(Path(v2.__file__)),
                      "claim_relation_prompt_module_sha256": _sha_file(Path(v1.__file__)), "backend_sha256": _sha_file(here)},
        "v18_identity": {k: ident[k] for k in ("v18_config_hash", "v18_prompt_template_hash")} | {"unchanged": ident["unchanged"]},
        "backend": {"kind": "claude -p (subscription)", "model": MODEL, "effort": EFFORT, "cli_flags": CLI_FLAGS, "cli_version": version,
                    "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), "timeout_s": TIMEOUT_S, "forbidden_env": list(FORBIDDEN_ENV)},
        "judge_input": "fixed prompt = policy + source + candidate only; labels/groups/baseline/splits are never read before predictions are saved",
        "decision_rule": dict(v1.FROZEN_RULE), "bootstrap": {"seed": v1.BOOTSTRAP_SEED, "draws": v1.BOOTSTRAP_DRAWS},
        "gate": "dev errors fixed > new dev errors AND dev 95% document-grouped lower bound of kappa gain > 0", "attempts_per_case": 1, "fail_fast": True,
        "api_spend_usd": 0, "cost_note": "subscription allowance; usage recorded as notional list-price cost",
    }


def freeze(inputs_dir: Path, out_path: Path, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    out_path = Path(out_path)
    if out_path.exists():
        raise v1.ClaimRelationError("protocol exists; a frozen protocol is never overwritten")
    body = _body(inputs_dir, cli_version(runner))
    if not body["v18_identity"]["unchanged"]:
        raise v1.ProtocolError("v18 identity differs from the frozen values")
    _check_inputs(inputs_dir, parse_joins=True)  # full consistency once, at freeze
    out_path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf8")
    return body


def verify(inputs_dir: Path, protocol_path: Path, expected_sha256: str | None, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    if expected_sha256 is not None and _sha_file(protocol_path) != expected_sha256:
        raise v1.ProtocolError("protocol file hash differs from the hash recorded at freeze time")
    frozen = json.loads(Path(protocol_path).read_text(encoding="utf8"))
    current = _body(inputs_dir, cli_version(runner))
    for key, value in current.items():
        if frozen.get(key) != value:
            raise v1.ProtocolError(f"frozen protocol differs from the current state at {key!r}")
    return frozen


# ---------------------------------------------------------------------------------------------------------------- run / compare
def run(inputs_dir: Path, protocol_path: Path, out_dir: Path, request: v1.Request, *, expected_sha256: str, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Verify freeze + subscription, then judge each case once in order; stop at the first anomaly. Labels/groups/baseline are not parsed here."""
    verify(inputs_dir, protocol_path, expected_sha256, runner)
    auth = verify_subscription(runner)
    cases = _check_inputs(inputs_dir, parse_joins=False)["cases"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "judgments.jsonl"
    if path.exists():
        raise v1.ClaimRelationError("judgments.jsonl exists; refusing to overwrite (runs are not silently repeated)")
    records: list[dict[str, Any]] = []
    stop = None
    with path.open("w", encoding="utf8") as fh:
        for case in cases:
            rec = v2.judge_case(case, request)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(rec)
            if rec["status"] != "ok":
                stop = {"case_id": rec["case_id"], "status": rec["status"], "error": rec.get("error")}
                break
    ok = sum(r["status"] == "ok" for r in records)
    u = [r["usage"] or {} for r in records]
    terminal = {"status": "COMPLETE" if stop is None else ("INCOMPLETE" if stop["status"] == "invalid_structure" else "ABORTED"), "cases": len(cases),
                "attempted": len(records), "ok": ok, "not_attempted": [c["case_id"] for c in cases[len(records):]], "stopped_on": stop,
                "verdicts": {v: sum(1 for r in records if r["status"] == "ok" and r["structured"]["verdict"] == v) for v in v1.VERDICTS},
                "models_reported": sorted({r["model"] for r in records if r["model"]}), "frozen_model": MODEL, "auth": auth,
                "token_totals": {k: sum((x.get(k) or 0) for x in u) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")},
                "notional_list_cost_usd": round(sum((x.get("notional_list_cost_usd") or 0) for x in u), 4), "latency_s_total": round(sum((x.get("latency_s") or 0) for x in u), 1),
                "variant": v2.VARIANT_ID, "content_hash": v2.content_hash(), "protocol_sha256": _sha_file(protocol_path), "no_automatic_retry": True}
    (out_dir / "terminal.json").write_text(json.dumps(terminal, indent=2), encoding="utf8")
    return terminal


def compare(inputs_dir: Path, run_dir: Path, protocol_path: Path, *, expected_sha256: str, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    frozen = verify(inputs_dir, protocol_path, expected_sha256, runner)
    run_dir = Path(run_dir)
    terminal = json.loads((run_dir / "terminal.json").read_text(encoding="utf8"))
    if terminal.get("protocol_sha256") != _sha_file(protocol_path) or terminal.get("content_hash") != v2.content_hash():
        raise v1.ProtocolError("the run was not produced under this frozen protocol and candidate code")
    target = run_dir / "comparison.json"
    if target.exists():
        raise v1.ClaimRelationError("comparison.json exists; the comparison is written once")
    records = [json.loads(x) for x in (run_dir / "judgments.jsonl").read_text(encoding="utf8").splitlines() if x.strip()]
    data = _check_inputs(inputs_dir, parse_joins=True)  # labels are parsed here for the first time, after predictions were saved
    by_case = {c["case_id"]: c for c in data["cases"]}
    for r in records:
        c = by_case.get(r["case_id"])
        if c is None or r["input"]["source_sha256"] != v1.sha256(c["source"]) or r["input"]["candidate_sha256"] != v1.sha256(c["candidate"]):
            raise v1.ProtocolError(f"run record {r['case_id']} does not match the frozen inputs (extra, altered or duplicated record)")
    result = v1.compare_to_baseline(terminal, records, data["labels.json"], data["baseline_v18.json"], data["groups.json"], data["splits.json"],
                                    uncertain_as=frozen["decision_rule"]["uncertain_as"], draws=frozen["bootstrap"]["draws"], seed=frozen["bootstrap"]["seed"])
    result["run"] = {"terminal_status": terminal["status"], "models_reported": terminal["models_reported"], "coverage": f"{terminal['ok']}/{terminal['cases']}",
                     "failures": terminal["stopped_on"], "notional_list_cost_usd": terminal["notional_list_cost_usd"], "token_totals": terminal["token_totals"]}
    target.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["build-inputs", "freeze", "auth", "run", "compare"])
    p.add_argument("--source-inputs", type=Path, default=Path(__file__).resolve().parents[2] / ".local" / "claim-relation-ablation-v1" / "inputs")
    p.add_argument("--inputs", type=Path, default=WORK / "inputs")
    p.add_argument("--protocol", type=Path, default=WORK / "protocol.json")
    p.add_argument("--out", type=Path, default=WORK / "run")
    p.add_argument("--protocol-sha256")
    p.add_argument("--confirm", default="")
    a = p.parse_args(argv)
    try:
        if a.command == "build-inputs":
            print(json.dumps(build_inputs(a.source_inputs, a.inputs), indent=2))
        elif a.command == "freeze":
            body = freeze(a.inputs, a.protocol)
            print(json.dumps({"protocol_sha256": _sha_file(a.protocol), **body}, indent=2))
        elif a.command == "auth":
            print(json.dumps(verify_subscription(), indent=2))
        else:
            if not a.protocol_sha256:
                print("REFUSED: --protocol-sha256 (printed at freeze) is required", file=sys.stderr)
                return 2
            if a.command == "run":
                if a.confirm != CONFIRM:
                    print(f"REFUSED: needs --confirm {CONFIRM}", file=sys.stderr)
                    return 2
                t = run(a.inputs, a.protocol, a.out, make_request(), expected_sha256=a.protocol_sha256)
                print(json.dumps(t, indent=2))
                return 0 if t["status"] == "COMPLETE" else 3
            print(json.dumps(compare(a.inputs, a.out, a.protocol, expected_sha256=a.protocol_sha256), indent=2))
    except v1.ClaimRelationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

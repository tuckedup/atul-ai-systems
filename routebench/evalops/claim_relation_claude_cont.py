"""Frozen, resumable CONTINUATION of the Claude-subscription claim-relation dev evaluation (protocol `claim-relation-claude-dev-v2-continuation`).

The original run (`.local/claim-relation-claude-v1/run`, 81 valid judgments, stopped INCOMPLETE on one malformed response) is preserved byte-for-byte and only READ.

Uniform rule for every case: at most TWO attempts for malformed structured output (one retry), same prompt, model and settings; every attempt is recorded and the
retry decision never sees a gold label. The 81 valid original judgments are imported only after their input, prompt, judge-code, model and parsed-structure
identities are re-verified; the one original malformed attempt counts as that case's first attempt. A case that is malformed twice is RETAINED as failed (never
dropped, no verdict invented) and the result is INCOMPLETE. Every attempt is RESERVED in the fsynced journal before the model is launched; a reservation with no
recorded outcome (crash, kill, lost write) still counts toward the call cap and toward that case's two attempts, and is never silently replayed. Request errors (CLI error, rate limit, timeout, model mismatch) are not malformed output: they stop the
run (ABORTED, resumable) and do not consume the retry. A kappa is published only when 128/128 cases hold a valid judgment, and the completeness check runs
BEFORE any label, group, baseline or split file is parsed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import claim_relation as v1
from . import claim_relation_claude as cc
from . import claim_relation_v2 as v2

PROTOCOL_NAME = "claim-relation-claude-dev-v3-restart-safe-continuation"
SUPERSEDED_CONTINUATION_SHA256 = "2ae67ff5a99690e6171d754e5e7c1daf50ca5249604efcee13b59fc4701a6491"  # frozen, never run; kept unchanged
MAX_ATTEMPTS_MALFORMED = 2  # first attempt + one retry
MAX_NEW_MODEL_CALLS = 100  # hard bound on model calls made under this protocol (47 pending cases x 2 attempts = 94, plus slack for request errors)
CONFIRM = "RUN-CLAUDE-CONTINUATION"
ROOT = Path(__file__).resolve().parents[2] / ".local" / "claim-relation-claude-v1"
ORIGINAL_RUN = ROOT / "run"
ORIGINAL_PROTOCOL = ROOT / "protocol.json"
ORIGINAL_PROTOCOL_SHA = "830f6a037d293f6c2c61589785a80adbb908d57d3e824e72d2c6b70e51a2a2bb"
WORK = ROOT / "continuation-v3"
INPUTS = ROOT / "inputs"


class ImportError_(v1.ProtocolError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _digest(ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------ import verification
def verify_imported_record(rec: dict[str, Any], case: dict[str, Any]) -> None:
    """Raise ImportError_ unless the stored record was produced for exactly this input by this judge, and its stored structure re-derives from its raw response."""
    prompt = v1.build_prompt(case["source"], case["candidate"], "")
    checks = {
        "variant": rec.get("variant") == v2.VARIANT_ID, "judge code (content hash)": rec.get("content_hash") == v2.content_hash(),
        "prompt": rec.get("prompt") == prompt and rec.get("prompt_sha256") == v1.sha256(prompt),
        "input source": rec.get("input", {}).get("source") == case["source"] and rec["input"].get("source_sha256") == v1.sha256(case["source"]),
        "input candidate": rec.get("input", {}).get("candidate") == case["candidate"] and rec["input"].get("candidate_sha256") == v1.sha256(case["candidate"]),
        "model": rec.get("model") == cc.MODEL, "raw response": isinstance(rec.get("raw_response"), str) and bool(rec["raw_response"]),
        "usage/session": isinstance(rec.get("usage"), dict) and bool(rec["usage"].get("session_id")),
        "status": rec.get("status") in ("ok", "invalid_structure"),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ImportError_(f"{case['case_id']}: identity check failed for {bad}")
    again = v2.reprocess_v1_record(rec)
    if again["status"] != rec["status"] or again.get("structured") != rec.get("structured") or again.get("error") != rec.get("error"):
        raise ImportError_(f"{case['case_id']}: stored structure does not re-derive from the stored raw response")


def import_original(inputs_dir: Path, original_run: Path, original_protocol: Path) -> list[dict[str, Any]]:
    """Verify and return the original run's records (valid ones and the one malformed attempt). The original files are only read."""
    if _sha(original_protocol) != ORIGINAL_PROTOCOL_SHA:
        raise ImportError_("original protocol file differs from the recorded hash")
    term = json.loads((Path(original_run) / "terminal.json").read_text(encoding="utf8"))
    if (term.get("status"), term.get("protocol_sha256"), term.get("content_hash")) != ("INCOMPLETE", ORIGINAL_PROTOCOL_SHA, v2.content_hash()):
        raise ImportError_("original terminal record does not match the preserved incomplete run under the original protocol and judge code")
    orig = json.loads(original_protocol.read_text(encoding="utf8"))
    for name, digest in orig["input_sha256"].items():
        if _sha(Path(inputs_dir) / name) != digest:
            raise ImportError_(f"input {name} differs from the original protocol")
    cases = {c["case_id"]: c for c in cc._check_inputs(inputs_dir, parse_joins=False)["cases"]}
    records = [json.loads(x) for x in (Path(original_run) / "judgments.jsonl").read_text(encoding="utf8").splitlines() if x.strip()]
    seen: set[str] = set()
    for r in records:
        cid = r.get("case_id")
        if cid not in cases or cid in seen:
            raise ImportError_(f"original record for {cid!r} is unknown or duplicated")
        seen.add(cid)
        verify_imported_record(r, cases[cid])
    if len(records) != term["attempted"] or sum(r["status"] == "ok" for r in records) != term["ok"]:
        raise ImportError_("original record counts differ from its terminal record")
    return records


# ------------------------------------------------------------------------------------------------------ state
class State:
    """Per-case state rebuilt from the append-only journal (restart recovery).

    Row kinds: `reserve` (written and fsynced BEFORE a model launch), `result` (the attempt's outcome, carrying the same `reservation_id`; imported rows have none).
    A reservation without a result is UNSETTLED: it counts as a model call and as a consumed attempt for its case, and is never replayed.
    A torn final line (crash mid-write) is, with repair=True, preserved byte-for-byte in a sidecar file and the journal is restored atomically (temp file + fsync +
    os.replace) so later restarts read a clean file; with repair=False (read-only callers) it is tolerated but the file is left alone.
    """

    def __init__(self, path: Path, *, repair: bool = False) -> None:
        self.path, self.rows, self.recovered, self.torn_files, self._unrepaired = Path(path), [], False, [], None
        if self.path.exists():
            data = self.path.read_bytes()
            good_end, pos = 0, 0
            segments = data.split(b"\n")  # the last element is whatever follows the final newline (empty for a clean file)
            for i, seg in enumerate(segments):
                pos += len(seg) + 1
                if not seg.strip():
                    if i < len(segments) - 1:
                        good_end = min(pos, len(data))
                    continue
                try:
                    self.rows.append(json.loads(seg.decode("utf8")))
                    good_end = min(pos, len(data))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if i >= len(segments) - 2:
                        self.recovered = True
                        break
                    raise v1.ClaimRelationError("attempts log is corrupt before its last line") from None
            missing_newline = bool(data) and not self.recovered and not data.endswith(b"\n")
            if self.recovered or missing_newline:
                if repair:
                    self._restore(data, good_end)
                else:  # read-only so far; the first append (if any) restores the journal before writing
                    self._unrepaired = (data, good_end)

    def _restore(self, data: bytes, good_end: int) -> None:
        damaged = data[good_end:]
        if damaged.strip():
            n = 1
            while (self.path.parent / f"{self.path.name}.torn.{n}").exists():
                n += 1
            side = self.path.parent / f"{self.path.name}.torn.{n}"
            with side.open("xb") as fh:
                fh.write(damaged)
                fh.flush()
                os.fsync(fh.fileno())
            self.torn_files.append(side.name)
        prefix = data[:good_end]
        if prefix and not prefix.endswith(b"\n"):
            prefix += b"\n"
        self._atomic_write(prefix)

    def _atomic_write(self, content: bytes) -> None:
        tmp = self.path.parent / (self.path.name + ".tmp")
        with tmp.open("wb") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def seed(self, rows: Sequence[dict[str, Any]]) -> None:
        """Create the journal with `rows` in ONE atomic step (no partially seeded journal can exist)."""
        if self.path.exists() and self.path.stat().st_size:
            raise v1.ClaimRelationError("journal already exists; seeding is only for a new journal")
        self._atomic_write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf8"))
        self.rows = list(rows)

    def append(self, row: dict[str, Any]) -> None:
        if self._unrepaired is not None:  # never write after damaged bytes
            data, good_end = self._unrepaired
            self._unrepaired = None
            self._restore(data, good_end)
        with self.path.open("ab") as fh:
            fh.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf8"))
            fh.flush()
            os.fsync(fh.fileno())
        self.rows.append(row)

    # ---- derived facts
    def _results(self, cid: str) -> list[dict[str, Any]]:
        return [r for r in self.rows if r.get("kind", "result") == "result" and r["case_id"] == cid]

    def valid(self, cid: str) -> dict[str, Any] | None:
        return next((r["record"] for r in self._results(cid) if r["record"]["status"] == "ok"), None)

    def malformed(self, cid: str) -> int:
        return sum(1 for r in self._results(cid) if r["record"]["status"] == "invalid_structure")

    def unsettled(self, cid: str | None = None) -> list[dict[str, Any]]:
        settled = {r["reservation_id"] for r in self.rows if r.get("kind", "result") == "result" and r.get("reservation_id")}
        return [r for r in self.rows if r.get("kind") == "reserve" and r["reservation_id"] not in settled and (cid is None or r["case_id"] == cid)]

    def attempts_used(self, cid: str) -> int:
        """Attempts that count against the two-attempt limit: malformed outputs plus unsettled reservations (outcome unknown, so conservatively consumed)."""
        return self.malformed(cid) + len(self.unsettled(cid))

    def new_calls(self) -> int:
        """Every reservation is a model call under the cap, whether or not its outcome was recorded."""
        return sum(1 for r in self.rows if r.get("kind") == "reserve")


def _pending(state: State, ids: Sequence[str]) -> list[str]:
    return [i for i in ids if state.valid(i) is None and state.attempts_used(i) < MAX_ATTEMPTS_MALFORMED]


# ------------------------------------------------------------------------------------------------------ freeze / verify
def _body(inputs_dir: Path, work: Path, runner: Callable[..., Any], original_run: Path, original_protocol: Path) -> dict[str, Any]:
    cases = cc._check_inputs(inputs_dir, parse_joins=False)["cases"]
    ids = [c["case_id"] for c in cases]
    records = import_original(inputs_dir, original_run, original_protocol)
    valid_ids = [r["case_id"] for r in records if r["status"] == "ok"]
    base = cc._body(inputs_dir, cc.cli_version(runner))
    base.pop("protocol", None)
    return base | {
        "protocol": PROTOCOL_NAME, "supersedes_for_continuation": {"original_protocol_sha256": ORIGINAL_PROTOCOL_SHA, "original_status": "INCOMPLETE (preserved)"},
        "supersedes_continuation_protocol_sha256": SUPERSEDED_CONTINUATION_SHA256, "restart_safety": "reserve-before-launch journal; unsettled reservations count toward cap and attempts; atomic torn-tail restore with preserved bytes; atomic idempotent import seeding",
        "retry_rule": {"max_attempts_per_case_for_malformed_output": MAX_ATTEMPTS_MALFORMED, "same_prompt_model_settings": True, "gold_labels_used": False,
                       "request_errors": "stop the run (ABORTED, resumable); a recorded request error does not consume the retry",
                       "unsettled_reservation": "consumes one attempt and one call under the cap; never replayed", "after_two_attempts": "case retained as failed; result INCOMPLETE"},
        "max_new_model_calls": MAX_NEW_MODEL_CALLS, "publish_kappa_only_if": "128/128 valid judgments; completeness is checked before any label file is parsed",
        "import_provenance": {
            "original_run_judgments_sha256": _sha(Path(original_run) / "judgments.jsonl"), "original_run_terminal_sha256": _sha(Path(original_run) / "terminal.json"),
            "original_protocol_file_sha256": _sha(original_protocol), "valid_judgments_imported": len(valid_ids), "valid_case_ids_digest": _digest(valid_ids),
            "malformed_attempt_imported_as_attempt_1": [r["case_id"] for r in records if r["status"] != "ok"],
            "pending_cases": len(ids) - len(valid_ids), "pending_case_ids_digest": _digest([i for i in ids if i not in set(valid_ids)]),
            "disclosure": "81 judgments were produced under the original protocol (CLI 2.1.289, same model, prompt, settings) and are reused unchanged after identity re-verification",
        },
        "backend_continuation_sha256": _sha(Path(__file__)),
    }


def freeze(inputs_dir: Path = INPUTS, work: Path = WORK, runner: Callable[..., Any] = subprocess.run, original_run: Path = ORIGINAL_RUN,
           original_protocol: Path = ORIGINAL_PROTOCOL) -> dict[str, Any]:
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    out = work / "protocol.json"
    if out.exists():
        raise v1.ClaimRelationError("protocol exists; a frozen protocol is never overwritten")
    cc._check_inputs(inputs_dir, parse_joins=True)
    body = _body(inputs_dir, work, runner, original_run, original_protocol)
    if not body["v18_identity"]["unchanged"]:
        raise v1.ProtocolError("v18 identity differs from the frozen values")
    out.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf8")
    return body


def verify(inputs_dir: Path, work: Path, expected_sha256: str, runner: Callable[..., Any] = subprocess.run, original_run: Path = ORIGINAL_RUN,
           original_protocol: Path = ORIGINAL_PROTOCOL) -> dict[str, Any]:
    proto = Path(work) / "protocol.json"
    if _sha(proto) != expected_sha256:
        raise v1.ProtocolError("protocol file hash differs from the hash recorded at freeze time")
    frozen = json.loads(proto.read_text(encoding="utf8"))
    current = _body(inputs_dir, work, runner, original_run, original_protocol)
    for key, value in current.items():
        if frozen.get(key) != value:
            raise v1.ProtocolError(f"frozen protocol differs from the current state at {key!r}")
    return frozen


# ------------------------------------------------------------------------------------------------------ run (resumable)
def usage_estimate(original_run: Path = ORIGINAL_RUN, pending: int = 47) -> dict[str, Any]:
    """ESTIMATES from the original calls (not guarantees). Only the call count is a hard bound (the protocol's cap); tokens, time and cost are extrapolations."""
    recs = [json.loads(x) for x in (Path(original_run) / "judgments.jsonl").read_text(encoding="utf8").splitlines() if x.strip()]
    u = [r["usage"] for r in recs if r.get("usage")]
    n = len(u)
    keys = ("output_tokens", "latency_s", "notional_list_cost_usd")
    mean = {k: sum((x.get(k) or 0) for x in u) / n for k in keys}
    worst = {k: max((x.get(k) or 0) for x in u) for k in keys}
    failure_rate = (len(recs) - sum(r["status"] == "ok" for r in recs)) / len(recs)
    expected_calls = pending * (1 + failure_rate)

    def scale(calls: float, per: dict[str, float]) -> dict[str, float]:
        return {"output_tokens": round(calls * per["output_tokens"]), "minutes": round(calls * per["latency_s"] / 60, 1),
                "notional_list_cost_usd": round(calls * per["notional_list_cost_usd"], 2)}

    return {"basis": f"{n} original calls", "per_call_mean": {k: round(v, 3) for k, v in mean.items()}, "per_call_max_observed": {k: round(v, 3) for k, v in worst.items()},
            "pending_cases": pending, "expected_calls_estimate": round(expected_calls, 1),
            "hard_bound_on_calls": {"if_no_request_errors": 2 * pending, "protocol_cap": MAX_NEW_MODEL_CALLS},
            "expected_estimate": scale(expected_calls, mean),
            "two_attempts_for_every_case_estimate_at_mean_cost": scale(2 * pending, mean),
            "two_attempts_for_every_case_estimate_at_max_observed_cost": scale(2 * pending, worst),
            "note": "estimates extrapolated from observed calls, NOT upper bounds on tokens/time/cost; subscription allowance, not an invoice; the plan limit may stop the run (ABORTED, resumable)"}


def run(inputs_dir: Path, work: Path, request: v1.Request, *, expected_sha256: str, runner: Callable[..., Any] = subprocess.run, original_run: Path = ORIGINAL_RUN,
        original_protocol: Path = ORIGINAL_PROTOCOL) -> dict[str, Any]:
    """Verify freeze + subscription, seed from the verified import (first invocation), then judge only cases lacking a valid result and still under the attempt
    limit. Safe to invoke again after an ABORTED/crashed invocation: valid cases are never re-judged and the attempt limit is enforced from the log."""
    verify(inputs_dir, work, expected_sha256, runner, original_run, original_protocol)
    auth = cc.verify_subscription(runner)
    work = Path(work)
    cases = cc._check_inputs(inputs_dir, parse_joins=False)["cases"]
    by_id = {c["case_id"]: c for c in cases}
    ids = [c["case_id"] for c in cases]
    state = State(work / "attempts.jsonl", repair=True)
    originals = import_original(inputs_dir, original_run, original_protocol)
    imported_rows = [{"case_id": r["case_id"], "origin": "imported", "attempt_note": "original run", "record": r} for r in originals]
    if not state.rows:
        state.seed(imported_rows)  # one atomic step: a partially seeded journal cannot exist
    else:  # idempotent completion + integrity: every verified original record must be present exactly once, unchanged
        have = {r["case_id"]: r for r in state.rows if r.get("origin") == "imported"}
        for row in imported_rows:
            if row["case_id"] not in have:
                state.append(row)
            elif have[row["case_id"]]["record"] != row["record"]:
                raise ImportError_(f"imported row for {row['case_id']} in the journal differs from the verified original record")
        if sum(1 for r in state.rows if r.get("origin") == "imported") != len(imported_rows):
            raise ImportError_("journal holds duplicate or unknown imported rows")
    started_calls, stop = state.new_calls(), None
    t0 = time.perf_counter()
    for cid in _pending(state, ids):
        while state.valid(cid) is None and state.attempts_used(cid) < MAX_ATTEMPTS_MALFORMED:
            if state.new_calls() >= MAX_NEW_MODEL_CALLS:
                stop = {"case_id": cid, "status": "call_cap", "error": f"{MAX_NEW_MODEL_CALLS} model calls reached"}
                break
            rid, attempt_no = f"{cid}#{state.new_calls() + 1}", state.attempts_used(cid) + 1
            # durable reservation BEFORE launching: if the process dies anywhere after this line the attempt still counts
            state.append({"kind": "reserve", "case_id": cid, "origin": "new", "reservation_id": rid, "attempt_no": attempt_no})
            rec = v2.judge_case(by_id[cid], request)
            state.append({"kind": "result", "case_id": cid, "origin": "new", "reservation_id": rid, "record": rec,
                          "attempt_note": "request error" if rec["status"] in ("aborted", "request_error") else f"attempt {attempt_no}"})
            if rec["status"] in ("aborted", "request_error"):
                stop = {"case_id": cid, "status": rec["status"], "error": rec.get("error")}
                break
        if stop:
            break
    return _terminal(work, ids, state, auth, stop, started_calls, time.perf_counter() - t0, inputs_dir, expected_sha256)


def _terminal(work: Path, ids: Sequence[str], state: State, auth: Any, stop: Any, started_calls: int, wall: float, inputs_dir: Path, protocol_sha: str) -> dict[str, Any]:
    valid = [i for i in ids if state.valid(i)]
    failed = [i for i in ids if state.valid(i) is None and state.attempts_used(i) >= MAX_ATTEMPTS_MALFORMED]
    untried = [i for i in ids if i not in set(valid) | set(failed)]
    new = [r["record"] for r in state.rows if r["origin"] == "new" and r.get("kind", "result") == "result"]
    u = [r.get("usage") or {} for r in new]
    status = "ABORTED" if stop else ("COMPLETE" if len(valid) == len(ids) else "INCOMPLETE")
    t = {"status": status, "cases": len(ids), "valid_judgments": len(valid), "imported_valid": sum(1 for r in state.rows if r["origin"] == "imported" and r["record"]["status"] == "ok"),
         "unsettled_reservations": [r["reservation_id"] for r in state.unsettled()], "torn_tail_sidecar_files": state.torn_files,
         "new_valid": sum(1 for r in state.rows if r["origin"] == "new" and r.get("kind", "result") == "result" and r["record"]["status"] == "ok"), "failed_after_retry": failed, "not_yet_attempted_or_interrupted": untried,
         "stopped_on": stop, "new_model_calls_total": state.new_calls(), "new_model_calls_this_invocation": state.new_calls() - started_calls,
         "attempt_counts": {str(k): sum(1 for i in ids if state.attempts_used(i) + (1 if state.valid(i) else 0) == k) for k in (1, 2)},
         "models_reported": sorted({r["model"] for r in new if r.get("model")}), "frozen_model": cc.MODEL, "auth": auth,
         "token_totals_new": {k: sum((x.get(k) or 0) for x in u) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")},
         "notional_list_cost_usd_new": round(sum((x.get("notional_list_cost_usd") or 0) for x in u), 4), "latency_s_new": round(sum((x.get("latency_s") or 0) for x in u), 1),
         "wall_s_this_invocation": round(wall, 1), "truncated_log_line_recovered": state.recovered, "protocol_sha256": protocol_sha, "variant": v2.VARIANT_ID,
         "content_hash": v2.content_hash(), "kappa": "not computed here; `compare` publishes it only when status is COMPLETE (128/128 valid)"}
    (Path(work) / "terminal.json").write_text(json.dumps(t, indent=2), encoding="utf8")
    with (Path(work) / "invocations.jsonl").open("a", encoding="utf8") as fh:
        fh.write(json.dumps({k: t[k] for k in ("status", "valid_judgments", "new_model_calls_this_invocation", "stopped_on", "wall_s_this_invocation")}) + "\n")
    return t


# ------------------------------------------------------------------------------------------------------ compare
def compare(inputs_dir: Path, work: Path, *, expected_sha256: str, runner: Callable[..., Any] = subprocess.run, original_run: Path = ORIGINAL_RUN,
            original_protocol: Path = ORIGINAL_PROTOCOL) -> dict[str, Any]:
    """Completeness FIRST, from the cases file and the attempts log only. Labels, groups, baseline and splits are parsed only if 128/128 judgments are valid."""
    frozen = verify(inputs_dir, work, expected_sha256, runner, original_run, original_protocol)
    work = Path(work)
    cases = cc._check_inputs(inputs_dir, parse_joins=False)["cases"]
    ids = [c["case_id"] for c in cases]
    state = State(work / "attempts.jsonl")
    valid = {i: state.valid(i) for i in ids}
    missing = [i for i in ids if valid[i] is None]
    if missing:
        result = {"status": "WITHHELD_INCOMPLETE", "valid_judgments": len(ids) - len(missing), "cases": len(ids), "missing_or_failed": missing,
                  "note": "no label, group, baseline or split file was read or parsed; no metric exists"}
        with (work / "comparison_attempts.jsonl").open("a", encoding="utf8") as fh:
            fh.write(json.dumps({k: result[k] for k in ("status", "valid_judgments", "cases")}) + "\n")
        return result
    target = work / "comparison.json"
    if target.exists():
        raise v1.ClaimRelationError("comparison.json exists; the comparison is written once")
    by_case = {c["case_id"]: c for c in cases}
    for i in ids:
        if valid[i]["input"]["source_sha256"] != v1.sha256(by_case[i]["source"]) or valid[i]["input"]["candidate_sha256"] != v1.sha256(by_case[i]["candidate"]):
            raise v1.ProtocolError(f"judgment for {i} does not match the frozen inputs")
    data = cc._check_inputs(inputs_dir, parse_joins=True)
    result = v1.compare_to_baseline({"status": "COMPLETE"}, [valid[i] for i in ids], data["labels.json"], data["baseline_v18.json"], data["groups.json"], data["splits.json"],
                                    uncertain_as=frozen["decision_rule"]["uncertain_as"], draws=frozen["bootstrap"]["draws"], seed=frozen["bootstrap"]["seed"])
    term = json.loads((work / "terminal.json").read_text(encoding="utf8"))
    result["run"] = {"coverage": f"{len(ids)}/{len(ids)}", "imported_valid": term["imported_valid"], "new_valid": term["new_valid"], "attempt_counts": term["attempt_counts"],
                     "models_reported_new_calls": term["models_reported"], "new_model_calls_total": term["new_model_calls_total"],
                     "notional_list_cost_usd_new": term["notional_list_cost_usd_new"], "token_totals_new": term["token_totals_new"]}
    target.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["freeze", "estimate", "run", "compare"])
    p.add_argument("--inputs", type=Path, default=INPUTS)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--protocol-sha256")
    p.add_argument("--confirm", default="")
    a = p.parse_args(argv)
    try:
        if a.command == "freeze":
            body = freeze(a.inputs, a.work)
            print(json.dumps({"protocol_sha256": _sha(a.work / "protocol.json"), "usage_estimate": usage_estimate(), **body}, indent=2))
        elif a.command == "estimate":
            print(json.dumps(usage_estimate(), indent=2))
        else:
            if not a.protocol_sha256:
                print("REFUSED: --protocol-sha256 (printed at freeze) is required", file=sys.stderr)
                return 2
            if a.command == "run":
                if a.confirm != CONFIRM:
                    print(f"REFUSED: needs --confirm {CONFIRM}", file=sys.stderr)
                    return 2
                t = run(a.inputs, a.work, cc.make_request(), expected_sha256=a.protocol_sha256)
                print(json.dumps(t, indent=2))
                return 0 if t["status"] == "COMPLETE" else 3
            print(json.dumps(compare(a.inputs, a.work, expected_sha256=a.protocol_sha256), indent=2))
    except v1.ClaimRelationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

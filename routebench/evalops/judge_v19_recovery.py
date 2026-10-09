"""Explicit, auditable recovery of ONE judge-v19 Stage 1a job whose failure provably happened before any request was sent (empty authorization token).

Why this exists: the frozen runner stores a failed job as a permanent row, so `--resume` alone would silently skip it. This module does NOT edit the frozen runner, protocol, journal, or the
failed row. It (1) records a note with the hashes of everything it relies on BEFORE any call, (2) re-judges exactly that job through the frozen recipe (`srcint_eval_v3.judge_case`: same prompt,
config, one journaled HTTP request per reservation) against the SAME spend journal and the SAME stage cap, and (3) writes the new row to a separate file, so both the original failed record and the
recovered record exist. `assemble_scores` then re-validates every row with the frozen validator and refuses unless the original failed row and the journal prefix are byte-identical to the note.
It refuses to recover anything whose failure is not the empty-token error with no usage recorded: a genuine model/parse failure is never retried this way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import judge_v19_phase1 as jv
from .audit2_v2 import Journal, SpendError

RECOVERY_CONFIRM = "RECOVER-JV19-STAGE-1A-JOB"
NOTE, RESULTS = "stage_1a_recovery_note.json", "recovery_results.jsonl"
NO_REQUEST_REASON = "LocalProtocolError: Illegal header value"


class RecoveryError(ValueError):
    pass


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _failed_row(sd: Path) -> tuple[str, dict[str, Any], bytes]:
    failed = []
    for line in (sd / "results.jsonl").read_bytes().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["record"].get("status") != "ok":
                failed.append((row["case_id"], row, line))
    if len(failed) != 1:
        raise RecoveryError(f"exactly one failed row is recoverable by this module, found {len(failed)}")
    return failed[0]


def preconditions(work: Path, sha: str) -> dict[str, Any]:
    frozen, _pop, _jobs, jcases = jv.verify(work, sha)
    sd = Path(work) / "stage_1a"
    term = jv._terminal(sd)
    if term is None or term.get("status") not in {"ABORTED", "INCOMPLETE"}:
        raise RecoveryError("recovery needs an ABORTED or INCOMPLETE Stage 1a terminal record")
    jid, row, line = _failed_row(sd)
    rec = row["record"]
    if not str(rec.get("reason", "")).startswith(NO_REQUEST_REASON) or rec.get("usage") or rec.get("identity_failure"):
        raise RecoveryError("the failed job is not the empty-token (no request sent) failure; a genuine failure is never retried by this module")
    cap = frozen["stages"]["1a"]["cap_usd"]
    state = jv.sv3.journal_state(sd / "spend_journal.jsonl", cap)
    if state["unsettled"] or state["over_cap"]:
        raise RecoveryError("the journal has unsettled reservations or is over its cap")
    for a in rec["attempt_ids"]:
        r = state["reserves"][a]
        if a not in state["settled"] or r.get("case") != f"{jv.VARIANT}:{jid}":
            raise RecoveryError("the failed row's attempts are not settled reservations of that job")
    if (sd / RESULTS).exists() or (Path(work) / NOTE).exists():
        raise RecoveryError("a recovery has already been recorded for this stage")
    if jid not in jcases["1a"]:
        raise RecoveryError("the failed job is not part of Stage 1a")
    return {"frozen": frozen, "sd": sd, "job": jid, "failed_line": line, "state": state, "cap": cap, "case": jcases["1a"][jid]}


def recover(work: Path, sha: str, confirm: str, request: Any = jv.sv3.http_request_raw) -> dict[str, Any]:
    if confirm != RECOVERY_CONFIRM:
        raise RecoveryError(f"recovery needs --confirm {RECOVERY_CONFIRM}")
    work = Path(work)
    p = preconditions(work, sha)
    sd, jid = p["sd"], p["job"]
    results_bytes, journal_bytes = (sd / "results.jsonl").read_bytes(), (sd / "spend_journal.jsonl").read_bytes()
    note = {"protocol_sha256": sha, "job": jid, "purpose": "re-judge the one job that failed before any request was sent; nothing frozen is edited",
            "failed_row_sha256": _sha_bytes(p["failed_line"]), "results_sha256_before": _sha_bytes(results_bytes), "journal_sha256_before": _sha_bytes(journal_bytes),
            "journal_prefix_bytes": len(journal_bytes), "committed_usd_before": round(p["state"]["committed"], 6), "cap_usd": p["cap"],
            "runner_sha256": jv._sha(Path(jv.__file__)), "recovery_module_sha256": jv._sha(Path(__file__)),
            "cause": "the API key is read from .env relative to the working directory; the first attempt ran from routebench/ where it was empty"}
    note_path = work / NOTE
    with note_path.open("x", encoding="utf8") as fh:  # written once, BEFORE any call
        fh.write(json.dumps(note, indent=2))
        fh.flush()
        os.fsync(fh.fileno())
    lock = sd / "run.lock"
    try:
        with lock.open("x", encoding="utf8") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "recovery": True}))
    except FileExistsError as exc:
        raise RecoveryError("another runner holds run.lock") from exc
    try:
        journal = Journal(sd / "spend_journal.jsonl", p["cap"])
        try:
            rec = jv.sv3.judge_case(p["case"], journal, request, sha)
        except SpendError as exc:
            return {"status": "ABORTED", "job": jid, "stopped_on": str(exc)}
        with (sd / RESULTS).open("x", encoding="utf8") as fh:
            fh.write(json.dumps({"case_id": jid, "protocol_sha256": sha, "record": rec}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        state = jv.sv3.journal_state(sd / "spend_journal.jsonl", p["cap"])
        return {"status": rec["status"], "job": jid, "reason": rec.get("reason"), "committed_usd_after": round(state["committed"], 4), "cap_usd": p["cap"]}
    finally:
        lock.unlink(missing_ok=True)


def assemble_scores(work: Path, sha: str) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Scores for all 1,028 Stage 1a jobs: 1,027 from the frozen runner's validated rows + 1 from the validated recovery row. Refuses on any inconsistency."""
    work = Path(work)
    frozen, _pop, jobs, jcases = jv.verify(work, sha)
    sd = work / "stage_1a"
    cap = frozen["stages"]["1a"]["cap_usd"]
    term = jv._terminal(sd)
    note = json.loads((work / NOTE).read_text(encoding="utf8"))
    jid = note["job"]
    if term is None or term.get("status") != "INCOMPLETE" or term.get("failed") != [jid] or term.get("not_attempted") != 0 or term.get("protocol_sha256") != sha:
        raise RecoveryError("expected an INCOMPLETE terminal record whose only failed job is the recorded one and which has no unattempted jobs")
    state = jv.sv3.journal_state(sd / "spend_journal.jsonl", cap)
    if state["unsettled"] or state["over_cap"]:
        raise RecoveryError("journal has unsettled reservations or is over its cap")
    if _sha_bytes((sd / "spend_journal.jsonl").read_bytes()[: note["journal_prefix_bytes"]]) != note["journal_sha256_before"]:
        raise RecoveryError("the journal prefix recorded before recovery has changed")
    items = [{"case_id": j["job_id"]} for j in jobs["1a"]]
    rows = jv.sv3.validate_rows(sd, frozen, items, jcases["1a"], state, sha)
    if len(rows) != len(items) or [c for c, r in rows.items() if r["status"] != "ok"] != [jid]:
        raise RecoveryError("the main results must hold 1,028 rows with exactly the recorded failed job")
    failed_lines = [line for line in (sd / "results.jsonl").read_bytes().splitlines() if line.strip() and json.loads(line)["case_id"] == jid]
    if len(failed_lines) != 1 or _sha_bytes(failed_lines[0]) != note["failed_row_sha256"]:
        raise RecoveryError("the original failed row is not byte-identical to the preserved record")
    with tempfile.TemporaryDirectory() as tmp:  # the frozen validator on the recovery row alone, against the same journal and case
        (Path(tmp) / "results.jsonl").write_bytes((sd / RESULTS).read_bytes())
        rec_rows = jv.sv3.validate_rows(Path(tmp), frozen, [{"case_id": jid}], {jid: jcases["1a"][jid]}, state, sha)
    if list(rec_rows) != [jid] or rec_rows[jid]["status"] != "ok":
        raise RecoveryError("the recovery row is missing or not ok")
    if set(rec_rows[jid]["attempt_ids"]) & set(rows[jid]["attempt_ids"]):
        raise RecoveryError("recovery attempts overlap the original failed attempts")
    rows[jid] = rec_rows[jid]
    out: dict[str, dict[int, float]] = {}
    for j in jobs["1a"]:
        out.setdefault(j["base"], {})[j["replicate"]] = float(rows[j["job_id"]]["score"])
    info = {"recovered_job": jid, "original_failed_row_sha256": note["failed_row_sha256"], "committed_usd": round(state["committed"], 4), "cap_usd": cap,
            "terminal_status": term["status"], "jobs_valid": len(rows)}
    return {b: [v[r] for r in sorted(v)] for b, v in out.items()}, info


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=jv.WORK)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--confirm", default="")
    a = p.parse_args(argv)
    try:
        print(json.dumps(recover(a.work, a.protocol_sha256, a.confirm), indent=2))
    except (RecoveryError, jv.Phase1Error, jv.sv3.SrcIntV3Error) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

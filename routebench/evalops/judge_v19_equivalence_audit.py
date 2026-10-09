"""Read-only consistency audit of a saved equivalence report. Never loads a model.

A legacy report lacks per-sentence vectors and producer identity. This audit states
that limitation rather than upgrading it into independently reproduced GPU evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from . import judge_v19_minicheck as mc


class ReportAuditError(ValueError):
    pass


def _number(value: Any, name: str, *, probability: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (probability and value > 1)):
        raise ReportAuditError(f"{name} is not a valid finite number")
    return float(value)


def audit(work: Path, protocol_sha: str, report: Path) -> dict[str, Any]:
    frozen, cases, _ = mc._verified_inputs(work, protocol_sha)
    n, tolerance = frozen["minicheck"]["equivalence_rows"], frozen["minicheck"]["tolerance"]
    if n != mc.EQUIVALENCE_ROWS or tolerance != mc.EQUIVALENCE_TOL or len(cases) < n:
        raise ReportAuditError("frozen equivalence controls differ from the expected 24-row protocol")
    cached = json.loads((work / "minicheck_cached_routebench.json").read_text(encoding="utf8"))
    body = json.loads(report.read_text(encoding="utf8"))
    if not isinstance(body, dict) or body.get("protocol_sha256") != protocol_sha:
        raise ReportAuditError("report belongs to a different protocol")
    if body.get("schema_version") not in (None, 2) or isinstance(body.get("schema_version"), bool):
        raise ReportAuditError("unknown report schema")
    if type(body.get("rows_checked")) is not int or body["rows_checked"] != n:
        raise ReportAuditError("report coverage differs from the frozen sample")
    if _number(body.get("tolerance"), "tolerance") != tolerance:
        raise ReportAuditError("report tolerance differs from the frozen tolerance")
    details = body.get("details")
    expected = sorted(cases, key=lambda cid: mc._h(f"equivalence|{cid}"))[:n]
    if (not isinstance(details, list) or len(details) != n
            or any(not isinstance(row, dict) for row in details)
            or [row.get("case_id") for row in details] != expected):
        raise ReportAuditError("report has missing, duplicate, reordered or unexpected case IDs")
    if body.get("sentence_split_mismatches") != []:
        raise ReportAuditError("report records a sentence-count mismatch")
    max_difference, vectors = 0.0, 0
    for row in details:
        cid = row["case_id"]
        if row.get("sentence_split_matches") is not True:
            raise ReportAuditError("report row does not confirm matching sentence counts")
        full = row.get("full")
        if not isinstance(full, list) or len(full) != 2:
            raise ReportAuditError("full-score pair is malformed")
        actual, reference = [_number(v, "full", probability=True) for v in full]
        if reference != _number(cached[cid]["full"], "cached full", probability=True):
            raise ReportAuditError("report reference differs from the frozen cache")
        difference = _number(row.get("max_abs_difference"), "row difference")
        if difference < abs(actual - reference) or difference > tolerance:
            raise ReportAuditError("row difference contradicts the recorded full-score pair")
        present = {"recomputed_sent", "cached_sent"} & row.keys()
        if present:
            left, right = row.get("recomputed_sent"), row.get("cached_sent")
            if (not isinstance(left, list) or not isinstance(right, list)
                    or not left or len(left) != len(right) or right != cached[cid]["sent"]):
                raise ReportAuditError("sentence vectors are malformed or differ from the frozen cache")
            diffs = [abs(_number(a, "sentence", probability=True) - _number(b, "sentence", probability=True))
                     for a, b in zip(left, right, strict=True)]
            if max([abs(actual - reference), *diffs]) != difference:
                raise ReportAuditError("sentence vectors contradict the row difference")
            vectors += 1
        max_difference = max(max_difference, difference)
    if _number(body.get("worst_abs_difference"), "worst difference") != max_difference:
        raise ReportAuditError("aggregate difference contradicts row differences")
    if body.get("passed") is not True:
        raise ReportAuditError("saved report did not pass")
    if body.get("schema_version") == 2 and (vectors != n or body.get("execution") not in {"local_model", "injected_checker"}):
        raise ReportAuditError("version-2 report lacks sentence evidence or execution identity")
    producer = body.get("producer_sha256")
    if body.get("schema_version") == 2 and (not isinstance(producer, str) or len(producer) != 64
                                           or any(c not in "0123456789abcdef" for c in producer)):
        raise ReportAuditError("version-2 producer digest is malformed")
    producer_matches = None if not producer else producer == mc._sha(Path(__file__).with_name("judge_v19_equivalence.py"))
    limitations = ["No GPU inference was independently rerun; hardware, time and VRAM are reported telemetry.",
                   "A 24-row consistency check does not establish equivalence on all populations or improve kappa."]
    if vectors != n:
        limitations.append("Legacy report omits per-sentence vectors: their zero difference is reported, not independently recalculated.")
    if not body.get("producer_sha256"):
        limitations.append("Legacy report has no producer digest; execution-source provenance is incomplete.")
    elif not producer_matches:
        limitations.append("The recorded producer does not match the current standalone checker source.")
    return {"status": "CONSISTENT_WITH_FROZEN_SAMPLE", "report_sha256": mc._sha(report),
            "protocol_sha256": protocol_sha, "rows_checked": n, "full_pairs_checked": n,
            "sentence_vectors_checked": vectors, "multi_sentence_rows_in_cache": sum(len(cached[c]["sent"]) > 1 for c in expected),
            "worst_recorded_difference": max_difference, "stage_1b_authorization_granted_by_audit": False,
            "release_record_present": (work / "release_1b.json").exists(),
            "producer_matches_current_checker": producer_matches,
            "kappa_improvement_measured": False, "limitations": limitations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.work, args.protocol_sha256, args.report), indent=2))
    except (OSError, ValueError, TypeError, KeyError, mc.MiniCheckJobError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Second-pass auditor, execution version 2 (supersedes the unrun v1 `audit2` execution layer; v1 files are preserved).

Prompt, decision rule and error taxonomy are IMPORTED UNCHANGED from `audit2` (both module files are hash-bound in the freeze). What v2 repairs:
- Spend is journaled durably (fsync) BEFORE each provider attempt and after it; a crash leaves an unsettled reservation that is charged at its bound.
  The cap covers the journal's total across stages and restarts. Every provider attempt, including failed/unparseable ones, is reserved and settled.
  Reservation uses a deliberately pessimistic input bound (1 token per 2 characters) plus the full output bound (reasoning tokens bill as output).
  A cost above its reservation, or a non-finite/negative cost, stops the run (ABORTED); a missing cost is charged at the bound, never zero.
- The frozen inputs are bound by full-content digests (source, candidate, label, v18 verdict per case), the data files, settings and both code files, and
  are re-derived and verified BEFORE any provider call; changed, missing, duplicate or malformed cases stop the run.
- Output fields are type-validated; every failure is recorded durably; the κ comparison is WITHHELD unless every audited case has an ok audit.
  κ is None when undefined (zero variance in chance agreement).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import audit2 as v1
from .audit2 import AuditParseError, AuditResult, build_prompt, decide

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "data"
WORK = ROOT / ".local" / "audit2-v2"
CONFIRM = "RUN-AUDIT2-PAID"
DATA_FILES = ("cases.jsonl", "annotations.jsonl", "judgment_cache.jsonl", "splits.json")
EXPECTED = {"train": 94, "dev": 88}
MODEL, MAX_TOKENS, MAX_ATTEMPTS = v1.MODEL, v1.MAX_TOKENS, v1.MAX_ATTEMPTS
TOTAL_CAP_USD = 3.0


class SpendError(RuntimeError):
    """Cap would be exceeded, or a cost was invalid / above its reservation. The run stops."""


class CapReached(SpendError):
    pass


class FreezeError(ValueError):
    """The inputs, code or settings differ from the frozen protocol."""


# ------------------------------------------------------------------------------------------------------------- spend
def prompt_tokens_bound(prompt: str) -> int:
    return len(prompt) // 2 + 200  # >= real count for English text (about 4 chars/token); intentionally pessimistic


def worst_case_usd(prompt: str) -> float:
    return (prompt_tokens_bound(prompt) * v1.INPUT_PER_1M + MAX_TOKENS * v1.OUTPUT_PER_1M) / 1e6


class Journal:
    """Append-only, fsynced spend log shared by every stage. Total committed = settled charges + unsettled reservations at their bound."""

    def __init__(self, path: Path, cap_usd: float) -> None:
        self.path, self.cap, self._lock = path, cap_usd, threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _entries(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.path.read_text(encoding="utf8").splitlines() if line.strip()]

    def _append(self, entry: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def committed(self) -> float:
        reserved: dict[str, float] = {}
        total = 0.0
        for e in self._entries():
            if e["op"] == "reserve":
                reserved[e["id"]] = e["usd"]
            else:
                reserved.pop(e["id"], None)
                total += e["charged"]
        return total + sum(reserved.values())  # crashed (unsettled) reservations count at their bound

    def reserve(self, case: str, attempt: int, usd: float) -> str:
        with self._lock:
            if self.committed() + usd > self.cap + 1e-12:
                raise CapReached(f"cap ${self.cap:.2f} would be exceeded (committed {self.committed():.4f}, call bound {usd:.4f})")
            rid = f"{case}#{attempt}#{len(self._entries())}"
            self._append({"op": "reserve", "id": rid, "case": case, "attempt": attempt, "usd": usd})
            return rid

    def settle(self, rid: str, reserved: float, actual: float | None) -> None:
        """Record the real cost. Missing -> charged at the bound. Invalid or above the reservation -> recorded, then the run must stop."""
        problem = None
        if actual is None:
            charged = reserved
        elif not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isfinite(actual) or actual < 0:
            charged, problem = reserved, f"invalid provider cost {actual!r}"
        elif actual > reserved:
            charged, problem = float(actual), f"cost {actual:.4f} exceeded its reservation {reserved:.4f}"
        else:
            charged = float(actual)
        with self._lock:
            self._append({"op": "settle", "id": rid, "charged": charged, "actual": None if actual is None or problem else float(actual)})
        if problem:
            raise SpendError(problem)


Call = Callable[[str], tuple[str, float | None]]


# ---------------------------------------------------------------------------------------------------------- parsing
def _opt_str(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AuditParseError(f"{key} must be a string or null")
    return value.strip() or None


def parse(text: str) -> AuditResult:
    """Type-validated parse. Any shape error is an AuditParseError (a recorded failure), never an uncaught exception."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise AuditParseError("no JSON object")
    try:
        raw = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise AuditParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise AuditParseError("top-level JSON must be an object")
    verdict = raw.get("verdict")
    if verdict not in ("supported", "unsupported"):
        raise AuditParseError("verdict must be supported|unsupported")
    kind = raw.get("error_type")
    if verdict == "unsupported" and kind not in v1.ERROR_TYPES:
        raise AuditParseError("unsupported verdict needs a known error_type")
    if kind is not None and not isinstance(kind, str):
        raise AuditParseError("error_type must be a string or null")
    details = raw.get("unsupported_details")
    if details is None:
        details = []
    if not isinstance(details, list) or any(not isinstance(d, str) for d in details):
        raise AuditParseError("unsupported_details must be a list of strings")
    explanation = raw.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        raise AuditParseError("explanation must be a string or null")
    return AuditResult(verdict, _opt_str(raw, "disputed_assertion"), kind, _opt_str(raw, "claim_says"), _opt_str(raw, "source_says"),
                       _opt_str(raw, "evidence_quote"), tuple(d.strip() for d in details if d.strip()), (explanation or "").strip())


# -------------------------------------------------------------------------------------------------------- auditing
def audit_one(item: Mapping[str, str], call: Call, journal: Journal) -> dict[str, Any]:
    """One case, every provider attempt journaled. Failures are recorded, never flipped. SpendError propagates (the run must stop)."""
    prompt, last = build_prompt(item["source"], item["candidate"]), ""
    bound = worst_case_usd(prompt)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rid = journal.reserve(item["id"], attempt, bound)
        try:
            text, cost = call(prompt)
        except Exception as exc:  # noqa: BLE001 - a provider failure may still have billed: charge the bound
            journal.settle(rid, bound, None)
            last = f"{type(exc).__name__}: {exc}"
            continue
        journal.settle(rid, bound, cost)
        try:
            result = parse(text)
            flip, reason = decide(result, item["source"], item["candidate"])
        except (AuditParseError, TypeError, AttributeError) as exc:
            last = f"parse: {exc}"
            continue
        return {"id": item["id"], "status": "ok", "attempts": attempt, "flip": flip, "reason": reason,
                "result": result.__dict__ | {"unsupported_details": list(result.unsupported_details)}}
    return {"id": item["id"], "status": "failed", "attempts": MAX_ATTEMPTS, "flip": False, "reason": last}


# ------------------------------------------------------------------------------------------------------------ binding
def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def population_digest(items: Sequence[Mapping[str, Any]]) -> str:
    """Digest over the FULL content the experiment depends on: ids, source, candidate, gold label and the frozen v18 verdict."""
    return _sha(sorted(([i["id"], i["source"], i["candidate"], i["label"], i["v18"]] for i in items), key=lambda r: r[0]))


def check_items(items: Sequence[Mapping[str, Any]], expected_n: int | None = None) -> None:
    ids = [i.get("id") for i in items]
    if any(not isinstance(x, str) or not x for x in ids):
        raise FreezeError("missing or non-string case id")
    if len(set(ids)) != len(ids):
        raise FreezeError("duplicate case ids")
    for i in items:
        if not all(isinstance(i.get(k), str) and i[k].strip() for k in ("source", "candidate")) or i.get("label") not in (0, 1) \
                or i.get("v18") not in (0, 1):
            raise FreezeError(f"malformed case {i.get('id')}")
    if expected_n is not None and len(items) != expected_n:
        raise FreezeError(f"expected {expected_n} cases, got {len(items)}")


def code_and_data_hashes() -> dict[str, str]:
    out = {"audit2.py": file_sha(Path(v1.__file__)), "audit2_v2.py": file_sha(Path(__file__))}
    out.update({f"data/{name}": file_sha(DATA / name) for name in DATA_FILES})
    out["settings"] = _sha([MODEL, MAX_TOKENS, MAX_ATTEMPTS, TOTAL_CAP_USD, "reasoning_effort=medium", "tokens=len//2+200"])
    return out


def prepare(work: Path = WORK) -> dict[str, Any]:
    population = v1._population()
    audited = {k: [r for r in v if r["v18"] == 1] for k, v in population.items()}
    for stage, items in audited.items():
        check_items(items, EXPECTED[stage])
    body = {"protocol": "audit2-v2", "supersedes": "audit2-v1 (never executed; .local/audit2-v1 preserved)", "hashes": code_and_data_hashes(),
            "cap_usd_total": TOTAL_CAP_USD, "stages": EXPECTED, "stage_order": ["train", "dev"],
            "population_digest": {k: population_digest(v) for k, v in audited.items()},
            "cost_plan": v1.plan_cost(sum(EXPECTED.values()))}
    work.mkdir(parents=True, exist_ok=False)
    (work / "protocol.json").write_text(json.dumps(body, indent=2), encoding="utf8")
    (work / "population.json").write_text(json.dumps(audited), encoding="utf8")
    return body


def verify_freeze(work: Path, stage: str, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Raise FreezeError before any provider call if code, data, settings or this stage's inputs differ from the frozen protocol."""
    protocol = json.loads((work / "protocol.json").read_text(encoding="utf8"))
    if protocol["hashes"] != code_and_data_hashes():
        raise FreezeError("code, data files or settings changed since the freeze")
    check_items(items, protocol["stages"][stage])
    if population_digest(items) != protocol["population_digest"][stage]:
        raise FreezeError(f"{stage} inputs differ from the frozen population (content digest mismatch)")
    return protocol


# ---------------------------------------------------------------------------------------------------------- running
def execute_stage(work: Path, stage: str, items: Sequence[Mapping[str, Any]], call: Call, *, cap_usd: float = TOTAL_CAP_USD,
                  resume: bool = False) -> dict[str, Any]:
    """Verify the freeze, then audit every v18-passed case. Rows and spend are durable. Returns the terminal record (also written to disk)."""
    protocol = verify_freeze(work, stage, items)
    cap = min(cap_usd, protocol["cap_usd_total"])
    terminal_path, rows_path = work / f"terminal_{stage}.json", work / f"rows_{stage}.jsonl"
    if terminal_path.exists():
        raise FreezeError("stage already has a terminal record; refusing a second attempt")
    if stage == "dev" and json.loads((work / "terminal_train.json").read_text(encoding="utf8")).get("status") != "COMPLETE":
        raise FreezeError("dev runs only after a COMPLETE train stage")
    if rows_path.exists() and not resume:
        raise FreezeError("partial rows exist; rerun with resume to continue (spend is already journaled)")
    rows: dict[str, Any] = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["status"] == "ok":  # failed rows are retried on resume (each retry is journaled and capped)
                    rows[row["id"]] = row
    journal = Journal(work / "spend_journal.jsonl", cap)
    status = "COMPLETE"
    with rows_path.open("a", encoding="utf8") as stream:
        for item in items:
            if item["id"] in rows:
                continue
            try:
                row = audit_one(item, call, journal)
            except SpendError as exc:
                status = f"ABORTED: {exc}"
                break
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            rows[item["id"]] = row
    if status == "COMPLETE" and any(r["status"] != "ok" for r in rows.values()):
        status = "INCOMPLETE: failed audits"
    ids = {i["id"] for i in items}
    report = score_stage(items, rows) if status == "COMPLETE" and set(rows) == ids else {
        "status": "WITHHELD_INCOMPLETE", "audited_ok": sum(1 for r in rows.values() if r["status"] == "ok"), "expected": len(ids)}
    terminal = {"stage": stage, "status": status, "spent_usd_total": round(journal.committed(), 5), "cap_usd": cap, "audited": len(rows),
                "flips": sum(bool(r["flip"]) for r in rows.values()),
                "reasons": {k: sum(1 for r in rows.values() if r["reason"] == k) for k in sorted({r["reason"] for r in rows.values()})},
                "report": report}
    terminal_path.write_text(json.dumps(terminal, indent=2), encoding="utf8")
    return terminal


def kappa(labels: Sequence[int], preds: Sequence[int]) -> float | None:
    n = len(labels)
    if n == 0:
        return None
    po = sum(a == b for a, b in zip(labels, preds, strict=True)) / n
    p1, q1 = sum(labels) / n, sum(preds) / n
    pe = p1 * q1 + (1 - p1) * (1 - q1)
    return None if pe >= 1 - 1e-12 else (po - pe) / (1 - pe)  # undefined, not 1.0, when chance agreement is total


def score_stage(items: Sequence[Mapping[str, Any]], rows: Mapping[str, Any]) -> dict[str, Any]:
    """Compare v18 with v18+audit over ALL stage cases. Only called with a complete, all-ok audit set (v18 verdict kept for unaudited rows)."""
    ids = sorted(i["id"] for i in items)
    if set(rows) != set(ids) or any(r["status"] != "ok" for r in rows.values()):
        raise ValueError("coverage incomplete; comparison withheld")
    gold, base = {i["id"]: i["label"] for i in items}, {i["id"]: i["v18"] for i in items}
    new = {i: 0 if rows[i]["flip"] else base[i] for i in ids}
    corrected = sum(1 for i in ids if base[i] != gold[i] and new[i] == gold[i])
    introduced = sum(1 for i in ids if base[i] == gold[i] and new[i] != gold[i])
    return {"status": "COMPLETE", "n_audited_cases": len(ids), "corrected": corrected, "introduced": introduced,
            "note": "kappa here is over the audited (v18-passed) cases only; full-population kappa is computed from the dev terminal plus v18 rejections",
            "kappa_v18_on_audited": kappa([gold[i] for i in ids], [base[i] for i in ids]),
            "kappa_audited": kappa([gold[i] for i in ids], [new[i] for i in ids])}


def score_population(full: Sequence[Mapping[str, Any]], rows: Mapping[str, Any]) -> dict[str, Any]:
    """Full-population comparison (v18 passes audited, v18 rejections unchanged). Withheld unless every v18 pass has an ok audit."""
    passes = [i for i in full if i["v18"] == 1]
    if any(i["id"] not in rows or rows[i["id"]]["status"] != "ok" for i in passes):
        return {"status": "WITHHELD_INCOMPLETE"}
    gold, base = {i["id"]: i["label"] for i in full}, {i["id"]: i["v18"] for i in full}
    new = {i: 0 if rows.get(i, {}).get("flip") else base[i] for i in gold}
    ids = sorted(gold)
    return {"status": "COMPLETE", "n": len(ids), "corrected": sum(1 for i in ids if base[i] != gold[i] and new[i] == gold[i]),
            "introduced": sum(1 for i in ids if base[i] == gold[i] and new[i] != gold[i]),
            "kappa_v18": kappa([gold[i] for i in ids], [base[i] for i in ids]), "kappa_audited": kappa([gold[i] for i in ids], [new[i] for i in ids])}


# ------------------------------------------------------------------------------------------------------------- CLI
def _live_call(prompt: str) -> tuple[str, float | None]:
    from aisys import llm

    result = llm.chat([{"role": "user", "content": prompt}], model=MODEL, max_tokens=MAX_TOKENS, fallback=getattr(llm, "NO_FALLBACK", []),
                      reasoning_effort="medium")
    if result.model and not result.model.startswith(MODEL):
        raise RuntimeError(f"served model {result.model!r} is not {MODEL!r}")
    return result.text, result.cost_usd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--stage", choices=["train", "dev"])
    parser.add_argument("--cap", type=float, default=TOTAL_CAP_USD)
    parser.add_argument("--confirm", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), indent=2))
        return 0
    if args.stage is None:
        parser.error("--stage required")
    if args.confirm != CONFIRM:
        raise SystemExit(f"paid run requires --confirm {CONFIRM}")
    population = v1._population()
    items = [r for r in population[args.stage] if r["v18"] == 1]
    terminal = execute_stage(WORK, args.stage, items, _live_call, cap_usd=args.cap, resume=args.resume)
    if args.stage == "dev":
        rows = {json.loads(x)["id"]: json.loads(x) for x in (WORK / "rows_dev.jsonl").read_text(encoding="utf8").splitlines() if x.strip()}
        final = score_population(population["dev"], rows) if terminal["status"] == "COMPLETE" else {"status": "WITHHELD_INCOMPLETE"}
        (WORK / "final_dev.json").write_text(json.dumps(final, indent=2), encoding="utf8")
        print(json.dumps(final, indent=2))
    print(json.dumps(terminal, indent=2))
    return 0 if terminal["status"] == "COMPLETE" else 3


if __name__ == "__main__":
    sys.exit(main())

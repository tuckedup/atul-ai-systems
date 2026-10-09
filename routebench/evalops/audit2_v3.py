"""Second-pass auditor, execution version 3 (supersedes v2, which is preserved and was never executed with paid calls).

Same experiment (prompt, decision rule, populations, $3.00 total cap). What v3 repairs, found by testing the real client path:
- ONE journal reservation = exactly ONE HTTP request. v2 called `llm.chat()`, which retries internally (up to settings.max_retries), so one reservation
  could cover several billed requests. v3 calls the client's single-request function `llm._call` directly; every retry is an outer, journaled attempt.
- The input-token bound is a real bound: a BPE tokenizer never emits more tokens than UTF-8 bytes, so tokens <= bytes(prompt) + a fixed message overhead.
  (v2's len//2 + 200 was an estimate; dense or non-ASCII text can exceed it.) The output side is capped by the API's max_completion_tokens
  (reasoning tokens included), so reservation = bound_in * input_price + MAX_TOKENS * output_price is a ceiling for one request.
- Provider usage is read from the RAW response and validated. Missing or malformed usage charges the full reservation (never zero tokens / zero cost);
  usage above the reserved token bounds aborts the run.
Population binding, durable journal, type-validated parsing and withheld-κ reporting are reused from v2 (`audit2_v2`) and re-hashed in the v3 freeze.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import audit2 as v1
from . import audit2_v2 as v2
from .audit2 import AuditParseError, build_prompt, decide
from .audit2_v2 import (
    FreezeError,
    Journal,
    SpendError,
    check_items,
    file_sha,
    parse,
    population_digest,
    score_population,
    score_stage,
)

ROOT = v2.ROOT
DATA = v2.DATA
WORK = ROOT / ".local" / "audit2-v3"
CONFIRM = v2.CONFIRM
MODEL, MAX_TOKENS, MAX_ATTEMPTS = v1.MODEL, v1.MAX_TOKENS, v1.MAX_ATTEMPTS
TOTAL_CAP_USD = v2.TOTAL_CAP_USD
EXPECTED = v2.EXPECTED
MESSAGE_OVERHEAD_TOKENS = 64  # chat framing per request, deliberately generous
BACKOFF_SECONDS = 2.0


class ModelMismatch(SpendError):
    """The provider served a different model than requested; the run stops (this audit would not be the frozen experiment)."""


# ---------------------------------------------------------------------------------------------------- bounds and usage
def prompt_tokens_bound(prompt: str) -> int:
    """Guaranteed ceiling for byte-level BPE tokenizers: every token covers at least one UTF-8 byte."""
    return len(prompt.encode("utf8")) + MESSAGE_OVERHEAD_TOKENS


def usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * v1.INPUT_PER_1M + completion_tokens * v1.OUTPUT_PER_1M) / 1e6


def worst_case_usd(prompt: str) -> float:
    return usd(prompt_tokens_bound(prompt), MAX_TOKENS)


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def validated_usage(raw_usage: Any) -> tuple[int, int] | None:
    """(prompt_tokens, completion_tokens) from the RAW provider usage, or None if missing or malformed. Never defaults to zero."""
    if not isinstance(raw_usage, Mapping):
        return None
    prompt, completion = _count(raw_usage.get("prompt_tokens")), _count(raw_usage.get("completion_tokens"))
    if prompt is None or completion is None or prompt == 0 or completion == 0:
        return None  # a billed reasoning call cannot have zero prompt or completion tokens
    return prompt, completion


Attempt = tuple[str, Any, str]  # (text, raw usage mapping, served model) for ONE HTTP request


def http_attempt(prompt: str) -> Attempt:
    """Exactly one HTTP request through the project's client (`llm._call`); no client-side retries or model fallback happen here."""
    from aisys import llm

    result = llm._call(MODEL, [{"role": "user", "content": prompt}], None, 0.0, MAX_TOKENS, {"reasoning_effort": "medium"})
    usage = result.raw.get("usage") if isinstance(result.raw, dict) else None
    return result.text, usage, result.model or ""


def _pricing_matches() -> bool:
    from aisys import llm

    price = llm._PRICING.get(MODEL, {})
    return price.get("input_per_1m") == v1.INPUT_PER_1M and price.get("output_per_1m") == v1.OUTPUT_PER_1M


# --------------------------------------------------------------------------------------------------------- auditing
def audit_one(item: Mapping[str, str], attempt_fn: Callable[[str], Attempt], journal: Journal,
              sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """One case. Each loop iteration is one journaled HTTP request. Failures are recorded, never flipped. SpendError propagates."""
    from aisys import llm

    prompt, last = build_prompt(item["source"], item["candidate"]), ""
    bound_in, reserve = prompt_tokens_bound(prompt), worst_case_usd(prompt)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            sleep(BACKOFF_SECONDS * (attempt - 1))
        rid = journal.reserve(item["id"], attempt, reserve)
        try:
            text, raw_usage, served = attempt_fn(prompt)
        except llm.QuotaExceededError as exc:
            journal.settle(rid, reserve, None)  # account state unknown: charge the bound and stop
            raise SpendError(f"provider quota error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - a failed request may still have billed: charge the bound
            journal.settle(rid, reserve, None)
            last = f"{type(exc).__name__}: {exc}"
            continue
        usage = validated_usage(raw_usage)
        if usage is None:
            journal.settle(rid, reserve, None)  # missing/malformed usage: the full reservation, never zero
            last = "provider usage missing or malformed (charged at the reservation)"
            if not text:
                continue
        else:
            prompt_tokens, completion_tokens = usage
            cost = usd(prompt_tokens, completion_tokens)
            if prompt_tokens > bound_in or completion_tokens > MAX_TOKENS:  # the reserved token ceilings were exceeded: record, then stop
                journal.settle(rid, reserve, max(cost, reserve * (1 + 1e-9)))
            journal.settle(rid, reserve, cost)
        if served and not served.startswith(MODEL):
            raise ModelMismatch(f"served model {served!r} is not {MODEL!r}")
        try:
            result = parse(text)
            flip, reason = decide(result, item["source"], item["candidate"])
        except (AuditParseError, TypeError, AttributeError) as exc:
            last = f"parse: {exc}"
            continue
        return {"id": item["id"], "status": "ok", "attempts": attempt, "flip": flip, "reason": reason,
                "usage_verified": usage is not None, "result": result.__dict__ | {"unsupported_details": list(result.unsupported_details)}}
    return {"id": item["id"], "status": "failed", "attempts": MAX_ATTEMPTS, "flip": False, "reason": last}


# ------------------------------------------------------------------------------------------------------------ freeze
def code_and_data_hashes() -> dict[str, str]:
    out = {"audit2.py": file_sha(Path(v1.__file__)), "audit2_v2.py": file_sha(Path(v2.__file__)), "audit2_v3.py": file_sha(Path(__file__))}
    out.update({f"data/{name}": file_sha(DATA / name) for name in v2.DATA_FILES})
    out["settings"] = v2._sha([MODEL, MAX_TOKENS, MAX_ATTEMPTS, TOTAL_CAP_USD, "reasoning_effort=medium",
                               f"tokens=utf8_bytes+{MESSAGE_OVERHEAD_TOKENS}", "single_http_attempt=llm._call"])
    return out


def prepare(work: Path = WORK) -> dict[str, Any]:
    if not _pricing_matches():
        raise FreezeError("aisys pricing for the model differs from the constants this protocol was costed with")
    population = v1._population()
    audited = {k: [r for r in v if r["v18"] == 1] for k, v in population.items()}
    for stage, items in audited.items():
        check_items(items, EXPECTED[stage])
    reserves = [worst_case_usd(build_prompt(r["source"], r["candidate"])) for items in audited.values() for r in items]
    body = {"protocol": "audit2-v3", "supersedes": "audit2-v2 (never executed with paid calls; .local/audit2-v2 preserved)",
            "hashes": code_and_data_hashes(), "cap_usd_total": TOTAL_CAP_USD, "stages": EXPECTED, "stage_order": ["train", "dev"],
            "population_digest": {k: population_digest(v) for k, v in audited.items()},
            "cost_plan": v1.plan_cost(sum(EXPECTED.values())),
            "reservation_usd": {"min": round(min(reserves), 5), "max": round(max(reserves), 5), "mean": round(sum(reserves) / len(reserves), 5)}}
    work.mkdir(parents=True, exist_ok=False)
    (work / "protocol.json").write_text(json.dumps(body, indent=2), encoding="utf8")
    (work / "population.json").write_text(json.dumps(audited), encoding="utf8")
    return body


def verify_freeze(work: Path, stage: str, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    protocol = json.loads((work / "protocol.json").read_text(encoding="utf8"))
    if protocol["hashes"] != code_and_data_hashes():
        raise FreezeError("code, data files or settings changed since the freeze")
    check_items(items, protocol["stages"][stage])
    if population_digest(items) != protocol["population_digest"][stage]:
        raise FreezeError(f"{stage} inputs differ from the frozen population (content digest mismatch)")
    return protocol


# ---------------------------------------------------------------------------------------------------------- running
def execute_stage(work: Path, stage: str, items: Sequence[Mapping[str, Any]], attempt_fn: Callable[[str], Attempt], *,
                  cap_usd: float = TOTAL_CAP_USD, resume: bool = False, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
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
            if line.strip() and json.loads(line)["status"] == "ok":
                rows[json.loads(line)["id"]] = json.loads(line)
    journal = Journal(work / "spend_journal.jsonl", cap)
    status = "COMPLETE"
    with rows_path.open("a", encoding="utf8") as stream:
        for item in items:
            if item["id"] in rows:
                continue
            try:
                row = audit_one(item, attempt_fn, journal, sleep)
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
                "flips": sum(bool(r["flip"]) for r in rows.values()), "usage_unverified": sum(1 for r in rows.values() if not r.get("usage_verified", True)),
                "reasons": {k: sum(1 for r in rows.values() if r["reason"] == k) for k in sorted({r["reason"] for r in rows.values()})},
                "report": report}
    terminal_path.write_text(json.dumps(terminal, indent=2), encoding="utf8")
    return terminal


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
    terminal = execute_stage(WORK, args.stage, items, http_attempt, cap_usd=args.cap, resume=args.resume)
    if args.stage == "dev":
        rows = {json.loads(x)["id"]: json.loads(x) for x in (WORK / "rows_dev.jsonl").read_text(encoding="utf8").splitlines() if x.strip()}
        final = score_population(population["dev"], rows) if terminal["status"] == "COMPLETE" else {"status": "WITHHELD_INCOMPLETE"}
        (WORK / "final_dev.json").write_text(json.dumps(final, indent=2), encoding="utf8")
        print(json.dumps(final, indent=2))
    print(json.dumps(terminal, indent=2))
    return 0 if terminal["status"] == "COMPLETE" else 3


if __name__ == "__main__":
    sys.exit(main())

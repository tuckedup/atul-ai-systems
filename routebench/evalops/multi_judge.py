"""Exploratory multi-judge runner for RouteBench TRAIN and dev (never the sealed test split).

Same rubric prompt and same decision rule as the frozen v18 (`judge.build_prompt` / `judge.parse`, pass iff every criterion passes); only the judge model changes.
One journaled HTTP request per reservation (fsynced spend journal, hard cap shared across runs), per-model prices, usage validated from the RAW response.
Every judgment is stored with its criterion verdicts so later analysis does not need a re-run. Results are DEVELOPMENT evidence only (selection-optimistic).

    python -m evalops.multi_judge run  --model gpt-5.4 --effort medium --splits dev,train --cap 8 --tag g54 [--limit N]
    python -m evalops.multi_judge show --tag g54
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import audit2_v3 as a3
from . import judge, rubrics
from .audit2_v2 import Journal, SpendError
from .dataset import HEADLINE_PROVENANCE, Annotation, CaseRecord, Corpus
from .run_calibration import GRID

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).resolve().parents[2] / ".local" / "multi-judge"
PRICES = {"o4-mini": (1.10, 4.40), "o3": (2.0, 8.0), "gpt-5.4": (2.5, 15.0), "gpt-5.5": (5.0, 30.0), "gpt-5": (1.25, 10.0), "gpt-5-mini": (0.25, 2.0)}
MAX_TOKENS = 8000
BACKOFF = 2.0


def usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICES[model]
    return (prompt_tokens * pin + completion_tokens * pout) / 1e6


_CASE_ID = re.compile(rb'"case_id"\s*:\s*"([^"\\]+)"')
_GROUP_ID = re.compile(rb'"group_id"\s*:\s*"([^"\\]+)"')


def load_cases(splits: list[str]) -> list[tuple[CaseRecord, int, str]]:
    """Cases and resolved labels for the REQUESTED non-test splits only.

    Held-out rows are excluded by identity BEFORE any typed parsing or label resolution: case rows are matched to their split from the raw group id, and an annotation row is parsed
    only if its case id belongs to a requested split. (Before 2026-10-04 this function validated every annotation, held-out ones included, and resolved all labels before filtering
    them out; no held-out value was displayed, stored or used.)"""
    if "test" in splits or not splits:
        raise ValueError("the sealed test split is not available to this tool")
    assignment = json.loads((DATA / "splits.json").read_text(encoding="utf8"))["assignment"]
    cases = []
    for line in (DATA / "cases.jsonl").read_bytes().splitlines():
        if not line.strip():
            continue
        group = _GROUP_ID.search(line)
        if group is None or assignment[group.group(1).decode("utf8")] not in splits:
            continue
        cases.append(CaseRecord.model_validate_json(line))
    allowed = {c.case_id for c in cases}
    annotations = []
    for line in (DATA / "annotations.jsonl").read_bytes().splitlines():
        if not line.strip():
            continue
        cid = _CASE_ID.search(line)
        if cid is not None and cid.group(1).decode("utf8") in allowed:
            annotations.append(Annotation.model_validate_json(line))
    labels = Corpus(cases, annotations).resolved_labels(allow=HEADLINE_PROVENANCE)
    return [(c, int(labels[c.case_id].label), assignment[c.group_id]) for c in sorted(cases, key=lambda c: c.case_id) if c.case_id in labels]


def config_for(model: str, effort: str | None, tag: str):
    base = next(c for c in GRID if c.variant_id == "v18-reasoning-o4mini")
    return dataclasses.replace(base, variant_id=f"mj-{tag}", model=model, reasoning_effort=effort, notes="exploratory multi-judge; v18 rubric and prompt")


def single_request(model: str, effort: str | None, messages: list[dict[str, str]]):
    from aisys import llm

    extra = {"reasoning_effort": effort} if effort else {}
    result = llm._call(model, messages, None, 0.0, MAX_TOKENS, extra)
    return result.text, (result.raw.get("usage") if isinstance(result.raw, dict) else None), result.model or ""


def judge_case(case: CaseRecord, cfg, journal: Journal, request=single_request, sleep=time.sleep) -> dict[str, Any]:
    from aisys import llm

    rubric = rubrics.for_task(case.task_class, cfg.rubric_dir)
    prompt = judge.build_prompt(case, rubric, cfg)
    messages, last = [{"role": "user", "content": prompt}], ""
    for attempt in range(1, cfg.max_attempts + 1):
        if attempt > 1:
            sleep(BACKOFF * (attempt - 1))
        bound_in = sum(len(m["content"].encode("utf8")) for m in messages) + a3.MESSAGE_OVERHEAD_TOKENS
        reserve = usd(cfg.model, bound_in, MAX_TOKENS)
        rid = journal.reserve(f"{cfg.variant_id}:{case.case_id}", attempt, reserve)
        try:
            text, raw_usage, served = request(cfg.model, cfg.reasoning_effort, messages)
        except llm.QuotaExceededError as exc:
            journal.settle(rid, reserve, None)
            raise SpendError(f"provider quota error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - a failed request may still have billed: charge the bound
            journal.settle(rid, reserve, None)
            last = f"{type(exc).__name__}: {exc}"
            continue
        usage = a3.validated_usage(raw_usage)
        if usage is None:
            journal.settle(rid, reserve, None)
        else:
            cost = usd(cfg.model, *usage)
            journal.settle(rid, reserve, max(cost, reserve * (1 + 1e-9)) if usage[0] > bound_in or usage[1] > MAX_TOKENS else cost)
        if served and not served.startswith(cfg.model):
            raise SpendError(f"served model {served!r} is not {cfg.model!r}")
        try:
            parsed = judge.parse(text, rubric, cfg, case)
        except (judge.JudgeOutputError, rubrics.RubricError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": text[:2000]},
                        {"role": "user", "content": f"That response was rejected by the parser: {exc}\nReturn ONLY the JSON object in the exact shape requested, "
                                                    "with no prose and no code fence."}]
            continue
        return {"id": case.case_id, "status": "ok", "score": float(parsed.score), "pass": int(parsed.score >= 1.0), "attempts": attempt,
                "criteria": parsed.criteria_detail, "rationale": parsed.rationale, "usage": list(usage) if usage else None, "served": served}
    return {"id": case.case_id, "status": "failed", "reason": last}


def _latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Use the same last-record-per-case view for resuming, coverage and reporting."""
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def run(model: str, effort: str | None, splits: list[str], cap: float, tag: str, limit: int | None = None, request=single_request) -> dict[str, Any]:
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    population = load_cases(splits)
    expected_ids = {c.case_id for c, _, _ in population}
    if not population:
        raise ValueError("empty evaluation population")
    if len(expected_ids) != len(population):
        raise ValueError("duplicate case IDs in evaluation population")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config_for(model, effort, tag)
    journal = Journal(OUT / "spend_journal.jsonl", cap)  # one journal: the cap covers every run in this directory
    path = OUT / f"rows_{tag}.jsonl"
    rows = _latest_rows(path)
    done = {cid for cid, row in rows.items() if row["status"] == "ok"}
    todo = [(c, y, s) for c, y, s in population if c.case_id not in done][:limit]
    attempted, completed_this_run = 0, 0
    status = "COMPLETE"
    with path.open("a", encoding="utf8") as f:
        for case, _y, _s in todo:
            attempted += 1
            try:
                row = judge_case(case, cfg, journal, request)
            except SpendError as exc:
                status = f"ABORTED: {exc}"
                break
            f.write(json.dumps(row) + chr(10))
            f.flush()
            os.fsync(f.fileno())
            rows[row["id"]] = row
            completed_this_run += int(row["status"] == "ok")
    completed_ids = {cid for cid, row in rows.items() if row["status"] == "ok"}
    completed_requested = len(completed_ids & expected_ids)
    if status == "COMPLETE" and completed_requested != len(expected_ids):
        status = "INCOMPLETE: requested population has missing or failed judgments"
    return {"tag": tag, "model": model, "status": status, "planned": len(todo), "expected": len(expected_ids),
            "attempted_this_run": attempted, "completed_ok_this_run": completed_this_run,
            "completed_ok_requested": completed_requested, "completed_ok_total_in_file": len(completed_ids),
            "spent_total_all_runs": round(journal.committed(), 4), "cap": cap}


def show(tag: str, splits: list[str]) -> dict[str, Any]:
    from .audit2_v2 import kappa

    rows = _latest_rows(OUT / f"rows_{tag}.jsonl")
    out: dict[str, Any] = {}
    for split in splits:
        cases = [(c, y) for c, y, s in load_cases([split])]
        ok = [(c, y) for c, y in cases if rows.get(c.case_id, {}).get("status") == "ok"]
        if not cases or len(ok) != len(cases):
            out[split] = {"coverage": f"{len(ok)}/{len(cases)}", "kappa": None}
            continue
        ys, ps = [y for _, y in ok], [rows[c.case_id]["pass"] for c, _ in ok]
        fp = sum(1 for y, p in zip(ys, ps, strict=True) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(ys, ps, strict=True) if y == 1 and p == 0)
        out[split] = {"coverage": f"{len(ok)}/{len(cases)}", "kappa": kappa(ys, ps), "false_passes": fp, "false_rejections": fn}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "show"])
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--splits", default="dev,train")
    parser.add_argument("--cap", type=float, default=8.0)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    splits = args.splits.split(",")
    if args.command == "run":
        result = run(args.model, args.effort or None, splits, args.cap, args.tag, args.limit)
        print(json.dumps(result, indent=1))
        return 0 if result["status"] == "COMPLETE" else 3
    else:
        print(json.dumps(show(args.tag, splits), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

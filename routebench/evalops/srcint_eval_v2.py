"""`srcint-eval-v2`: the same frozen fresh-judgment comparison of v18 on the source-integrity-v2 repaired copy as v1, with the execution/reporting safeguards found missing by the
independent review (docs/ROUTEBENCH_SRCINT_EVAL_INDEPENDENT_REVIEW_20261004.md). v1 (code, freeze `a52d3716...`) is preserved unchanged and was never run. Nothing here has been run
against a provider.

What v2 adds, all fail-closed:
  * every saved result row is validated before resume or comparison: exact expected case ids, no duplicate or foreign ids, protocol/prompt/config/rubric bindings, finite non-boolean
    score in [0, 1], the stored raw response re-parses to exactly the stored score and criteria, a non-empty served model that matches, and attempt ids that exist, belong to the case
    and are settled in the spend journal;
  * the journal must be consistent: no unsettled reservation (an operator may settle one explicitly AT ITS BOUND, never discard it), committed spend never above the cap (an over-cap
    run is permanently refused), and COMPLETE requires validated rows + a clean journal + durable output - ID coverage alone is not enough;
  * a terminal ABORTED/INCOMPLETE state needs an explicit --resume; a COMPLETE state is final; a lock file permits one active runner;
  * the freeze binds the real runtime dependencies (audit2_v3 token bound and usage validator, the bounded loader, dataset schema, calibration grid);
  * held-out cases are excluded by identity before any annotation is parsed (bounded loader), and labels/baseline are parsed only after all of the above;
  * no p-value or causal claim: verdict flips on changed vs unchanged inputs are DESCRIPTIVE counts only (changed cases come from different documents and datasets).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import audit2_v3 as a3
from . import matched_v16_v18 as mv
from . import srcint_eval as v1
from .audit2_v2 import Journal, SpendError, kappa
from .dataset import CaseRecord

ROOT = v1.ROOT
SRC = v1.SRC
WORK = ROOT / ".local" / "srcint-eval-v2"
VARIANT, DATASET, CAP_USD = v1.VARIANT, v1.DATASET, v1.CAP_USD
CONFIRM = "RUN-SRCINT-OPENAI-V2"
PROTOCOL_NAME = "srcint-eval-v2"
V1_PROTOCOL_SHA256 = "a52d3716d5b6ba71559302a0f0b196eca50725b30bf6f78f1bd7d0416e185c29"
BOOT_SEED, BOOT_DRAWS = v1.BOOT_SEED, v1.BOOT_DRAWS
ROW_KEYS = {"case_id", "protocol_sha256", "record"}


class SrcIntV2Error(ValueError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    from . import dataset, multi_judge, run_calibration

    extra = {"audit2_v3.py": _sha(Path(a3.__file__)), "multi_judge.py": _sha(Path(multi_judge.__file__)), "dataset.py": _sha(Path(dataset.__file__)),
             "run_calibration.py": _sha(Path(run_calibration.__file__)), "srcint_eval.py": _sha(Path(v1.__file__)), "srcint_eval_v2.py": _sha(Path(__file__))}
    return mv.code_hashes() | extra


# ---------------------------------------------------------------------------------------------------------------------------- freeze
def freeze(work: Path = WORK) -> dict[str, Any]:
    work = Path(work)
    if (work / "protocol.json").exists():
        raise SrcIntV2Error("protocol exists; a frozen protocol is never overwritten")
    items, cases, labels, baseline = v1.build_population()  # the bounded loader excludes held-out cases before any annotation is parsed
    work.mkdir(parents=True, exist_ok=True)
    (work / "cases.jsonl").write_text("".join(cases[i["case_id"]].model_dump_json() + "\n" for i in items), encoding="utf8")
    (work / "population.json").write_text(json.dumps(items, sort_keys=True), encoding="utf8")
    (work / "labels.json").write_text(json.dumps(labels, sort_keys=True), encoding="utf8")  # parsed only by `compare`, after all validation
    (work / "baseline_cached_v18.json").write_text(json.dumps(baseline, sort_keys=True), encoding="utf8")
    v1_protocol = json.loads((v1.WORK / "protocol.json").read_text(encoding="utf8")) if (v1.WORK / "protocol.json").exists() else None
    if v1_protocol is not None and (mv._digest(items) != v1_protocol["population_digest"] or _sha(v1.WORK / "protocol.json") != V1_PROTOCOL_SHA256):
        raise SrcIntV2Error("the v2 population differs from the preserved v1 population")
    cfg = mv.cfg_of(VARIANT)
    expected = round(sum((len(cases[i["case_id"]].context) / 4 + 900) * 1.1 + 1263 * 4.4 for i in items) / 1e6, 2)
    body = {"protocol": PROTOCOL_NAME, "supersedes": f"srcint-eval-v1 {V1_PROTOCOL_SHA256} (never run; preserved unchanged)", "dataset": DATASET,
            "repair_manifest_sha256": _sha(SRC / "manifest.json"), "repaired_cases_sha256": v1.SRC_CASES_SHA,
            "judge": {"variant": VARIANT, "config_hash": cfg.config_hash, "model": cfg.model, "max_tokens": cfg.max_tokens, "reasoning_effort": cfg.reasoning_effort,
                      "prompt_template_hash": mv.judge.prompt_template_hash(cfg.mode)}, "rubric_hash": mv.rubrics.for_task("summarize").hash,
            "decision_rule": {"pass_iff_score_at_least": mv.PASS_THRESHOLD, "thresholds_searched": False},
            "population": {"cases": len(items), "dev": sum(i["pop"] == "dev" for i in items), "train": sum(i["pop"] == "train" for i in items),
                           "source_changed": sum(i["source_changed"] for i in items), "unchanged_inputs": sum(not i["source_changed"] for i in items), "removed_cases": 0},
            "population_digest": mv._digest(items),
            "input_sha256": {n: _sha(work / n) for n in ("cases.jsonl", "population.json", "labels.json", "baseline_cached_v18.json")},
            "fresh_judgments": "all 257 repaired cases judged fresh in an isolated result store; cached verdicts are only the comparison baseline",
            "safeguards": ["row validation with raw-response re-parse", "journal consistency and over-cap refusal", "terminal-state resume rules", "single-runner lock", "bounded TRAIN/DEV loader",
                           "dependencies frozen", "no p-value or causal claim"],
            "bootstrap": {"seed": BOOT_SEED, "draws": BOOT_DRAWS, "unit": "document, paired, within dev"}, "cap_usd": CAP_USD, "price_usd_per_1m": mv.PRICES["o4-mini"],
            "estimate": {"calls": len(items), "expected_usd": expected, "basis": "prompt tokens ~ source chars/4 + 900 framing, completion 1,263 tokens (mean of cached v18 calls); an estimate, not a bound",
                         "hard_bound": "fsynced spend journal; one HTTP request per reservation; cap enforced; over-cap runs are refused permanently"},
            "code": code_hashes(), "reporting": ["dev primary", "train diagnostic", "descriptive verdict-flip counts (no test, no causal claim)", "false passes and false rejections", "accuracy and kappa"]}
    (work / "protocol.json").write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf8")
    return body


def verify(work: Path, expected_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, CaseRecord]]:
    work = Path(work)
    if _sha(work / "protocol.json") != expected_sha256:
        raise SrcIntV2Error("protocol hash differs from the hash recorded at freeze time")
    frozen = json.loads((work / "protocol.json").read_text(encoding="utf8"))
    for n, d in frozen["input_sha256"].items():
        if _sha(work / n) != d:
            raise SrcIntV2Error(f"input file {n} changed after the freeze")
    if frozen["code"] != code_hashes():
        raise SrcIntV2Error("code, dependency, rubric or pricing changed after the freeze")
    if _sha(SRC / "cases.jsonl") != frozen["repaired_cases_sha256"] or _sha(SRC / "manifest.json") != frozen["repair_manifest_sha256"]:
        raise SrcIntV2Error("the repaired copy changed after the freeze")
    items = json.loads((work / "population.json").read_text(encoding="utf8"))
    cases = {}
    for line in (work / "cases.jsonl").read_text(encoding="utf8").splitlines():
        c = CaseRecord.model_validate_json(line)
        cases[c.case_id] = c
    if mv._digest(items) != frozen["population_digest"] or {i["case_id"] for i in items} != set(cases):
        raise SrcIntV2Error("population differs from the frozen population")
    cfg = mv.cfg_of(VARIANT)
    if frozen["judge"]["config_hash"] != cfg.config_hash or frozen["judge"]["prompt_template_hash"] != mv.judge.prompt_template_hash(cfg.mode):
        raise SrcIntV2Error("judge identity changed")
    return frozen, items, cases


# ---------------------------------------------------------------------------------------------------------------------------- journal
def journal_state(path: Path, cap: float) -> dict[str, Any]:
    """Strict read of the spend journal. Never repairs or discards anything."""
    reserves: dict[str, dict[str, Any]] = {}
    settled: dict[str, float] = {}
    p = Path(path)
    for n, line in enumerate(p.read_text(encoding="utf8").splitlines() if p.exists() else [], start=1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SrcIntV2Error(f"spend journal line {n} is unreadable") from exc
        if e.get("op") == "reserve" and isinstance(e.get("id"), str) and isinstance(e.get("usd"), (int, float)) and e["id"] not in reserves:
            reserves[e["id"]] = e
        elif e.get("op") == "settle" and e.get("id") in reserves and e["id"] not in settled and isinstance(e.get("charged"), (int, float)) and math.isfinite(e["charged"]) and e["charged"] >= 0:
            settled[e["id"]] = float(e["charged"])
        else:
            raise SrcIntV2Error(f"spend journal line {n} is inconsistent (duplicate, unknown or malformed entry)")
    unsettled = sorted(set(reserves) - set(settled))
    committed = sum(settled.values()) + sum(float(reserves[i]["usd"]) for i in unsettled)
    return {"committed": committed, "unsettled": unsettled, "settled": set(settled), "reserves": reserves, "over_cap": committed > cap + 1e-9}


def settle_unsettled_at_bound(work: Path, cap: float) -> list[str]:
    """Explicit operator action: charge each unsettled reservation at its bound (the ledger keeps the charge; nothing is discarded)."""
    st = journal_state(Path(work) / "spend_journal.jsonl", cap)
    j = Journal(Path(work) / "spend_journal.jsonl", cap)
    for rid in st["unsettled"]:
        j.settle(rid, float(st["reserves"][rid]["usd"]), None)
    return st["unsettled"]


# ---------------------------------------------------------------------------------------------------------------------------- results
def validate_rows(work: Path, frozen: dict[str, Any], items: Sequence[dict[str, Any]], cases: dict[str, CaseRecord], state: dict[str, Any], sha: str) -> dict[str, dict[str, Any]]:
    """Return {case_id: record} for every saved row, or raise. Duplicate, foreign, unbound or unverifiable rows are refused, never overwritten or skipped."""
    path = Path(work) / "results.jsonl"
    if not path.exists():
        return {}
    cfg = mv.cfg_of(VARIANT)
    rub = mv.rubrics.for_task("summarize", cfg.rubric_dir)
    expected = {i["case_id"] for i in items}
    rows: dict[str, dict[str, Any]] = {}
    used_attempts: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SrcIntV2Error(f"results line {n} is unreadable") from exc
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise SrcIntV2Error(f"results line {n}: unexpected row shape")
        cid, rec = row["case_id"], row["record"]
        if cid not in expected:
            raise SrcIntV2Error(f"results line {n}: foreign case id {cid!r}")
        if cid in rows:
            raise SrcIntV2Error(f"results line {n}: duplicate result for {cid} (conflicting duplicates are never merged)")
        if row["protocol_sha256"] != sha or not isinstance(rec, dict):
            raise SrcIntV2Error(f"results line {n}: row is not bound to this protocol")
        ids = rec.get("attempt_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(a, str) for a in ids):
            raise SrcIntV2Error(f"results line {n}: missing attempt ids")
        for a in ids:
            r = state["reserves"].get(a)
            if r is None or a not in state["settled"] or r.get("case") != f"{VARIANT}:{cid}" or a in used_attempts:
                raise SrcIntV2Error(f"results line {n}: attempt {a} is not a settled reservation of this case")
            used_attempts.add(a)
        prompt = mv.judge.build_prompt(cases[cid], rub, cfg)
        if rec.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf8")).hexdigest() or rec.get("config_hash") != cfg.config_hash or rec.get("rubric_hash") != rub.hash:
            raise SrcIntV2Error(f"results line {n}: prompt/config/rubric binding does not match")
        if rec.get("status") == "failed":
            if not isinstance(rec.get("reason"), str):
                raise SrcIntV2Error(f"results line {n}: failed row without a reason")
        elif rec.get("status") == "ok":
            score = rec.get("score")
            served = rec.get("model_served")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise SrcIntV2Error(f"results line {n}: invalid score {score!r}")
            if not isinstance(served, str) or not served or not served.startswith(cfg.model):
                raise SrcIntV2Error(f"results line {n}: served model identity missing or wrong")
            raw = rec.get("raw_response")
            if not isinstance(raw, str) or not raw:
                raise SrcIntV2Error(f"results line {n}: raw response not retained")
            try:
                parsed = mv.judge.parse(raw, rub, cfg, cases[cid])
            except (mv.judge.JudgeOutputError, mv.rubrics.RubricError) as exc:
                raise SrcIntV2Error(f"results line {n}: stored raw response does not parse ({exc})") from exc
            if abs(float(parsed.score) - float(score)) > 1e-9 or parsed.criteria_detail != rec.get("criteria"):
                raise SrcIntV2Error(f"results line {n}: stored score/criteria differ from the stored raw response")
        else:
            raise SrcIntV2Error(f"results line {n}: unknown status {rec.get('status')!r}")
        rows[cid] = rec
    return rows


def judge_case(case: CaseRecord, journal: Journal, request: Callable[..., Any], sha: str) -> dict[str, Any]:
    """The frozen recipe (judge.build_prompt / judge.parse, registered config); one journaled HTTP request per attempt; the raw response and attempt ids are retained."""
    from aisys import llm

    cfg = mv.cfg_of(VARIANT)
    pin, pout = mv.PRICES[cfg.model]
    rub = mv.rubrics.for_task(case.task_class, cfg.rubric_dir)
    prompt = mv.judge.build_prompt(case, rub, cfg)
    base = {"prompt_sha256": hashlib.sha256(prompt.encode("utf8")).hexdigest(), "config_hash": cfg.config_hash, "rubric_hash": rub.hash}
    messages, last, ids = [{"role": "user", "content": prompt}], "", []
    for attempt in range(1, cfg.max_attempts + 1):
        bound_in = sum(len(m["content"].encode("utf8")) for m in messages) + a3.MESSAGE_OVERHEAD_TOKENS
        reserve = (bound_in * pin + cfg.max_tokens * pout) / 1e6
        rid = journal.reserve(f"{VARIANT}:{case.case_id}", attempt, reserve)
        ids.append(rid)
        try:
            text, raw_usage, served = request(cfg.model, cfg.reasoning_effort, cfg.temperature, cfg.max_tokens, messages)
        except llm.QuotaExceededError as exc:
            journal.settle(rid, reserve, None)
            raise SpendError(f"provider quota error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - a failed request may still have billed
            journal.settle(rid, reserve, None)
            last = f"{type(exc).__name__}: {exc}"
            continue
        usage = a3.validated_usage(raw_usage)
        if usage is None:
            journal.settle(rid, reserve, None)
        else:
            cost = (usage[0] * pin + usage[1] * pout) / 1e6
            journal.settle(rid, reserve, max(cost, reserve * (1 + 1e-9)) if usage[0] > bound_in or usage[1] > cfg.max_tokens else cost)
        if not isinstance(served, str) or not served:
            return base | {"status": "failed", "reason": "served model identity missing; judgment unverifiable", "attempt_ids": ids}
        if not served.startswith(cfg.model):
            raise SpendError(f"served model {served!r} is not {cfg.model!r}")
        try:
            parsed = mv.judge.parse(text, rub, cfg, case)
        except (mv.judge.JudgeOutputError, mv.rubrics.RubricError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": text[:2000]},
                        {"role": "user", "content": f"That response was rejected by the parser: {exc}\nReturn ONLY the JSON object in the exact shape requested, with no prose and no code fence."}]
            continue
        return base | {"status": "ok", "score": float(parsed.score), "criteria": parsed.criteria_detail, "raw_response": text, "usage": list(usage) if usage else None,
                       "model_served": served, "attempts": attempt, "attempt_ids": ids}
    return base | {"status": "failed", "reason": last or "no usable response", "attempt_ids": ids}


def _write_terminal(work: Path, terminal: dict[str, Any]) -> None:
    tmp = Path(work) / "terminal.json.tmp"
    with tmp.open("w", encoding="utf8") as fh:
        fh.write(json.dumps(terminal, indent=2))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, Path(work) / "terminal.json")


def run(work: Path, expected_sha256: str, request: Callable[..., Any] = mv.http_request, *, resume: bool = False, settle_unsettled: bool = False) -> dict[str, Any]:
    frozen, items, cases = verify(work, expected_sha256)
    work = Path(work)
    cap = frozen["cap_usd"]
    term_path = work / "terminal.json"
    prior = json.loads(term_path.read_text(encoding="utf8")) if term_path.exists() else None
    if prior is not None:
        if prior.get("status") == "COMPLETE":
            raise SrcIntV2Error("a COMPLETE terminal record exists; the run is final and is never repeated")
        if prior.get("status") == "REFUSED_OVER_CAP":
            raise SrcIntV2Error("an over-cap run was recorded; it is permanently refused")
        if not resume:
            raise SrcIntV2Error(f"a {prior.get('status')} terminal record exists; resuming needs an explicit --resume")
    lock = work / "run.lock"
    try:
        with lock.open("x", encoding="utf8") as fh:
            fh.write(json.dumps({"pid": os.getpid()}))
    except FileExistsError as exc:
        raise SrcIntV2Error("another runner holds run.lock (or a crash left it): inspect and remove it manually") from exc
    try:
        if settle_unsettled:
            settle_unsettled_at_bound(work, cap)
        state = journal_state(work / "spend_journal.jsonl", cap)
        if state["over_cap"]:
            _write_terminal(work, {"status": "REFUSED_OVER_CAP", "committed_usd": round(state["committed"], 4), "cap_usd": cap, "protocol_sha256": expected_sha256})
            raise SrcIntV2Error(f"committed spend ${state['committed']:.4f} exceeds the ${cap:.2f} cap: refused")
        if state["unsettled"]:
            raise SrcIntV2Error(f"{len(state['unsettled'])} unsettled reservation(s) in the journal; settle them at their bound explicitly with --settle-unsettled before resuming")
        rows = validate_rows(work, frozen, items, cases, state, expected_sha256)
        journal = Journal(work / "spend_journal.jsonl", cap)
        status, stop = "COMPLETE", None
        with (work / "results.jsonl").open("a", encoding="utf8") as fh:
            for it in items:
                cid = it["case_id"]
                if cid in rows:
                    continue
                try:
                    rec = judge_case(cases[cid], journal, request, expected_sha256)
                except SpendError as exc:
                    status, stop = "ABORTED", str(exc)
                    break
                fh.write(json.dumps({"case_id": cid, "protocol_sha256": expected_sha256, "record": rec}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                rows[cid] = rec
        state = journal_state(work / "spend_journal.jsonl", cap)
        if state["over_cap"]:
            status, stop = "REFUSED_OVER_CAP", f"committed spend ${state['committed']:.4f} exceeds the ${cap:.2f} cap"
        elif status == "COMPLETE" and state["unsettled"]:
            status, stop = "ABORTED", "journal inconsistent after the run (unsettled reservation)"
        valid = validate_rows(work, frozen, items, cases, state, expected_sha256)  # re-read from disk: durable output, not memory
        failed = sorted(c for c, r in valid.items() if r["status"] != "ok")
        missing = [i["case_id"] for i in items if i["case_id"] not in valid]
        if status == "COMPLETE" and (failed or missing):
            status = "INCOMPLETE"
        t = {"status": status, "cases": len(items), "ok": sum(1 for r in valid.values() if r["status"] == "ok"), "failed": failed, "not_attempted": missing, "stopped_on": stop,
             "committed_usd": round(state["committed"], 4), "cap_usd": cap, "unsettled": state["unsettled"], "protocol_sha256": expected_sha256}
        _write_terminal(work, t)
        with (work / "invocations.jsonl").open("a", encoding="utf8") as fh:
            fh.write(json.dumps({"resume": resume, "status": status, "ok": t["ok"], "committed_usd": t["committed_usd"]}) + "\n")
        return t
    finally:
        lock.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------------------------------- compare
def compare(work: Path, expected_sha256: str) -> dict[str, Any]:
    frozen, items, cases = verify(work, expected_sha256)
    work = Path(work)
    cap = frozen["cap_usd"]
    # --- every integrity gate runs BEFORE labels or the cached baseline are parsed
    state = journal_state(work / "spend_journal.jsonl", cap)
    term = json.loads((work / "terminal.json").read_text(encoding="utf8")) if (work / "terminal.json").exists() else None
    problems = []
    if state["over_cap"]:
        problems.append("committed spend exceeds the cap")
    if state["unsettled"]:
        problems.append("unsettled reservations in the journal")
    if term is None or term.get("status") != "COMPLETE" or term.get("protocol_sha256") != expected_sha256:
        problems.append("no COMPLETE terminal record for this protocol")
    rows = validate_rows(work, frozen, items, cases, state, expected_sha256)
    missing = [i["case_id"] for i in items if i["case_id"] not in rows or rows[i["case_id"]]["status"] != "ok"]
    if missing:
        problems.append(f"{len(missing)} cases lack a valid judgment")
    if problems:
        return {"status": "WITHHELD_INCOMPLETE", "problems": problems, "valid": len(items) - len(missing), "expected": len(items), "note": "labels were not read; no metric exists"}
    target = work / "comparison.json"
    if target.exists():
        raise SrcIntV2Error("comparison.json exists; the comparison is written once")
    labels = json.loads((work / "labels.json").read_text(encoding="utf8"))
    base = json.loads((work / "baseline_cached_v18.json").read_text(encoding="utf8"))
    fresh = {c: int(r["score"] >= mv.PASS_THRESHOLD) for c, r in rows.items()}
    rng = random.Random(BOOT_SEED)

    def block(sel: Sequence[dict[str, Any]], boot: bool = False) -> dict[str, Any]:
        ids = [i["case_id"] for i in sel]
        y = [labels[c] for c in ids]
        out: dict[str, Any] = {"n": len(ids), "documents": len({i["group"] for i in sel})}
        for name, pred in (("cached_original_input", base), ("fresh_repaired_input", fresh)):
            p = [pred[c] for c in ids]
            out[name] = {"kappa": kappa(y, p) if len(set(y)) > 1 else None, "accuracy": sum(a == b for a, b in zip(y, p, strict=True)) / len(ids),
                         "false_passes": sum(1 for a, b in zip(y, p, strict=True) if a == 0 and b == 1), "false_rejections": sum(1 for a, b in zip(y, p, strict=True) if a == 1 and b == 0)}
        out["verdict_flips_vs_cached"] = sum(1 for c in ids if base[c] != fresh[c])
        out["flips_fixing_an_error"] = sum(1 for c, a in zip(ids, y, strict=True) if base[c] != a and fresh[c] == a)
        out["flips_creating_an_error"] = sum(1 for c, a in zip(ids, y, strict=True) if base[c] == a and fresh[c] != a)
        if boot and out["cached_original_input"]["kappa"] is not None:
            groups = sorted({i["group"] for i in sel})
            members = {g: [k for k, i in enumerate(sel) if i["group"] == g] for g in groups}
            diffs = []
            for _ in range(BOOT_DRAWS):
                idx = [k for g in (rng.choice(groups) for _ in groups) for k in members[g]]
                yy = [y[k] for k in idx]
                if len(set(yy)) < 2:
                    continue
                a, b = kappa(yy, [base[ids[k]] for k in idx]), kappa(yy, [fresh[ids[k]] for k in idx])
                if a is not None and b is not None:
                    diffs.append(b - a)
            diffs.sort()
            out["kappa_difference_fresh_minus_cached_95_document_bootstrap"] = [diffs[int(0.025 * len(diffs))], diffs[max(0, int(0.975 * len(diffs)) - 1)]] if diffs else None
        return out

    dev, train = [i for i in items if i["pop"] == "dev"], [i for i in items if i["pop"] == "train"]
    changed, unchanged = [i for i in items if i["source_changed"]], [i for i in items if not i["source_changed"]]
    flips = lambda sel: sum(1 for i in sel if base[i["case_id"]] != fresh[i["case_id"]])
    result = {"status": "COMPLETE", "dataset": DATASET, "label": "DEVELOPMENT result on REPAIRED inputs with fresh judgments; not the original benchmark, not held-out, not a pass of the 0.74 target",
              "dev_primary": block(dev, True), "train_diagnostic": block(train, True), "all_257_supplementary": block(items),
              "dev_subsets": {s: block([i for i in dev if i["subset"] == s], False) for s in sorted({i["subset"] for i in dev})},
              "descriptive_verdict_flips": {"changed_inputs": {"n": len(changed), "flips": flips(changed)}, "unchanged_inputs": {"n": len(unchanged), "flips": flips(unchanged)},
                                            "caveat": "COUNTS ONLY, no test and no causal reading: changed cases come from 7 documents (mostly MediaSum) while unchanged cases include other datasets and difficulty "
                                                      "profiles; sentences share documents; the allocation is not randomised; provider drift is possible. Unchanged-input flips are a reproducibility "
                                                      "diagnostic, not an exchangeable control group."},
              "interpretation": "the dev paired document-bootstrap interval describes the whole changed-input pipeline comparison; it does not isolate the effect of the punctuation repair",
              "removed_cases": 0}
    target.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["freeze", "run", "compare"])
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--protocol-sha256")
    p.add_argument("--confirm", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--settle-unsettled", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.command == "freeze":
            body = freeze(a.work)
            print(json.dumps({"protocol_sha256": _sha(a.work / "protocol.json"), **body}, indent=2))
        else:
            if not a.protocol_sha256:
                print("REFUSED: --protocol-sha256 is required", file=sys.stderr)
                return 2
            if a.command == "run":
                if a.confirm != CONFIRM:
                    print(f"REFUSED: needs --confirm {CONFIRM}", file=sys.stderr)
                    return 2
                t = run(a.work, a.protocol_sha256, resume=a.resume, settle_unsettled=a.settle_unsettled)
                print(json.dumps(t, indent=2))
                return 0 if t["status"] == "COMPLETE" else 3
            print(json.dumps(compare(a.work, a.protocol_sha256), indent=2))
    except (SrcIntV2Error, v1.SrcIntError, mv.MatchedError, SpendError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

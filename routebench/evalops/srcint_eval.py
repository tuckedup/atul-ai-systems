"""Frozen fresh-judgment evaluation of frozen v18 on the source-integrity-v2 REPAIRED copy of the 257 headline TRAIN/dev cases (dataset `routebench-srcint-v2`).

Nothing here replaces the original benchmark: the repaired copy has new case ids, an isolated result store and no shared cache; cached verdicts for the original inputs are used only as the
comparison BASELINE, never as judgments of the repaired inputs. Every one of the 257 retained cases is judged fresh (63 have repaired sources, 194 are byte-identical), so the 194 unchanged cases
measure the judge's own run-to-run noise and let the effect of the repair be told apart from it. No case is ever removed because the judge fails it. Labels sit in a separate file that is
parsed only after completeness is established. Dev (128) is primary; TRAIN (129) is diagnostic. A result on changed inputs is development evidence, not a pass of the original benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import matched_v16_v18 as mv
from .audit2_v2 import Journal, SpendError, kappa
from .dataset import CaseRecord
from .multi_judge import load_cases

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / ".local" / "source-integrity-v2-20261004"
WORK = ROOT / ".local" / "srcint-eval-v1"
VARIANT = "v18-reasoning-o4mini"
DATASET = "routebench-srcint-v2"
CAP_USD = 3.0
CONFIRM = "RUN-SRCINT-OPENAI"
BOOT_SEED, BOOT_DRAWS = 20261106, 2000
SRC_CASES_SHA = "e4057d79db40f9e0785ca2693094e294d40d1004c97b3271e865ce52a7f8c738"


class SrcIntError(ValueError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_population() -> tuple[list[dict[str, Any]], dict[str, CaseRecord], dict[str, int], dict[str, int]]:
    """(items, repaired cases by new id, labels by new id, cached baseline pass by new id). Labels and baseline are returned for the freeze files only."""
    if _sha(SRC / "cases.jsonl") != SRC_CASES_SHA:
        raise SrcIntError("repaired cases file differs from the recorded hash")
    manifest = json.loads((SRC / "manifest.json").read_text(encoding="utf8"))
    mapping = {m["case_id"]: m for m in manifest["case_mapping"]}
    if manifest.get("labels_read") or manifest.get("predictions_read") or manifest.get("model_calls"):
        raise SrcIntError("the repair manifest says it read labels/predictions or made model calls")
    originals = {c.case_id: (c, y, s) for c, y, s in load_cases(["train", "dev"])}
    items, cases, labels, baseline = [], {}, {}, {}
    for line in (SRC / "cases.jsonl").read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        new = CaseRecord.model_validate_json(line)
        m = mapping[new.case_id]
        orig, y, split = originals[m["original_case_id"]]
        if (new.group_id, split) != (m["group_id"], m["split"]) or new.candidate_output != orig.candidate_output or new.meta.get("subset") != orig.meta.get("subset"):
            raise SrcIntError(f"{new.case_id}: repaired record does not match its original (group, split, candidate or subset)")
        cached = mv.verified_cached(orig, VARIANT, split)
        if cached is None:
            raise SrcIntError(f"{orig.case_id}: no verified cached v18 baseline")
        changed = new.context != orig.context
        items.append({"case_id": new.case_id, "original_case_id": orig.case_id, "pop": split, "subset": new.meta.get("subset", "?"), "group": new.group_id, "source_changed": changed,
                      "source_sha256_repaired": hashlib.sha256(new.context.encode("utf8")).hexdigest(), "source_sha256_original": hashlib.sha256(orig.context.encode("utf8")).hexdigest()})
        cases[new.case_id], labels[new.case_id], baseline[new.case_id] = new, y, int(cached["score"] >= mv.PASS_THRESHOLD)
    if len(items) != 257 or len({i["case_id"] for i in items}) != 257:
        raise SrcIntError("expected 257 retained cases with unique ids")
    return items, cases, labels, baseline


def freeze(work: Path = WORK) -> dict[str, Any]:
    work = Path(work)
    if (work / "protocol.json").exists():
        raise SrcIntError("protocol exists; a frozen protocol is never overwritten")
    items, cases, labels, baseline = build_population()
    work.mkdir(parents=True, exist_ok=True)
    (work / "cases.jsonl").write_text("".join(cases[i["case_id"]].model_dump_json() + "\n" for i in items), encoding="utf8")
    (work / "population.json").write_text(json.dumps(items, sort_keys=True), encoding="utf8")
    (work / "labels.json").write_text(json.dumps(labels, sort_keys=True), encoding="utf8")  # parsed only by `compare`, after completeness
    (work / "baseline_cached_v18.json").write_text(json.dumps(baseline, sort_keys=True), encoding="utf8")
    cfg = mv.cfg_of(VARIANT)
    expected = round(sum((len(cases[i["case_id"]].context) / 4 + 900) * 1.1 + 1263 * 4.4 for i in items) / 1e6, 2)
    body = {"protocol": "srcint-eval-v1", "dataset": DATASET, "repair_manifest_sha256": _sha(SRC / "manifest.json"), "repaired_cases_sha256": SRC_CASES_SHA,
            "judge": {"variant": VARIANT, "config_hash": cfg.config_hash, "model": cfg.model, "max_tokens": cfg.max_tokens, "reasoning_effort": cfg.reasoning_effort,
                      "prompt_template_hash": mv.judge.prompt_template_hash(cfg.mode)}, "rubric_hash": mv.rubrics.for_task("summarize").hash,
            "decision_rule": {"pass_iff_score_at_least": mv.PASS_THRESHOLD, "thresholds_searched": False},
            "population": {"cases": len(items), "dev": sum(i["pop"] == "dev" for i in items), "train": sum(i["pop"] == "train" for i in items),
                           "source_changed": sum(i["source_changed"] for i in items), "unchanged_noise_controls": sum(not i["source_changed"] for i in items),
                           "changed_by_pop_subset": {f"{p}/{s}": sum(1 for i in items if i["source_changed"] and i["pop"] == p and i["subset"] == s) for p, s in sorted({(i["pop"], i["subset"]) for i in items if i["source_changed"]})},
                           "removed_cases": 0}, "population_digest": mv._digest(items),
            "input_sha256": {n: _sha(work / n) for n in ("cases.jsonl", "population.json", "labels.json", "baseline_cached_v18.json")},
            "fresh_judgments": "all 257 repaired cases judged fresh in an isolated result store; cached verdicts are only the comparison baseline",
            "bootstrap": {"seed": BOOT_SEED, "draws": BOOT_DRAWS, "unit": "document, paired, within dev"}, "cap_usd": CAP_USD, "price_usd_per_1m": mv.PRICES["o4-mini"],
            "estimate": {"calls": len(items), "expected_usd": expected, "basis": "prompt tokens ~ source chars/4 + 900 rubric/framing, completion 1,263 tokens (mean of cached v18 calls); an estimate, not a bound",
                         "hard_bound": "fsynced spend journal, one HTTP request per reservation, cap enforced"},
            "code": mv.code_hashes() | {"srcint_eval.py": _sha(Path(__file__))}, "reporting": ["dev primary", "train diagnostic", "changed vs unchanged verdict-flip rates", "false passes and false rejections", "accuracy and kappa"]}
    (work / "protocol.json").write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf8")
    return body


def verify(work: Path, expected_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, CaseRecord]]:
    work = Path(work)
    if _sha(work / "protocol.json") != expected_sha256:
        raise SrcIntError("protocol hash differs from the hash recorded at freeze time")
    frozen = json.loads((work / "protocol.json").read_text(encoding="utf8"))
    for n, d in frozen["input_sha256"].items():
        if _sha(work / n) != d:
            raise SrcIntError(f"input file {n} changed after the freeze")
    if frozen["code"] != mv.code_hashes() | {"srcint_eval.py": _sha(Path(__file__))}:
        raise SrcIntError("code, rubric or pricing changed after the freeze")
    if _sha(SRC / "cases.jsonl") != frozen["repaired_cases_sha256"] or _sha(SRC / "manifest.json") != frozen["repair_manifest_sha256"]:
        raise SrcIntError("the repaired copy changed after the freeze")
    items = json.loads((work / "population.json").read_text(encoding="utf8"))
    cases = {}
    for line in (work / "cases.jsonl").read_text(encoding="utf8").splitlines():
        c = CaseRecord.model_validate_json(line)
        cases[c.case_id] = c
    if mv._digest(items) != frozen["population_digest"] or {i["case_id"] for i in items} != set(cases):
        raise SrcIntError("population differs from the frozen population")
    cfg = mv.cfg_of(VARIANT)
    if frozen["judge"]["config_hash"] != cfg.config_hash or frozen["judge"]["prompt_template_hash"] != mv.judge.prompt_template_hash(cfg.mode):
        raise SrcIntError("judge identity changed")
    return frozen, items, cases


def run(work: Path, expected_sha256: str, request: Callable[..., Any] = mv.http_request) -> dict[str, Any]:
    frozen, items, cases = verify(work, expected_sha256)
    work = Path(work)
    results = work / "results.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if results.exists():
        for line in results.read_text(encoding="utf8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["case_id"]] = row["record"]
    journal = Journal(work / "spend_journal.jsonl", frozen["cap_usd"])
    status, stop = "COMPLETE", None
    with results.open("a", encoding="utf8") as fh:
        for it in items:
            cid = it["case_id"]
            if cid in done:  # finished (ok) or failed after the judge's own attempts: neither is repeated
                continue
            try:
                rec = mv.judge_pair(VARIANT, cases[cid], journal, request)
            except SpendError as exc:
                status, stop = "ABORTED", str(exc)
                break
            fh.write(json.dumps({"case_id": cid, "record": rec}) + "\n")
            fh.flush()
            done[cid] = rec
    failed = sorted(k for k, r in done.items() if r["status"] != "ok")
    missing = [i["case_id"] for i in items if i["case_id"] not in done]
    if status == "COMPLETE" and (failed or missing):
        status = "INCOMPLETE"
    t = {"status": status, "cases": len(items), "ok": sum(1 for r in done.values() if r["status"] == "ok"), "failed": failed, "not_attempted": missing, "stopped_on": stop,
         "spent_usd_journal": round(journal.committed(), 4), "cap_usd": frozen["cap_usd"], "protocol_sha256": expected_sha256}
    (work / "terminal.json").write_text(json.dumps(t, indent=2), encoding="utf8")
    return t


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided Fisher test for [[a, b], [c, d]] (a,b = changed flipped/not; c,d = unchanged flipped/not)."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    if n == 0:
        return 1.0
    pmf = lambda x: math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p0 = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if pmf(x) <= p0 * (1 + 1e-9)))


def compare(work: Path, expected_sha256: str) -> dict[str, Any]:
    _frozen, items, _cases = verify(work, expected_sha256)
    work = Path(work)
    results = {}
    if (work / "results.jsonl").exists():
        for line in (work / "results.jsonl").read_text(encoding="utf8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["record"]["status"] == "ok":
                    results[row["case_id"]] = row["record"]["score"]
    missing = [i["case_id"] for i in items if i["case_id"] not in results]
    if missing:  # completeness first: no label or baseline file has been parsed
        return {"status": "WITHHELD_INCOMPLETE", "valid": len(items) - len(missing), "expected": len(items), "missing_or_failed": missing, "note": "labels were not read; no metric exists"}
    target = work / "comparison.json"
    if target.exists():
        raise SrcIntError("comparison.json exists; the comparison is written once")
    labels = json.loads((work / "labels.json").read_text(encoding="utf8"))
    base = json.loads((work / "baseline_cached_v18.json").read_text(encoding="utf8"))
    rng = random.Random(BOOT_SEED)
    fresh = {i: int(results[i] >= mv.PASS_THRESHOLD) for i in results}

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
    a = sum(1 for i in changed if base[i["case_id"]] != fresh[i["case_id"]])
    c = sum(1 for i in unchanged if base[i["case_id"]] != fresh[i["case_id"]])
    result = {"status": "COMPLETE", "dataset": DATASET, "label": "DEVELOPMENT result on REPAIRED inputs with fresh judgments; not the original benchmark, not held-out, not a pass of the 0.74 target",
              "dev_primary": block(dev, True), "train_diagnostic": block(train, True), "all_257_supplementary": block(items),
              "dev_subsets": {s: block([i for i in dev if i["subset"] == s], False) for s in sorted({i["subset"] for i in dev})},
              "noise_control": {"changed_inputs": {"n": len(changed), "verdict_flips": a}, "unchanged_inputs": {"n": len(unchanged), "verdict_flips": c},
                                "flip_rate_changed": a / len(changed) if changed else None, "flip_rate_unchanged": c / len(unchanged) if unchanged else None,
                                "fisher_exact_two_sided_p": fisher_two_sided(a, len(changed) - a, c, len(unchanged) - c),
                                "reading": "unchanged inputs are byte-identical, so their flips are the judge's own run-to-run noise; a repair effect needs the changed-input flip rate to exceed it"},
              "removed_cases": 0}
    target.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["freeze", "run", "compare"])
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--protocol-sha256")
    p.add_argument("--confirm", default="")
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
                t = run(a.work, a.protocol_sha256)
                print(json.dumps(t, indent=2))
                return 0 if t["status"] == "COMPLETE" else 3
            print(json.dumps(compare(a.work, a.protocol_sha256), indent=2))
    except (SrcIntError, mv.MatchedError, SpendError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

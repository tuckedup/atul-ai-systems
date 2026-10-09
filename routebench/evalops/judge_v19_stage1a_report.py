"""Offline Stage 1a checkpoint report for judge-v19 Phase 1 (a separate helper: the frozen runner and analysis modules are not touched, no provider call, no GPU, no model fitting).

Reads the COMPLETE, fully re-validated Stage 1a results (via the frozen runner's own `stage_scores`), the cached replicate 1, and - only for this report - the original RouteBench labels.
Nothing it computes is written back into any fitting input: `fit` never opens the RouteBench labels, and this helper writes only `stage_1a_report.json` (once).
Per population (dev and TRAIN reported separately, pooled supplementary): kappa, accuracy, false passes and false rejections for each of the five single v18 runs (run1 = the cached frozen
judgment) and for the majority vote A1 (>= 3 of 5 original pass/fail verdicts); errors CORRECTED and INTRODUCED relative to A0 (= run1); coverage; committed spend; document-grouped paired
bootstrap intervals (reported, never gating); run-to-run stability. A1 is the only vote definition; a mean-score threshold is not computed anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import judge_v19_analysis as an
from . import judge_v19_phase1 as jv
from . import judge_v19_recovery as rec
from .audit2_v2 import kappa

REPORT = "stage_1a_report.json"


def _metrics(y: Sequence[int], p: Sequence[int]) -> dict[str, Any]:
    n = len(y)
    return {"kappa": kappa(list(y), list(p)) if len(set(y)) > 1 else None, "accuracy": sum(a == b for a, b in zip(y, p, strict=True)) / n,
            "false_passes": sum(1 for a, b in zip(y, p, strict=True) if a == 0 and b == 1), "false_rejections": sum(1 for a, b in zip(y, p, strict=True) if a == 1 and b == 0), "n": n}


def _vs_a0(y: Sequence[int], a0: Sequence[int], other: Sequence[int]) -> dict[str, int]:
    return {"errors_corrected": sum(1 for a, u, v in zip(y, a0, other, strict=True) if u != a and v == a),
            "errors_introduced": sum(1 for a, u, v in zip(y, a0, other, strict=True) if u == a and v != a),
            "verdicts_changed": sum(1 for u, v in zip(a0, other, strict=True) if u != v)}


def block(ids: Sequence[str], groups: dict[str, str], labels: dict[str, int], five: dict[str, list[float]]) -> dict[str, Any]:
    """Pure function (tested on synthetic data): metrics for the given case ids."""
    y = [labels[c] for c in ids]
    gs = [groups[c] for c in ids]
    runs = {f"run{r + 1}": [an.verdict(five[c][r]) for c in ids] for r in range(an.K)}
    a0, a1 = runs["run1"], [an.a1_majority(five[c]) for c in ids]
    out: dict[str, Any] = {"cases": len(ids), "documents": len(set(gs)), "A0_cached_run1": _metrics(y, a0), "A1_majority_vote": _metrics(y, a1) | _vs_a0(y, a0, a1)
                           | {"kappa_minus_A0_95_document_bootstrap": an.paired_bootstrap(gs, y, a0, a1)}, "single_runs": {}}
    for name, p in runs.items():
        out["single_runs"][name] = _metrics(y, p) | _vs_a0(y, a0, p) | ({"kappa_minus_A0_95_document_bootstrap": an.paired_bootstrap(gs, y, a0, p)} if name != "run1" else {})
    ks = [m["kappa"] for m in out["single_runs"].values() if m["kappa"] is not None]
    unanimous = sum(1 for i in range(len(ids)) if len({runs[r][i] for r in runs}) == 1)
    out["run_to_run"] = {"single_run_kappa_min": min(ks), "single_run_kappa_max": max(ks), "single_run_kappa_mean": sum(ks) / len(ks), "cases_with_all_five_verdicts_equal": unanimous,
                         "cases_with_any_disagreement": len(ids) - unanimous}
    return out


def report(work: Path, expected_sha256: str) -> dict[str, Any]:
    work = Path(work)
    target = work / REPORT
    if target.exists():
        raise jv.Phase1Error(f"{REPORT} exists; the checkpoint is written once")
    frozen, population, jobs, _jc = jv.verify(work, expected_sha256)
    if (work / rec.NOTE).exists():  # one job was recovered through the audited recovery path; the original failed row is checked to be byte-identical
        scores, recovery = rec.assemble_scores(work, expected_sha256)
    else:
        scores, recovery = jv.stage_scores(work, "1a", expected_sha256), None  # COMPLETE terminal record + every row re-validated against its stored raw response and the settled journal
    term = json.loads((work / "stage_1a" / "terminal.json").read_text(encoding="utf8"))
    state = jv.sv3.journal_state(work / "stage_1a" / "spend_journal.jsonl", frozen["stages"]["1a"]["cap_usd"])
    cached = json.loads((work / "cached_v18.json").read_text(encoding="utf8"))
    rb = population["routebench"]
    ids = [i["case_id"] for i in rb]
    five = {c: [float(cached[c]["score"]), *scores[c]] for c in ids}
    if any(len(v) != an.K for v in five.values()):
        raise jv.Phase1Error("not every case has five v18 judgments")
    labels = json.loads((work / "labels_routebench.json").read_text(encoding="utf8"))  # opened only now, after validation
    groups = {i["case_id"]: i["group"] for i in rb}
    by_pop = {p: [i["case_id"] for i in rb if i["pop"] == p] for p in ("dev", "train")}
    rep = {"status": "COMPLETE", "label": "STAGE 1A CHECKPOINT on already-inspected, original-input, evaluation-only data: a description of v18 run-to-run behaviour and of voting; not a validation, not a pass of the "
                                       "original benchmark target; no fitting used these labels",
           "protocol_sha256": expected_sha256,
           "coverage": {"stage_1a_jobs_expected": len(jobs["1a"]), "stage_1a_jobs_valid": len(jobs["1a"]), "stage_1a_terminal_status": term["status"], "judgments_per_case": an.K, "cases": len(ids),
                        "cached_replicate_1_present": sum(1 for c in ids if c in cached), "removed_cases": 0},
           "recovery": recovery, "spend": {"committed_usd": round(state["committed"], 4), "terminal_committed_usd": term["committed_usd"], "cap_usd": frozen["stages"]["1a"]["cap_usd"], "unsettled": state["unsettled"],
                     "estimate_usd": frozen["stages"]["1a"]["estimated_usd"]},
           "dev": block(by_pop["dev"], groups, labels, five), "train": block(by_pop["train"], groups, labels, five), "pooled_supplementary": block(ids, groups, labels, five),
           "definitions": {"A0": "cached frozen v18 verdict (run1)", "A1": ">= 3 of 5 original all-criteria-pass verdicts at 1.0 (not a mean-score threshold)",
                           "intervals": "95% document-grouped paired bootstrap of kappa(other) - kappa(A0), 2,000 draws, seed 20261105; reported, never gating"}}
    target.write_text(json.dumps(rep, indent=2), encoding="utf8")
    return rep


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=jv.WORK)
    p.add_argument("--protocol-sha256", required=True)
    a = p.parse_args(argv)
    try:
        print(json.dumps(report(a.work, a.protocol_sha256), indent=2))
    except (jv.Phase1Error, jv.sv3.SrcIntV3Error, an.AnalysisError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

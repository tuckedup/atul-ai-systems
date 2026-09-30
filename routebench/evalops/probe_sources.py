"""Diagnostic probe: run ONE fixed judge across additional independent label sources.

The question this answers is narrow and important: when the groundedness judge scores kappa 0.65
on TofuEval + AggreFact-CNN, is that the judge's limit or that annotation source's limit?

Method: hold the judge completely fixed (v16-holistic-4.1, same rubric, same threshold grid) and
vary only the label source. If kappa moves a lot across sources, the binding constraint is the
data; if it sits at 0.65 everywhere, it is the judge.

Isolation matters here. This writes to `data/probe/` and its own judgment cache, and never touches
`cases.jsonl`, `annotations.jsonl`, `splits.json` or the main judgment cache. Appending probe cases
to the real corpus would change `dataset_hash` and invalidate every binding and the frozen-split
guarantee, which is a high price for a diagnostic.

These probe subsets are attribution / entailment tasks rather than document summarisation, which is
exactly why the main corpus excludes them. They are usable as a diagnostic on the judge's
behaviour, and are NOT admissible as the project's groundedness claim.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

from evalops.dataset import (
    Annotation,
    CaseRecord,
    LabelProvenance,
    content_hash,
    write_jsonl,
)
from evalops.experiments import JudgmentCache, SpendMeter, run_variant
from evalops.metrics import agreement, binarize
from evalops.run_calibration import GRID

RAW = pathlib.Path(__file__).parent / "data" / "raw" / "aggrefact.jsonl"
PROBE = pathlib.Path(__file__).parent / "data" / "probe"
TRUNCATE = 6000
GRID_T = [round(0.05 * n, 2) for n in range(1, 21)]


def build(subset: str, per_class: int, seed: int) -> tuple[list[CaseRecord], list[Annotation]]:
    rows = [
        json.loads(line)
        for line in RAW.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pool = [r for r in rows if r["subset"] == subset and r["claim"].strip() and r["doc"].strip()]
    rng = random.Random(f"{seed}:{subset}")
    cases: list[CaseRecord] = []
    anns: list[Annotation] = []
    seen: set[str] = set()
    for label in (1, 0):
        bucket = [r for r in pool if int(r["label"]) == label]
        rng.shuffle(bucket)
        taken = 0
        for r in bucket:
            if taken >= per_class:
                break
            doc = r["doc"].strip()[:TRUNCATE]
            claim = r["claim"].strip()
            cid = f"probe-{subset.lower()}-{content_hash(doc, claim)}"
            if cid in seen:
                continue
            seen.add(cid)
            cases.append(CaseRecord(
                case_id=cid,
                group_id=f"probedoc-{content_hash(doc)}",
                task_class="summarize",
                task_input=(
                    "Summarise the source document faithfully. Every statement in your summary "
                    "must be supported by the document."
                ),
                context=doc,
                candidate_output=claim,
                reference="",
                reference_provenance="none; the source document is the only admissible support",
                generator_id=f"published:{subset}",
                source=f"LLM-AggreFact/{subset}",
                source_license="cc-by-nd-4.0 (original dataset card)",
                origin="real",
                meta={"subset": subset, "probe": True},
            ))
            anns.append(Annotation(
                case_id=cid, annotator_id=f"published-expert:{subset}", label=int(r["label"]),
                provenance=LabelProvenance.HUMAN_EXPERT,
                rubric_version="aggrefact-annotation-guidelines",
                reason="published expert annotation: 1 = claim supported by document",
            ))
            taken += 1
    return cases, anns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsets", default="Wice,Reveal")
    ap.add_argument("--per-class", type=int, default=64)
    ap.add_argument("--variant", default="v16-holistic-4.1")
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args(argv)

    config = next((c for c in GRID if c.variant_id == args.variant), None)
    if config is None:
        print(f"unknown variant {args.variant}", file=sys.stderr)
        return 1

    PROBE.mkdir(parents=True, exist_ok=True)
    cache = JudgmentCache(PROBE / "probe_judgment_cache.jsonl")
    meter = SpendMeter(cap_usd=args.budget)
    results: dict[str, dict] = {}

    for subset in [s.strip() for s in args.subsets.split(",") if s.strip()]:
        cases, anns = build(subset, args.per_class, args.seed)
        write_jsonl(PROBE / f"cases_{subset.lower()}.jsonl", cases)
        write_jsonl(PROBE / f"annotations_{subset.lower()}.jsonl", anns)
        pos = sum(1 for a in anns if a.label == 1)
        print(f"{subset}: {len(cases)} cases ({pos} pass / {len(anns) - pos} fail)")
        if args.build_only:
            continue

        judgments, summary = run_variant(
            cases, config, cache=cache, meter=meter, concurrency=config.concurrency or 3
        )
        by_case = {j.case_id: j for j in judgments if j.usable}
        label_of = {a.case_id: a.label for a in anns}
        human, scores, groups = [], [], []
        for c in cases:
            j = by_case.get(c.case_id)
            if j is None or j.raw_score is None:
                continue
            human.append(label_of[c.case_id])
            scores.append(float(j.raw_score))
            groups.append(c.group_id)
        completion = len(human) / len(cases) if cases else 0.0
        best = None
        for t in GRID_T:
            a = agreement(human, binarize(scores, t), groups=groups, bootstrap=0)
            if a.kappa is not None and (best is None or a.kappa > best[1]):
                best = (t, a.kappa, a)
        row = {
            "subset": subset, "n": len(human), "completion": completion,
            "best_threshold": best[0] if best else None,
            "best_kappa": best[1] if best else None,
            "confusion": best[2].confusion.as_dict() if best else None,
            "agreement": best[2].observed_agreement if best else None,
            "human_prevalence": best[2].human_prevalence if best else None,
            "errors": summary.errors, "spend_after_usd": round(meter.spent_usd, 4),
        }
        results[subset] = row
        k = row["best_kappa"]
        print(f"  -> n={row['n']} completion={completion:.1%} "
              f"kappa={'n/a' if k is None else round(k, 4)} @t={row['best_threshold']} "
              f"confusion={row['confusion']} spend=${meter.spent_usd:.4f}")

    if results:
        out = PROBE / "probe_results.json"
        out.write_text(json.dumps(
            {"variant": args.variant, "config_hash": config.config_hash,
             "note": "Diagnostic only: these subsets are attribution/entailment tasks, not "
                     "document summarisation, and are not admissible as the project's "
                     "groundedness claim.",
             "results": results}, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {out}  total spend ${meter.spent_usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

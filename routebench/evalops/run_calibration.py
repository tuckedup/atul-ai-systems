"""End-to-end calibration driver: dev grid -> freeze -> single held-out measurement.

Run order is enforced by the CLI, not by documentation:

    python -m evalops.run_calibration dev     # judge the dev split across the variant grid
    python -m evalops.run_calibration freeze  # pick the winner + threshold, write the bundle
    python -m evalops.run_calibration test    # judge the test split with the frozen bundle only

`test` refuses to run without a bundle on disk, and refuses to re-select anything. That
ordering is the entire experimental protocol: every choice is made on dev, and the test split
is scored exactly once with a configuration that can no longer change.

The variant grid is declared in `GRID` below, before any of it was run, so the experiment
record shows what was tried rather than only what won.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .calibrate import (
    ROUTEBENCH_MIN_KAPPA,
    CalibrationBundle,
    CalibrationError,
    Paired,
    evaluate,
    freeze,
    pair,
    select_threshold,
)
from .dataset import HEADLINE_PROVENANCE, Corpus, LabelProvenance, content_hash
from .experiments import JudgmentCache, RunSummary, SpendMeter, run_variant
from .judge import JudgeConfig
from .metrics import agreement, binarize
from .splits import SplitPlan, select, verify
from .taxonomy import canonical

DATA = Path(__file__).parent / "data"

#: Pre-declared candidate grid. gpt-5-mini was in the original plan but this key returns HTTP
#: 404 for the gpt-5 family on /v1/chat/completions, so it is recorded as unavailable rather
#: than quietly dropped.
GRID: tuple[JudgeConfig, ...] = (
    JudgeConfig(
        variant_id="v0-generic-4o-mini", model="gpt-4o-mini", mode="generic",
        include_reference=True, include_boundary_examples=False,
        notes="baseline: reproduces the pre-existing single-rubric judge, for comparison only",
    ),
    JudgeConfig(
        variant_id="v1-rubric-4o-mini", model="gpt-4o-mini", mode="rubric",
        notes="task rubric + reference + boundary examples",
    ),
    JudgeConfig(
        variant_id="v2-rubric-noref-4o-mini", model="gpt-4o-mini", mode="rubric",
        include_reference=False,
        notes="ablation: isolates the contribution of the verified reference answer",
    ),
    JudgeConfig(
        variant_id="v3-rubric-noboundary-4o-mini", model="gpt-4o-mini", mode="rubric",
        include_boundary_examples=False,
        notes="ablation: isolates the contribution of the rubric's boundary cases",
    ),
    JudgeConfig(
        variant_id="v4-rubric-4.1-mini", model="gpt-4.1-mini", mode="rubric",
        notes="stronger judge, same rubric",
    ),
    JudgeConfig(
        variant_id="v5-rubric-4.1", model="gpt-4.1", mode="rubric", concurrency=3,
        notes="strongest available judge, same rubric; low concurrency to stay inside TPM limits",
    ),
    JudgeConfig(
        variant_id="v6-rubric-fewshot-4.1-mini", model="gpt-4.1-mini", mode="rubric",
        exemplars_path=str(DATA / "exemplars.json"),
        notes="rubric + 2 pass/2 fail TRAIN-split exemplars for the case's own task class",
    ),
    JudgeConfig(
        variant_id="v7-rubric-fewshot-4.1", model="gpt-4.1", mode="rubric", concurrency=3,
        exemplars_path=str(DATA / "exemplars.json"),
        notes="strongest judge plus train-split exemplars",
    ),
    # --- groundedness specialists -------------------------------------------------------------
    # The first grid showed the rubric judge plateauing on groundedness around 0.62-0.70 while
    # few-shot exemplars made it WORSE twice (v6 vs v4, v7 vs v5). The fact-checking literature
    # points at a different mechanism rather than a better prompt: verify each atomic assertion
    # against the source separately. MiniCheck (arXiv:2404.10774) trains for exactly that and
    # reaches GPT-4 accuracy at 400x less cost; DEEP (arXiv:2406.13009) ensembles prompts for
    # factual-error detection. Both are mechanisms this judge did not have.
    #
    # These variants are restricted to `summarize` because that is what they are designed for --
    # per-fact decomposition of a SQL query is not a coherent operation -- and because the
    # headline track is groundedness.
    JudgeConfig(
        variant_id="v8-decompose-4.1", model="gpt-4.1", mode="decompose", concurrency=3,
        restrict_tasks=("summarize",), max_tokens=2000,
        notes="per-fact decomposition, strongest judge (MiniCheck-style mechanism)",
    ),
    JudgeConfig(
        variant_id="v9-decompose-4.1-mini", model="gpt-4.1-mini", mode="decompose",
        restrict_tasks=("summarize",), max_tokens=2000,
        notes="per-fact decomposition on the cheap judge: does the mechanism or the model carry it?",
    ),
    JudgeConfig(
        variant_id="v10-rubric-ensemble3-4.1-mini", model="gpt-4.1-mini", mode="rubric",
        samples=3, temperature=0.4, restrict_tasks=("summarize",),
        notes="3-sample majority vote per criterion (DEEP-style prompt ensembling)",
    ),
    JudgeConfig(
        variant_id="v11-decompose-ensemble3-4.1-mini", model="gpt-4.1-mini", mode="decompose",
        samples=3, temperature=0.4, restrict_tasks=("summarize",), max_tokens=2000,
        notes="decomposition plus 3-sample median: both mechanisms together",
    ),
)

UNAVAILABLE = {
    "gpt-5-mini": "HTTP 404 from /v1/chat/completions for the gpt-5 family on this key",
    "gpt-5": "HTTP 404 from /v1/chat/completions for the gpt-5 family on this key",
}


def _scope(cases: list[Any], config: JudgeConfig) -> list[Any]:
    """Apply a variant's `restrict_tasks`. An unrestricted variant sees everything."""
    if not config.restrict_tasks:
        return cases
    wanted = {canonical(x) for x in config.restrict_tasks}
    return [c for c in cases if c.task_class in wanted]


def _load() -> tuple[Corpus, SplitPlan, str]:
    corpus = Corpus.load(DATA)
    plan_path = DATA / "splits.json"
    if not plan_path.exists():
        raise CalibrationError(f"{plan_path} missing; run the split planner first")
    sp = SplitPlan.load(plan_path)
    verify(corpus.cases, sp)
    dataset_hash = content_hash(sorted(c.fingerprint for c in corpus.cases))
    return corpus, sp, dataset_hash


def _dev_row(p: Paired, threshold: float) -> dict[str, Any]:
    a = agreement(p.human, binarize(p.judge_scores, threshold), groups=p.groups,
                  tasks=p.tasks, bootstrap=0)
    return {"kappa": a.kappa, "defined": a.defined, "agreement": a.observed_agreement,
            "n": a.n, "confusion": a.confusion.as_dict()}


def cmd_dev(args: argparse.Namespace) -> int:
    corpus, sp, dataset_hash = _load()
    dev_cases = select(corpus.cases, sp, "dev")
    labels = corpus.resolved_labels()
    cache = JudgmentCache(DATA / "judgment_cache.jsonl")
    meter = SpendMeter(cap_usd=args.budget, reserve_usd=0.02)
    print(f"dev: {len(dev_cases)} cases, {len(GRID)} variants, cap ${args.budget:.2f}")

    rows: list[dict[str, Any]] = []
    summaries: list[RunSummary] = []
    for config in GRID:
        if config.model in UNAVAILABLE:
            rows.append({"variant_id": config.variant_id, "skipped": UNAVAILABLE[config.model]})
            continue
        scoped = _scope(dev_cases, config)
        judgments, summary = run_variant(
            scoped, config, cache=cache, meter=meter,
            concurrency=config.concurrency or args.concurrency,
        )
        summaries.append(summary)
        p = pair(scoped, labels, judgments)
        best = None
        try:
            best = select_threshold(p)
        except CalibrationError as e:
            print(f"  {config.variant_id}: threshold selection failed: {e}")
        row: dict[str, Any] = {
            "variant_id": config.variant_id, "model": config.model, "mode": config.mode,
            "notes": config.notes, "config_hash": config.config_hash,
            "n_paired": len(p), "completion": p.completion, "errors": p.errors,
            "spend_after_usd": round(meter.spent_usd, 4),
            "at_0.5": _dev_row(p, 0.5) if len(p) else None,
            "best_threshold": best.threshold if best else None,
            "best_dev_kappa": best.dev_kappa if best else None,
            "threshold_grid": best.grid if best else [],
            "models_served": summary.models_served,
        }
        rows.append(row)
        k = row["best_dev_kappa"]
        print(f"  {config.variant_id:32s} n={len(p):4d} kappa@0.5="
              f"{(row['at_0.5'] or {}).get('kappa')} best={k if k is None else round(k, 4)}"
              f" @t={row['best_threshold']} spend=${meter.spent_usd:.3f}")

    out = {
        "dataset_hash": dataset_hash, "split_seed": sp.seed, "n_dev_cases": len(dev_cases),
        "spend": meter.as_dict(), "unavailable": UNAVAILABLE, "variants": rows,
        "run_summaries": [s.as_dict() for s in summaries],
    }
    (DATA / "dev_experiments.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {DATA / 'dev_experiments.json'}  total spend ${meter.spent_usd:.4f}")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    corpus, sp, dataset_hash = _load()
    dev = select(corpus.cases, sp, "dev")
    labels = corpus.resolved_labels()
    experiments = json.loads((DATA / "dev_experiments.json").read_text(encoding="utf-8"))
    usable = [
        r for r in experiments["variants"]
        if not r.get("skipped") and r.get("best_dev_kappa") is not None
        and r.get("completion", 0) >= 0.98
    ]
    if not usable:
        print("no variant produced a usable dev result", file=sys.stderr)
        return 1
    if args.variant:
        chosen = next((r for r in usable if r["variant_id"] == args.variant), None)
        if chosen is None:
            print(f"variant {args.variant} has no usable dev result", file=sys.stderr)
            return 1
    else:
        chosen = max(usable, key=lambda r: r["best_dev_kappa"])

    config = next(c for c in GRID if c.variant_id == chosen["variant_id"])
    # Which dev labels the threshold is optimised against. The judge serves every task class, so
    # `pooled` (one global threshold over all admissible dev labels) is the operational default.
    # `headline` optimises against human judgments of the response only, which is the track the
    # primary claim is made on; the pooled set is numerically dominated by gold-oracle labels
    # whose class prevalence differs, so the two can disagree. Whichever is used is recorded in
    # the bundle, and both are computed on DEV -- the test split plays no part either way.
    track = {
        "pooled": None,
        "headline": HEADLINE_PROVENANCE,
        "oracle": frozenset({LabelProvenance.GOLD_ORACLE}),
    }[args.select_on]
    # Recompute the threshold from the cached dev judgments, so the bundle records the same
    # numbers the dev table reported. No new provider calls are made here by construction.
    from .experiments import _cache_key

    cache = JudgmentCache(DATA / "judgment_cache.jsonl")
    judgments = [j for j in (cache.get(_cache_key(c, config)) for c in dev) if j is not None]
    pooled = pair(dev, labels, judgments)
    selection = pooled if track is None else pooled.restrict(track)
    choice = select_threshold(selection)

    # Report what the chosen threshold does on every dev track, so a threshold that is good
    # pooled but poor on the headline track is visible at freeze time rather than after test.
    cross = {
        "pooled": _dev_row(pooled, choice.threshold),
        "headline": _dev_row(h, choice.threshold)
        if len(h := pooled.restrict(HEADLINE_PROVENANCE)) >= 20 else {"n": len(h), "skipped": True},
        "oracle": _dev_row(o, choice.threshold)
        if len(o := pooled.restrict(frozenset({LabelProvenance.GOLD_ORACLE}))) >= 20
        else {"n": len(o), "skipped": True},
    }

    bundle = freeze(
        choice, variant_id=config.variant_id, judge_config=config.as_dict(),
        dataset_hash=dataset_hash, split_seed=sp.seed, dev=selection,
        min_kappa=args.min_kappa,
        notes=(
            f"Threshold selected on the dev split only, optimising the '{args.select_on}' label "
            f"track (n={len(selection)}). Variant chosen by dev kappa among {len(usable)} "
            "variants with >=98% completion. Test split not touched at freeze time. "
            f"Dev kappa at this threshold by track: "
            + ", ".join(
                f"{k}={row.get('kappa')}" for k, row in cross.items() if not row.get("skipped")
            )
        ),
    )
    path = bundle.save(DATA / "calibration_bundle.json")
    print(f"froze {config.variant_id} at threshold {choice.threshold} "
          f"(selection track '{args.select_on}', dev kappa {choice.dev_kappa:.4f}, n={len(selection)})")
    for name, row in cross.items():
        if row.get("skipped"):
            print(f"  dev {name:9s} n={row['n']:4d}  too few to report")
        else:
            print(f"  dev {name:9s} n={row['n']:4d}  kappa={row['kappa']:.4f}  "
                  f"agreement={row['agreement']:.3f}")
    print(f"wrote {path}")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    corpus, sp, dataset_hash = _load()
    path = DATA / "calibration_bundle.json"
    if not path.exists():
        print("no calibration_bundle.json; run `freeze` first", file=sys.stderr)
        return 1
    bundle = CalibrationBundle.load(path)
    if bundle.dataset_hash != dataset_hash:
        print(f"bundle dataset_hash {bundle.dataset_hash} != corpus {dataset_hash}; the corpus "
              "changed after freezing, so this bundle cannot be measured on it", file=sys.stderr)
        return 1
    if bundle.frozen and not args.remeasure:
        existing = (bundle.test_result or {}).get("tracks", {})
        print("REFUSING: this bundle already carries a measured test_result.", file=sys.stderr)
        for name, row in sorted(existing.items()):
            if not row.get("insufficient"):
                print(f"  {name}: kappa={row.get('kappa')} n={row.get('n')}", file=sys.stderr)
        print(
            "The held-out split is measured ONCE per frozen bundle. Re-measuring and overwriting "
            "would let a disappointing number be quietly replaced by a luckier one, which is the "
            "failure mode the freeze exists to prevent.\n"
            "  --remeasure re-runs it and writes to a NEW timestamped file, preserving this one.\n"
            "  Re-selecting anything after seeing test requires a genuinely new confirmation set.",
            file=sys.stderr,
        )
        return 1

    # Reconstruct the judge the BUNDLE names, not whatever `GRID` currently defines under that
    # variant id. Reading it from `GRID` meant an edit to the grid silently changed what
    # "measuring the frozen bundle" actually executed -- the frozen configuration was frozen in
    # name only. The recomputed hash must match the stored one or this refuses to run.
    try:
        config = JudgeConfig.from_dict(bundle.judge_config)
    except (TypeError, ValueError) as e:
        print(f"cannot reconstruct the frozen judge config from the bundle: {e}", file=sys.stderr)
        return 1
    stored_hash = str(bundle.judge_config.get("config_hash", ""))
    if stored_hash and config.config_hash != stored_hash:
        print(
            "FROZEN CONFIG MISMATCH -- refusing to measure.\n"
            f"  bundle recorded config_hash {stored_hash}\n"
            f"  reconstructing it here yields {config.config_hash}\n"
            "Something the judge depends on changed after the freeze: a rubric, the prompt "
            "templates, the exemplar file contents, or a JudgeConfig field. Re-run dev and freeze "
            "again rather than measuring a judge the bundle does not describe.",
            file=sys.stderr,
        )
        return 1
    if config.variant_id != bundle.variant_id:
        print(f"bundle variant_id {bundle.variant_id!r} != reconstructed "
              f"{config.variant_id!r}", file=sys.stderr)
        return 1
    print(f"  reconstructed frozen judge {config.variant_id} "
          f"(config_hash {config.config_hash}, verified against the bundle)")

    test_cases = select(corpus.cases, sp, "test")
    labels = corpus.resolved_labels()
    cache = JudgmentCache(DATA / "judgment_cache.jsonl")
    meter = SpendMeter(cap_usd=args.budget, reserve_usd=0.02)
    print(f"test: {len(test_cases)} cases, variant {bundle.variant_id}, "
          f"threshold {bundle.threshold} (frozen), cap ${args.budget:.2f}")

    scoped = _scope(test_cases, config)
    judgments, summary = run_variant(
        scoped, config, cache=cache, meter=meter,
        concurrency=config.concurrency or args.concurrency,
    )
    p = pair(scoped, labels, judgments)
    result = evaluate(bundle, p, bootstrap=args.bootstrap)
    result["run_summary"] = summary.as_dict()
    bundle.test_result = result
    if args.remeasure:
        # Never overwrite an existing measurement; a re-measurement is a separate artifact so
        # both remain auditable and the first result cannot be quietly discarded.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = path.with_name(f"calibration_bundle.remeasured.{stamp}.json")
        bundle.save(out)
        print(f"re-measurement written to {out} (original {path} left untouched)")
        path = out
    else:
        bundle.save(path)
    (DATA / "calibration_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    print()
    for name, row in sorted(result["tracks"].items()):
        if row.get("insufficient"):
            print(f"  {name:32s} n={row['n']:4d}  {row['note']}")
            continue
        ci = row["interval"]
        bounds = (f"[{ci['low']:.3f}, {ci['high']:.3f}]" if ci.get("usable") else "unavailable")
        print(f"  {name:32s} n={row['n']:4d}  kappa={row['kappa']:.4f}  95% CI {bounds}  "
              f"agreement={row['observed_agreement']:.3f}  pass={row['passes_min_kappa']}")
    print(f"\ncompletion {result['completion']['completion_rate']:.1%}  "
          f"spend ${meter.spent_usd:.4f}")
    print(f"wrote {DATA / 'calibration_report.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("dev", cmd_dev), ("freeze", cmd_freeze), ("test", cmd_test)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--budget", type=float, default=25.0, help="hard spend cap in USD")
        p.add_argument("--concurrency", type=int, default=8)
        p.add_argument("--min-kappa", type=float, default=ROUTEBENCH_MIN_KAPPA)
        if name == "freeze":
            p.add_argument("--variant", default=None, help="override the dev-kappa winner")
            p.add_argument("--select-on", choices=["pooled", "headline", "oracle"],
                           default="pooled",
                           help="which dev label track the threshold is optimised against")
        if name == "test":
            p.add_argument("--bootstrap", type=int, default=2000)
            p.add_argument("--remeasure", action="store_true")
    args = ap.parse_args(argv)
    try:
        return int(args.fn(args))
    except CalibrationError as e:
        print(f"calibration error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

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
    expected_bundle_id,
    freeze,
    pair,
    select_threshold,
)
from .dataset import HEADLINE_PROVENANCE, Corpus, LabelProvenance, content_hash
from .experiments import (
    JudgmentCache,
    RunSummary,
    SpendMeter,
    _cache_key,
    run_variant,
    worst_case_call_usd,
)
from .judge import JudgeConfig, prompt_template_hash
from .metrics import agreement, binarize
from .rubrics import bundle_hash
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
    # --- second round: matched comparison, addressed evidence, and a diverse ensemble ----------
    # Three findings shaped this round, all of them from `data/COMBINER_PROBE.md`:
    #
    # 1. Combining v0-v7 beat the strongest single judge on NO track that backs the headline claim
    #    (margin +0.0000 pooled and on groundedness). Eight variants sharing one rubric mechanism
    #    have correlated errors, so there is nothing for a combiner to average out. An ensemble
    #    needs components that fail DIFFERENTLY -- hence `contradict`, which asks the complement
    #    question, rather than a ninth rubric variant.
    # 2. A controlled study (arXiv:2603.28005) found detailed holistic judging matching or beating
    #    single-prompt atomic decomposition on completeness-sensitive QA. So decomposition is a
    #    hypothesis to test against a MATCHED holistic baseline, not an upgrade to assume. v16
    #    exists only to be that baseline: identical model, scope, token budget and concurrency to
    #    v8, differing in mechanism alone.
    # 3. `decompose` can only check whether a quote appears SOMEWHERE in the document, which a
    #    topically-similar sentence satisfies while supporting nothing. `decompose_addressed`
    #    makes the judge name the sentences, so the citation itself is checkable.
    JudgeConfig(
        variant_id="v12-addressed-4.1", model="gpt-4.1", mode="decompose_addressed",
        concurrency=3, restrict_tasks=("summarize",), max_tokens=2000,
        notes="source-addressed claim verification: per-fact, with [S<n>] sentence citations",
    ),
    JudgeConfig(
        variant_id="v13-addressed-4.1-mini", model="gpt-4.1-mini", mode="decompose_addressed",
        restrict_tasks=("summarize",), max_tokens=2000,
        notes="addressed verification on the cheap judge: mechanism or model?",
    ),
    JudgeConfig(
        variant_id="v14-contradict-4.1", model="gpt-4.1", mode="contradict", concurrency=3,
        restrict_tasks=("summarize",), max_tokens=1200,
        notes="contradiction-focused judge; an ensemble member with a different failure mode",
    ),
    JudgeConfig(
        variant_id="v15-contradict-4.1-mini", model="gpt-4.1-mini", mode="contradict",
        restrict_tasks=("summarize",), max_tokens=1200,
        notes="contradiction detection on the cheap judge",
    ),
    JudgeConfig(
        variant_id="v16-holistic-4.1", model="gpt-4.1", mode="rubric", concurrency=3,
        restrict_tasks=("summarize",), max_tokens=2000,
        notes=("matched holistic baseline for v8/v12: same model, scope, token budget and "
               "concurrency, whole-response rubric instead of per-fact decomposition"),
    ),
)

#: Comparisons declared BEFORE the run, so "decomposition helped" is a prediction being tested
#: rather than a pattern found afterwards. Each entry is (label, treatment, matched control) and
#: names the single thing that differs between them.
MATCHED_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("decomposition vs holistic", "v8-decompose-4.1", "v16-holistic-4.1",
     "mechanism: per-fact verification vs one whole-response rubric judgement"),
    ("addressed vs unaddressed decomposition", "v12-addressed-4.1", "v8-decompose-4.1",
     "whether the judge must cite the source sentences it relies on"),
    ("addressed: strong vs cheap model", "v12-addressed-4.1", "v13-addressed-4.1-mini",
     "model only; the mechanism is identical"),
)

#: Ensemble candidates for the combiner, declared before fitting. Diversity is the design goal:
#: each candidate mixes mechanisms rather than models, because the probe showed that mixing models
#: within one mechanism buys nothing.
ENSEMBLE_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("holistic+decomposed", ("v16-holistic-4.1", "v8-decompose-4.1")),
    ("holistic+addressed", ("v16-holistic-4.1", "v12-addressed-4.1")),
    ("holistic+addressed+contradiction",
     ("v16-holistic-4.1", "v12-addressed-4.1", "v14-contradict-4.1")),
    # The negative control: three variants of one mechanism. The probe predicts this gains
    # nothing, and an ensemble study that omits its own null case is not evidence.
    ("three-rubric-control",
     ("v4-rubric-4.1-mini", "v5-rubric-4.1", "v6-rubric-fewshot-4.1-mini")),
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


def _track(p: Paired, name: str) -> Paired:
    if name == "pooled":
        return p
    return p.restrict(HEADLINE_PROVENANCE if name == "headline" else
                      frozenset({LabelProvenance.GOLD_ORACLE}))


def _requested_grid(args: argparse.Namespace) -> list[JudgeConfig]:
    names = getattr(args, "variants", None)
    if not names:
        return list(GRID)
    by_id = {c.variant_id: c for c in GRID}
    unknown = sorted(set(names) - by_id.keys())
    if unknown:
        raise CalibrationError(f"unknown variants: {unknown}")
    return [by_id[n] for n in dict.fromkeys(names)]


def _dev_cases(corpus: Corpus, sp: SplitPlan, name: str) -> list[Any]:
    cases = select(corpus.cases, sp, "dev")
    if name == "pooled":
        return cases
    allowed = HEADLINE_PROVENANCE if name == "headline" else {LabelProvenance.GOLD_ORACLE}
    labels = corpus.resolved_labels()
    return [c for c in cases if c.case_id in labels and labels[c.case_id].provenance in allowed]


def cmd_plan(args: argparse.Namespace) -> int:
    """No provider calls, model downloads, writes or held-out judgments."""
    corpus, sp, _ = _load()
    cases = _dev_cases(corpus, sp, args.select_on)
    cache = JudgmentCache(DATA / "judgment_cache.jsonl")
    rows = []
    for config in _requested_grid(args):
        scoped = _scope(cases, config)
        missing = sum(1 for c in scoped
                      if (j := cache.get(_cache_key(c, config))) is None or not j.usable)
        bound = worst_case_call_usd(config.model, config) * config.max_attempts * config.samples
        rows.append({"variant_id": config.variant_id, "cases": len(scoped),
                     "missing_current_judgments": missing,
                     "reservation_per_case_usd": round(bound, 6),
                     "reservation_envelope_usd": round(bound * missing, 6)})
    print(json.dumps({"split": "dev", "selection_track": args.select_on,
                      "variants": rows,
                      "note": "Reservation envelope, not a measured price quote; no calls made."},
                     indent=2))
    return 0


def cmd_dev(args: argparse.Namespace) -> int:
    corpus, sp, dataset_hash = _load()
    selection_track = getattr(args, "select_on", "pooled")
    dev_cases = _dev_cases(corpus, sp, selection_track)
    grid = _requested_grid(args)
    labels = corpus.resolved_labels()
    cache = JudgmentCache(DATA / "judgment_cache.jsonl")
    meter = SpendMeter(cap_usd=args.budget, reserve_usd=0.02)
    print(f"dev: {len(dev_cases)} cases, {len(grid)} variants, cap ${args.budget:.2f}")

    rows: list[dict[str, Any]] = []
    summaries: list[RunSummary] = []
    blocked = None
    for config in grid:
        if config.model in UNAVAILABLE:
            rows.append({"variant_id": config.variant_id, "skipped": UNAVAILABLE[config.model]})
            continue
        scoped = _scope(dev_cases, config)
        judgments, summary = run_variant(
            scoped, config, cache=cache, meter=meter,
            concurrency=config.concurrency or args.concurrency,
        )
        summaries.append(summary)
        p = pair(scoped, labels, judgments, split="dev")
        best = None
        try:
            best = select_threshold(p)
        except CalibrationError as e:
            print(f"  {config.variant_id}: threshold selection failed: {e}")
        row: dict[str, Any] = {
            "variant_id": config.variant_id, "model": config.model, "mode": config.mode,
            "notes": config.notes, "config_hash": config.config_hash,
            # The FULL config, so the experiment record describes the judge that produced these
            # numbers without depending on `GRID` still saying the same thing. `cmd_freeze` used
            # to look the variant up in `GRID`, which made the record unusable the moment any
            # `config_hash` input changed: the cache lookup silently found nothing and freeze
            # failed with "0 paired items" rather than naming the drift.
            "judge_config": config.as_dict(),
            # Recorded separately because it is one of only two `config_hash` inputs that the
            # config dict does NOT carry (the other is `exemplars_hash`). Storing it is what makes
            # a later hash mismatch attributable: "the prompts changed" instead of "something did".
            "prompt_template_hash": prompt_template_hash(config.mode),
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
        if summary.errors.get("quota_exhausted"):
            blocked = "quota_exhausted"
            print("Provider quota exhausted; stopping remaining variants. "
                  "Resolve credits/limits before resuming.", file=sys.stderr)
            break

    out = {
        "dataset_hash": dataset_hash, "split_seed": sp.seed, "n_dev_cases": len(dev_cases),
        "annotations_hash": corpus.annotations_hash,
        "split_membership_hash": sp.membership_hash,
        "selection_track": selection_track,
        "blocked": blocked,
        "spend": meter.as_dict(), "unavailable": UNAVAILABLE, "variants": rows,
        "run_summaries": [s.as_dict() for s in summaries],
    }
    record_path = DATA / "dev_experiments.json"
    if record_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record_path.with_name(f"dev_experiments.previous.{stamp}.json").write_text(
            record_path.read_text(encoding="utf-8"), encoding="utf-8")
    record_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {DATA / 'dev_experiments.json'}  total spend ${meter.spent_usd:.4f}")
    return 3 if blocked else 0


def _recorded_config(row: dict[str, Any], experiments: dict[str, Any]) -> JudgeConfig:
    """Rebuild the judge that produced a dev row, from the experiment record rather than `GRID`.

    This is the same rule `cmd_test` already applies to a frozen bundle -- reconstruct the judge
    the RECORD names, not whatever the grid currently defines under that id -- applied one step
    earlier, because freeze had the identical hole and it had already bitten:

        $ python -m evalops.run_calibration freeze
        calibration error: threshold selection needs a usable dev set; got 0 paired items.

    with 2,968 usable judgments sitting in the cache. Every recorded `config_hash` in
    `dev_experiments.json` differed from what `GRID` reconstructed, so `cache.get(_cache_key(...))`
    missed on every case, `pooled` came back empty, and the error blamed the dev set. The cause is
    that `config_hash` covers every `JudgeConfig` field and the prompt scaffolding, so adding a
    field or editing a template after a run -- which is what declaring the v8-v11 variants did --
    renames every judge in the record.

    Older records predate `judge_config`; for those the full config is recovered from
    `run_summaries`, which always carried it. If neither has it, this refuses rather than falling
    back to `GRID`, because a silent fallback is how the drift went unnoticed.
    """
    variant = row["variant_id"]
    saved = row.get("judge_config")
    if not saved:
        for summary in experiments.get("run_summaries", []):
            cfg = summary.get("config") or {}
            if cfg.get("variant_id") == variant:
                saved = cfg
                break
    if not saved:
        raise CalibrationError(
            f"the dev record for {variant!r} carries no judge_config, and none of its "
            "run_summaries does either. Re-run `dev` so the record describes its own judge; "
            "reconstructing it from GRID is what let the code and the experiment diverge "
            "without anything noticing."
        )
    config = JudgeConfig.from_dict(saved)
    recorded_hash = str(saved.get("config_hash", "") or row.get("config_hash", ""))
    if recorded_hash and config.config_hash != recorded_hash:
        legacy = config.legacy_config_hash
        if recorded_hash == legacy:
            raise CalibrationError(
                f"variant {variant!r} was recorded under the PRE-2026-09-28 config_hash formula.\n"
                f"  recorded config_hash:    {recorded_hash}  (reproduced exactly by the legacy "
                "formula)\n"
                f"  current config_hash:     {config.config_hash}\n"
                "The legacy formula omitted `prompt_template` entirely, so a prompt edit was "
                "invisible to it. The difference above therefore shows only that the FORMULA "
                "changed; it is NOT evidence that the prompts changed, and an earlier revision of "
                "docs/KAPPA_DESIGN.md wrongly claimed it was.\n"
                "What IS verified: every `rubric_version` in those cached judgments matches the "
                "rubrics in this tree. What is NOT covered by anything: the `_ROLE` and "
                "output-format scaffolding. So these judgments have unknown prompt provenance — "
                "usable for a labelled development probe, never admissible behind a frozen bundle. "
                "Re-judging under pinned prompts is the only way to make them admissible; nothing "
                "here will relabel them as current."
            )
        recorded_prompt = str(row.get("prompt_template_hash", "") or "not recorded")
        raise CalibrationError(
            f"variant {variant!r} cannot be reconstructed.\n"
            f"  recorded config_hash:      {recorded_hash}\n"
            f"  rebuilt from the record:   {config.config_hash}\n"
            f"  legacy formula would give: {legacy}  (also does not match)\n"
            f"  recorded prompt_template:  {recorded_prompt}\n"
            f"  prompt_template now:       {prompt_template_hash(config.mode)}\n"
            "Every stored JudgeConfig FIELD round-trips exactly and neither the current nor the "
            "known legacy formula reproduces the recorded hash, so this record was written by a "
            "third formula or against different exemplars. Re-run `dev` rather than guessing."
        )
    live = next((c for c in GRID if c.variant_id == variant), None)
    if live is not None and live.config_hash != config.config_hash:
        print(
            f"  note: GRID's {variant} now hashes to {live.config_hash}, but the dev record was "
            f"measured at {config.config_hash}. Freezing the RECORDED judge. A later `dev` run "
            "will re-judge under the new hash rather than reuse these judgments."
        )
    return config


def _saved_config(row: dict[str, Any], experiments: dict[str, Any]) -> dict[str, Any] | None:
    """The stored config dict for a row, from the row or from `run_summaries`. Never raises."""
    saved = row.get("judge_config")
    if saved:
        return dict(saved)
    for summary in experiments.get("run_summaries", []):
        cfg = summary.get("config") or {}
        if cfg.get("variant_id") == row.get("variant_id"):
            return dict(cfg)
    return None


def verify_record(experiments: dict[str, Any]) -> list[dict[str, Any]]:
    """Check every recorded variant against the code in the tree. One row per variant.

    Separate from `freeze` because the answer to "can this experiment record still be used?" should
    not require attempting a freeze and reading a misleading error. `reproducible` is False when the
    judgments the record describes cannot be found or rebuilt from the current code.
    """
    rows: list[dict[str, Any]] = []
    for row in experiments.get("variants", []):
        variant = row.get("variant_id", "?")
        if row.get("skipped"):
            rows.append({"variant_id": variant, "skipped": row["skipped"]})
            continue
        entry: dict[str, Any] = {
            "variant_id": variant,
            "recorded_config_hash": row.get("config_hash"),
            "recorded_dev_kappa": row.get("best_dev_kappa"),
            "recorded_completion": row.get("completion"),
        }
        try:
            config = _recorded_config(row, experiments)
        except CalibrationError as e:
            entry.update({"reproducible": False, "reason": str(e).splitlines()[0],
                          "detail": str(e)})
            # Distinguish "written under the recovered legacy formula" from "written under
            # something we cannot identify". The first says the judgments are findable and their
            # prompt provenance is unknown; the second says the record is opaque. Collapsing them
            # is what produced the overstated "the prompts are lost" claim.
            try:
                live_cfg = JudgeConfig.from_dict(
                    row.get("judge_config") or _saved_config(row, experiments) or {}
                )
            except (TypeError, ValueError):
                live_cfg = None
            if live_cfg is not None and entry["recorded_config_hash"] == \
                    live_cfg.legacy_config_hash:
                entry.update({
                    "legacy_formula": True,
                    "prompt_provenance": "unverified_legacy",
                    "note": ("recorded under the pre-2026-09-28 config_hash, which omitted "
                             "prompt_template; the judgments are findable but their prompt "
                             "provenance cannot be established either way"),
                })
        else:
            live = next((c for c in GRID if c.variant_id == variant), None)
            entry.update({
                "reproducible": True,
                "rebuilt_config_hash": config.config_hash,
                "grid_config_hash": live.config_hash if live else None,
                "grid_matches_record": bool(live and live.config_hash == config.config_hash),
            })
        rows.append(entry)
    return rows


def cmd_verify(args: argparse.Namespace) -> int:
    corpus, sp, dataset_hash = _load()
    path = DATA / "dev_experiments.json"
    if not path.exists():
        print(f"{path} missing; nothing to verify", file=sys.stderr)
        return 1
    experiments = json.loads(path.read_text(encoding="utf-8"))
    rows = verify_record(experiments)

    recorded_dataset = experiments.get("dataset_hash")
    dataset_ok = recorded_dataset == dataset_hash
    print(f"corpus              {len(corpus.cases)} cases, splits verified free of group and "
          "content leakage")
    recorded_seed = experiments.get("split_seed")
    print(f"split_seed          record={recorded_seed}  plan={sp.seed}  "
          f"{'OK' if recorded_seed == sp.seed else 'MISMATCH'}")
    print(f"dataset_hash        record={recorded_dataset}  corpus={dataset_hash}  "
          f"{'OK' if dataset_ok else 'MISMATCH'}")
    print(f"rubric_bundle_hash  {bundle_hash()}")
    print()

    broken = []
    for row in rows:
        if row.get("skipped"):
            print(f"  {row['variant_id']:34s} skipped: {row['skipped']}")
            continue
        if not row.get("reproducible"):
            broken.append(row)
            label = ("LEGACY FORMULA (findable, prompt provenance unverified)"
                     if row.get("legacy_formula") else "NOT REPRODUCIBLE (formula unidentified)")
            print(f"  {row['variant_id']:34s} {label}")
            continue
        note = "" if row["grid_matches_record"] else "  (GRID has since changed)"
        print(f"  {row['variant_id']:34s} ok  {row['rebuilt_config_hash']}{note}")

    if broken:
        legacy = [r for r in broken if r.get("legacy_formula")]
        if len(legacy) == len(broken):
            print(f"\nAll {len(broken)} recorded variants were written under the recovered "
                  "pre-2026-09-28 config_hash formula. Their judgments are findable in the cache "
                  "and their prompt provenance is unverifiable — not lost, and not current. "
                  "Re-judge under pinned prompts before freezing anything.\n", file=sys.stderr)
        else:
            print(f"\n{len(broken)} of {len(rows)} recorded variants cannot be reproduced from "
                  f"this tree ({len(legacy)} under the known legacy formula, "
                  f"{len(broken) - len(legacy)} unidentified). The first one in detail:\n",
                  file=sys.stderr)
        print(broken[0]["detail"], file=sys.stderr)
        return 1
    if not dataset_ok:
        print("\nthe corpus changed since the recorded run", file=sys.stderr)
        return 1
    print("\nevery recorded variant is reproducible from this tree")
    return 0


def overwrite_refusal(bundle_path: Path, *, replace: bool) -> str | None:
    """Why freezing must not write to `bundle_path`, or None if it may.

    `cmd_test` refuses to re-measure a bundle that already carries a `test_result`, so the held-out
    split is scored once per bundle. Nothing stopped `cmd_freeze` REPLACING that file, which made
    "freeze a different variant and measure that one" the same loophole reached one step earlier:
    each bundle is measured once, and you may have as many bundles as you like against one test
    split.

    Factored out of `cmd_freeze` so the guard is testable without a corpus on disk. It also has to
    run before any corpus work, or a missing corpus becomes the reason a measured artifact survives.
    """
    if replace or not bundle_path.exists():
        return None
    try:
        existing = CalibrationBundle.load(bundle_path)
    except Exception:  # noqa: BLE001 - an unreadable artifact is not worth protecting
        return None
    if not existing.frozen:
        return None
    lines = [
        "REFUSING: a bundle carrying a measured test_result is already on disk.",
        f"  {bundle_path}",
        f"  variant {existing.variant_id} at threshold {existing.threshold}",
    ]
    for name, row in sorted((existing.test_result or {}).get("tracks", {}).items()):
        if isinstance(row, dict) and not row.get("insufficient"):
            lines.append(f"  {name}: kappa={row.get('kappa')} n={row.get('n')}")
    lines += [
        ("Overwriting it would discard a held-out measurement and let a second variant be measured "
         "on the same test split -- the loophole `cmd_test --remeasure` exists to close, reopened "
         "one step earlier."),
        "  --replace archives the existing artifact beside it and proceeds.",
        "  Re-selecting after seeing test requires a genuinely new confirmation set.",
    ]
    return "\n".join(lines)


def cmd_freeze(args: argparse.Namespace) -> int:
    bundle_path = DATA / "calibration_bundle.json"
    refusal = overwrite_refusal(bundle_path, replace=args.replace)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    corpus, sp, dataset_hash = _load()
    dev = select(corpus.cases, sp, "dev")
    labels = corpus.resolved_labels()
    experiments = json.loads((DATA / "dev_experiments.json").read_text(encoding="utf-8"))
    for key, expected in (("dataset_hash", dataset_hash), ("split_seed", sp.seed),
                          ("annotations_hash", corpus.annotations_hash),
                          ("split_membership_hash", sp.membership_hash)):
        if experiments.get(key) != expected:
            raise CalibrationError(f"dev record {key} missing or changed; rerun dev before freeze")
    cache = JudgmentCache(DATA / "judgment_cache.jsonl")
    usable = []
    for row in experiments["variants"]:
        if row.get("skipped") or (args.variant and row["variant_id"] != args.variant):
            continue
        config = _recorded_config(row, experiments)
        judgments = [j for c in _scope(dev, config)
                     if (j := cache.get(_cache_key(c, config))) is not None]
        # Pair against ALL labelled dev cases before track restriction. Scoping must not erase
        # unjudged cases from the target population and inflate completion.
        pooled = pair(dev, labels, judgments, split="dev")
        selection = _track(pooled, args.select_on)
        if selection.completion < 0.98:
            continue
        try:
            choice = select_threshold(selection)
        except CalibrationError:
            continue
        usable.append((choice.dev_kappa, config, pooled, selection, choice))
    if not usable:
        print("no variant produced a usable dev result", file=sys.stderr)
        return 1
    _, config, pooled, selection, choice = max(usable, key=lambda item: item[0])

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
        annotations_hash=corpus.annotations_hash,
        split_membership_hash=sp.membership_hash,
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
    if args.replace and bundle_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = bundle_path.with_name(f"calibration_bundle.replaced.{stamp}.json")
        archived.write_text(bundle_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"archived the previous artifact to {archived}")
    path = bundle.save(bundle_path)
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
    checks = {
        "bundle_id": (bundle.bundle_id, expected_bundle_id(bundle)),
        "annotations_hash": (bundle.annotations_hash, corpus.annotations_hash),
        "split_membership_hash": (bundle.split_membership_hash, sp.membership_hash),
        "rubric_bundle_hash": (bundle.rubric_bundle_hash, bundle_hash()),
        "split_seed": (bundle.split_seed, sp.seed),
    }
    for name, (recorded, current) in checks.items():
        if recorded != current:
            print(f"frozen {name} missing or changed; refusing before inference", file=sys.stderr)
            return 1
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
    if summary.errors.get("quota_exhausted"):
        print("Provider quota exhausted; cached outputs retained, but bundle and test report "
              "left unchanged. Resume the SAME frozen judge after quota is restored.",
              file=sys.stderr)
        return 3
    # Include the declared task scope AND the complete headline population. A specialist need
    # not grade SQL, but cannot make human-labelled cases disappear by narrowing its scope.
    scoped_ids = {c.case_id for c in scoped}
    population = [c for c in test_cases if c.case_id in scoped_ids or
                  (c.case_id in labels and labels[c.case_id].provenance in HEADLINE_PROVENANCE)]
    p = pair(population, labels, judgments, split="test")
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
    for name, fn in (("plan", cmd_plan), ("dev", cmd_dev), ("verify", cmd_verify), ("freeze", cmd_freeze),
                     ("test", cmd_test)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--budget", type=float, default=25.0, help="hard spend cap in USD")
        p.add_argument("--concurrency", type=int, default=8)
        p.add_argument("--min-kappa", type=float, default=ROUTEBENCH_MIN_KAPPA)
        if name in {"dev", "plan"}:
            p.add_argument("--variants", nargs="+", help="explicit variant IDs, in execution order")
        if name == "freeze":
            p.add_argument("--variant", default=None, help="override the dev-kappa winner")
            p.add_argument("--replace", action="store_true",
                           help="replace an existing MEASURED bundle, archiving it first")
        if name in {"plan", "dev", "freeze"}:
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

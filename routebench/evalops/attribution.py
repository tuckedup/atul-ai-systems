"""The claim-level attribution experiment: build -> dev -> freeze -> test, once.

The historical corpus was governed by `docs/ATTRIBUTION_EXPERIMENT_DECLARATION.md`. That corpus is
immutable. A future fresh build must name a successor protocol covering full-source retention and
prior-exposure exclusion; this module will not retrofit those changes onto the old declaration.

It deliberately uses its own data directory (`data/attribution/`) and its own judgment cache. The
groundedness corpus, its splits and its bindings are untouched, so the separate 0.6532 groundedness
result remains exactly as measured and the two claims cannot be confused.

Phases are separate commands on purpose: `test` refuses to run before `freeze`, and refuses to
overwrite a bundle that already carries a measurement. That ordering *is* the protocol.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from datetime import datetime, timezone
from typing import Any

from evalops.calibrate import (
    ROUTEBENCH_MIN_KAPPA,
    CalibrationBundle,
    CalibrationError,
    evaluate,
    expected_bundle_id,
    freeze,
    pair,
    select_threshold,
)
from evalops.dataset import (
    Annotation,
    CaseRecord,
    Corpus,
    LabelProvenance,
    content_hash,
    write_jsonl,
)
from evalops.experiments import JudgmentCache, SpendMeter, _cache_key, run_variant
from evalops.judge import JudgeConfig, prompt_template_hash
from evalops.rubrics import bundle_hash
from evalops.run_calibration import GRID
from evalops.splits import SplitPlan, plan, select, verify

HERE = pathlib.Path(__file__).parent
RAW = HERE / "data" / "raw" / "aggrefact.jsonl"
DATA = HERE / "data" / "attribution"
EXPOSED_CASES = (
    HERE / "data" / "probe" / "cases_reveal.jsonl",
    HERE / "data" / "probe" / "cases_wice.jsonl",
)
LEGACY_PROTOCOL = HERE.parents[1] / "docs" / "ATTRIBUTION_EXPERIMENT_DECLARATION.md"

#: Every LLM-AggreFact subset whose unit of judgment is a single claim against supplied evidence.
#: No selection inside the class -- see declaration section 2.
SUBSETS = ("Reveal", "ClaimVerify", "FactCheck-GPT", "Wice", "ExpertQA", "Lfqa")
PER_CLASS = 50
SEED = 20260928
CANDIDATES = ("v16-holistic-4.1", "v8-decompose-4.1")
THRESHOLDS = [round(0.05 * n, 2) for n in range(1, 21)]
DATA_ARTIFACTS = (
    "cases.jsonl",
    "annotations.jsonl",
    "splits.json",
    "judgment_cache.jsonl",
    "dev_experiments.json",
    "calibration_bundle.json",
    "calibration_report.json",
    "build_manifest.json",
)


# ---------------------------------------------------------------- build
def _exposed_documents() -> dict[str, tuple[str, ...]]:
    """Probe documents by subset, matched on content rather than prefixed case/group IDs.

    The probe mirror may itself hold a truncated prefix of the raw source. A raw document is
    therefore exposed when it starts with a non-empty probe context from the same subset. This is
    stricter than matching `(context, claim)`: once any claim from a document was examined, every
    other claim from that document is excluded from a future confirmation corpus.
    """
    by_subset: dict[str, set[str]] = {}
    for exposed_path in EXPOSED_CASES:
        if not exposed_path.exists():
            raise CalibrationError(
                f"cannot audit prior exposure because {exposed_path} is missing; refusing to build"
            )
        for line_no, line in enumerate(
            exposed_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                context = str(row.get("context", "")).strip()
                meta = row.get("meta") or {}
                subset = str(meta.get("subset", "")).strip()
            except (json.JSONDecodeError, AttributeError) as exc:
                raise CalibrationError(
                    f"cannot audit prior exposure: invalid {exposed_path}:{line_no}: {exc}"
                ) from exc
            if context and subset:
                by_subset.setdefault(subset, set()).add(context)
    if not by_subset:
        raise CalibrationError("prior probe files contain no subset-bound document contexts")
    return {subset: tuple(sorted(contexts)) for subset, contexts in by_subset.items()}


def build() -> tuple[list[CaseRecord], list[Annotation], dict[str, Any]]:
    rows = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    present = {r["subset"] for r in rows}
    missing = [s for s in SUBSETS if s not in present]
    if missing:
        print(f"WARNING: declared subsets absent from the raw mirror: {missing}", file=sys.stderr)

    cases: list[CaseRecord] = []
    anns: list[Annotation] = []
    seen: set[str] = set()
    exposed = _exposed_documents()
    exposed_hashes = {
        content_hash(context) for contexts in exposed.values() for context in contexts
    }
    audit: dict[str, Any] = {
        "prior_exposure_sources": [str(path) for path in EXPOSED_CASES],
        "prior_exposure_document_hashes": len(exposed_hashes),
        "excluded_prior_exposure_rows": 0,
        "excluded_prior_exposure_documents": 0,
        "retained_over_6000_chars": 0,
        "by_subset_label": {},
    }
    excluded_docs: set[str] = set()
    for subset in SUBSETS:
        pool = [
            r for r in rows
            if r["subset"] == subset and r["claim"].strip() and r["doc"].strip()
        ]
        rng = random.Random(f"{SEED}:{subset}")
        for label in (1, 0):
            labelled = [r for r in pool if int(r["label"]) == label]
            bucket = []
            bucket_seen: set[str] = set()
            excluded_rows = 0
            for row in labelled:
                doc = row["doc"].strip()
                doc_hash = content_hash(doc)
                if any(doc.startswith(prefix) for prefix in exposed.get(subset, ())):
                    excluded_rows += 1
                    excluded_docs.add(doc_hash)
                    continue
                cid = f"attr-{subset.lower()}-{content_hash(doc, row['claim'].strip())}"
                if cid in bucket_seen or cid in seen:
                    continue
                bucket_seen.add(cid)
                bucket.append(row)
            key = f"{subset}:{label}"
            audit["by_subset_label"][key] = {
                "raw_rows": len(labelled),
                "excluded_prior_exposure_rows": excluded_rows,
                "eligible_unique_rows": len(bucket),
                "selected": min(PER_CLASS, len(bucket)),
            }
            audit["excluded_prior_exposure_rows"] += excluded_rows
            if len(bucket) < PER_CLASS:
                raise CalibrationError(
                    f"after content-based prior-exposure exclusion, {subset} label={label} has "
                    f"only {len(bucket)} eligible unique rows; protocol requires {PER_CLASS}. "
                    "Refusing an under-quota or silently substituted corpus."
                )
            rng.shuffle(bucket)
            for r in bucket[:PER_CLASS]:
                # The source is evidence. Truncating it can remove the support (or contradiction)
                # that defines the published label, so fresh protocols retain it in full.
                doc = r["doc"].strip()
                claim = r["claim"].strip()
                cid = f"attr-{subset.lower()}-{content_hash(doc, claim)}"
                seen.add(cid)
                if len(doc) > 6000:
                    audit["retained_over_6000_chars"] += 1
                cases.append(CaseRecord(
                    case_id=cid,
                    group_id=f"attrdoc-{content_hash(doc)}",
                    task_class="summarize",   # reuses the groundedness rubric
                    task_input=(
                        "Verify the claim below against the supplied source material. Every "
                        "assertion in the claim must be supported by that material."
                    ),
                    context=doc,
                    candidate_output=claim,
                    reference="",
                    reference_provenance="none; the supplied source material is the only admissible support",
                    generator_id=f"published:{subset}",
                    source=f"LLM-AggreFact/{subset}",
                    source_license="cc-by-nd-4.0 (original dataset card)",
                    origin="real",
                    meta={"subset": subset, "experiment": "attribution"},
                ))
                anns.append(Annotation(
                    case_id=cid, annotator_id=f"published-expert:{subset}", label=int(r["label"]),
                    provenance=LabelProvenance.HUMAN_EXPERT,
                    rubric_version="llm-aggrefact-annotation-guidelines",
                    reason="published expert annotation: 1 = claim supported by the source",
                ))
    audit["excluded_prior_exposure_documents"] = len(excluded_docs)
    audit["selected_cases"] = len(cases)
    return cases, anns, audit


def cmd_build(args: argparse.Namespace) -> int:
    existing = [DATA / name for name in DATA_ARTIFACTS if (DATA / name).exists()]
    if existing:
        print(
            "REFUSING: attribution build never overwrites an existing dataset or measurement "
            "artifact. Start a separately named experiment after auditing the existing files:\n  "
            + "\n  ".join(str(p) for p in existing),
            file=sys.stderr,
        )
        return 1
    protocol_arg = getattr(args, "protocol", None)
    if not protocol_arg:
        print(
            "REFUSING: the original attribution declaration did not govern the now-required "
            "full-source and prior-exposure safeguards. Supply --protocol PATH naming a new, "
            "predeclared experiment protocol; the original declaration remains immutable.",
            file=sys.stderr,
        )
        return 1
    protocol = pathlib.Path(protocol_arg).resolve()
    if not protocol.is_file():
        print(f"REFUSING: protocol file does not exist: {protocol}", file=sys.stderr)
        return 1
    if protocol == LEGACY_PROTOCOL.resolve():
        print(
            "REFUSING: the original declaration cannot be retrofitted to authorize a rebuilt "
            "corpus. Name a new protocol and a new experiment.",
            file=sys.stderr,
        )
        return 1
    cases, anns, audit = build()
    DATA.mkdir(parents=True, exist_ok=True)
    write_jsonl(DATA / "cases.jsonl", cases)
    write_jsonl(DATA / "annotations.jsonl", anns)
    sp = plan(cases, seed=SEED, weights={"train": 0.34, "dev": 0.33, "test": 0.33})
    print(json.dumps(verify(cases, sp), indent=2))
    sp.save(DATA / "splits.json")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_path": str(protocol),
        "protocol_hash": content_hash(protocol.read_text(encoding="utf-8")),
        "source_policy": "full source retained; no truncation",
        **audit,
    }
    (DATA / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    import collections
    per = collections.Counter((c.meta["subset"], a.label) for c, a in zip(cases, anns))
    print(f"\n{'subset':16s}{'pass':>6}{'fail':>6}")
    for s in SUBSETS:
        print(f"{s:16s}{per[(s, 1)]:>6}{per[(s, 0)]:>6}")
    print(f"\ntotal {len(cases)} cases, {len({c.group_id for c in cases})} groups")
    print(json.dumps(sp.stats["by_split"], indent=2))
    return 0


# ---------------------------------------------------------------- helpers
def _load() -> tuple[Corpus, SplitPlan, str]:
    corpus = Corpus.load(DATA)
    sp = SplitPlan.load(DATA / "splits.json")
    verify(corpus.cases, sp)
    return corpus, sp, content_hash(sorted(c.fingerprint for c in corpus.cases))


def _cache() -> JudgmentCache:
    return JudgmentCache(DATA / "judgment_cache.jsonl")


def _protocol_binding() -> dict[str, str]:
    """Validate and return the fresh-build protocol binding."""
    manifest_path = DATA / "build_manifest.json"
    if not manifest_path.exists():
        raise CalibrationError(
            "build_manifest.json is missing. This corpus predates the full-source and "
            "prior-exposure safeguards and cannot start or freeze a new run; preserve it for "
            "audit and create a new experiment under a new protocol."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CalibrationError(f"cannot read {manifest_path}: {exc}") from exc
    path_raw = manifest.get("protocol_path")
    recorded_hash = str(manifest.get("protocol_hash", ""))
    if not isinstance(path_raw, str) or not path_raw or not recorded_hash:
        raise CalibrationError("build manifest lacks protocol_path/protocol_hash")
    protocol = pathlib.Path(path_raw).resolve()
    if protocol == LEGACY_PROTOCOL.resolve():
        raise CalibrationError(
            "build manifest points at the immutable original declaration, which did not govern "
            "the full-source and prior-exposure safeguards"
        )
    if not protocol.is_file():
        raise CalibrationError(f"recorded protocol file is missing: {protocol}")
    current_hash = content_hash(protocol.read_text(encoding="utf-8"))
    if current_hash != recorded_hash:
        raise CalibrationError(
            f"recorded protocol changed after build ({recorded_hash} -> {current_hash})"
        )
    if manifest.get("source_policy") != "full source retained; no truncation":
        raise CalibrationError("build manifest does not attest the required full-source policy")
    return {"path": str(protocol), "hash": recorded_hash}


def _validate_runtime(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise CalibrationError("--concurrency must be at least 1")
    if args.budget < 0:
        raise CalibrationError("--budget must be non-negative")


def _candidate_configs() -> list[JudgeConfig]:
    """Resolve the declared grid once and fail closed on an ambiguous declaration."""
    by_id: dict[str, JudgeConfig] = {}
    duplicates: set[str] = set()
    for config in GRID:
        if config.variant_id in by_id:
            duplicates.add(config.variant_id)
        by_id[config.variant_id] = config
    if duplicates:
        raise CalibrationError(f"duplicate GRID variant ids: {sorted(duplicates)}")
    missing = [name for name in CANDIDATES if name not in by_id]
    if missing:
        raise CalibrationError(f"declared attribution candidates absent from GRID: {missing}")
    return [by_id[name] for name in CANDIDATES]


def _recorded_config(row: dict[str, Any]) -> JudgeConfig:
    """Reconstruct a dev judge from its record; never guess from today's GRID."""
    saved = row.get("judge_config")
    if not isinstance(saved, dict) or not saved:
        raise CalibrationError(
            f"dev record for {row.get('variant_id')!r} has no full judge_config. This older "
            "record cannot be frozen safely; run a new dev experiment rather than rebuilding "
            "its judge from the current GRID."
        )
    try:
        config = JudgeConfig.from_dict(saved)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(
            f"dev record for {row.get('variant_id')!r} has an invalid judge_config: {exc}"
        ) from exc
    if config.variant_id != row.get("variant_id"):
        raise CalibrationError(
            f"dev row variant_id {row.get('variant_id')!r} does not match its recorded config "
            f"{config.variant_id!r}"
        )
    recorded_hash = str(saved.get("config_hash", "") or row.get("config_hash", ""))
    if not recorded_hash:
        raise CalibrationError(f"dev record for {config.variant_id!r} has no config_hash")
    if config.config_hash != recorded_hash:
        raise CalibrationError(
            f"dev record for {config.variant_id!r} cannot be reconstructed: recorded "
            f"config_hash {recorded_hash}, current reconstruction {config.config_hash}. "
            "Run a new dev experiment; do not relabel old judgments as current."
        )
    recorded_prompt = str(row.get("prompt_template_hash", ""))
    if not recorded_prompt:
        raise CalibrationError(
            f"dev record for {config.variant_id!r} has no prompt_template_hash; its prompt "
            "provenance is incomplete, so it cannot be frozen"
        )
    if recorded_prompt != prompt_template_hash(config.mode):
        raise CalibrationError(
            f"dev record for {config.variant_id!r} has prompt_template_hash {recorded_prompt}, "
            f"but the reconstructed prompt is {prompt_template_hash(config.mode)}"
        )
    return config


def _refuse_after_freeze(command: str) -> str | None:
    """Protect the one-shot experiment before dev/freeze can mutate its provenance."""
    bundle_path = DATA / "calibration_bundle.json"
    report_path = DATA / "calibration_report.json"
    if bundle_path.exists():
        try:
            bundle = CalibrationBundle.load(bundle_path)
        except Exception as exc:  # noqa: BLE001 - corrupt protocol state must block mutation
            return f"REFUSING {command}: existing bundle is unreadable ({type(exc).__name__}: {exc})"
        if bundle.frozen:
            result = bundle.test_result if isinstance(bundle.test_result, dict) else {}
            completion_row = result.get("completion", {})
            completion = (completion_row.get("completion_rate")
                          if isinstance(completion_row, dict) else None)
            state = "incomplete " if isinstance(completion, (int, float)) and completion < 1 else ""
            return (
                f"REFUSING {command}: an {state}held-out measurement already exists. It must "
                "not be rebound, refrozen, or replaced. Preserve it and use an audited migration "
                "or a genuinely new experiment."
            )
        return (
            f"REFUSING {command}: a frozen, unmeasured bundle already exists. Resume `test` "
            "against that untouched configuration; do not regenerate dev or refreeze it."
        )
    if report_path.exists():
        return (
            f"REFUSING {command}: {report_path} exists without its bundle. Preserve and audit "
            "the orphaned measurement; do not silently replace it."
        )
    return None


# ---------------------------------------------------------------- dev
def cmd_dev(args: argparse.Namespace) -> int:
    refusal = _refuse_after_freeze("dev")
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    _validate_runtime(args)
    corpus, sp, dhash = _load()
    protocol = _protocol_binding()
    dev = select(corpus.cases, sp, "dev")
    labels = corpus.resolved_labels()
    cache, meter = _cache(), SpendMeter(cap_usd=args.budget)
    print(f"dev: {len(dev)} cases, candidates {list(CANDIDATES)}, cap ${args.budget:.2f}")

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    blocked = None
    for cfg in _candidate_configs():
        judgments, summary = run_variant(
            dev, cfg, cache=cache, meter=meter, concurrency=args.concurrency
        )
        p = pair(dev, labels, judgments, split="dev")
        best = None
        try:
            best = select_threshold(p, grid=THRESHOLDS)
        except CalibrationError as exc:
            print(f"  {cfg.variant_id}: threshold selection failed: {exc}", file=sys.stderr)
        rows.append({
            "variant_id": cfg.variant_id,
            "config_hash": cfg.config_hash,
            "judge_config": cfg.as_dict(),
            "prompt_template_hash": prompt_template_hash(cfg.mode),
            "n_paired": len(p),
            "completion": p.completion, "errors": p.errors,
            "best_threshold": best.threshold if best else None,
            "best_dev_kappa": best.dev_kappa if best else None,
            "threshold_grid": best.grid if best else [],
            "models_served": summary.models_served,
            "spend_after_usd": round(meter.spent_usd, 4),
        })
        summaries.append(summary.as_dict())
        k = rows[-1]["best_dev_kappa"]
        print(f"  {cfg.variant_id:24s} n={len(p):4d} completion={p.completion:.1%} "
              f"kappa={'n/a' if k is None else round(k, 4)} @t={rows[-1]['best_threshold']} "
              f"spend=${meter.spent_usd:.4f}")
        if summary.errors.get("quota_exhausted") or summary.errors.get("quota_stopped"):
            blocked = "quota_exhausted"
            print(
                "Provider quota exhausted; stopping remaining attribution variants. Cached "
                "outputs are retained for a later resume.",
                file=sys.stderr,
            )
            break

    out = {
        "dataset_hash": dhash,
        "annotations_hash": corpus.annotations_hash,
        "split_seed": sp.seed,
        "split_membership_hash": sp.membership_hash,
        "n_dev": len(dev),
        "protocol": protocol,
        "blocked": blocked,
        "spend": meter.as_dict(),
        "variants": rows,
        "run_summaries": summaries,
    }
    record_path = DATA / "dev_experiments.json"
    if record_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = record_path.with_name(f"dev_experiments.previous.{stamp}.json")
        archived.write_text(record_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"archived previous dev record to {archived}")
    record_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {DATA / 'dev_experiments.json'}  spend ${meter.spent_usd:.4f}")
    return 3 if blocked else 0


# ---------------------------------------------------------------- freeze
def cmd_freeze(args: argparse.Namespace) -> int:
    path = DATA / "calibration_bundle.json"
    refusal = _refuse_after_freeze("freeze")
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    corpus, sp, dhash = _load()
    protocol = _protocol_binding()
    record_path = DATA / "dev_experiments.json"
    if not record_path.exists():
        raise CalibrationError("no dev_experiments.json; run dev before freeze")
    exp = json.loads(record_path.read_text(encoding="utf-8"))
    if exp.get("blocked"):
        raise CalibrationError(
            f"dev record is blocked by {exp['blocked']}; resume dev before freezing"
        )
    for key, expected in (
        ("dataset_hash", dhash),
        ("annotations_hash", corpus.annotations_hash),
        ("split_seed", sp.seed),
        ("split_membership_hash", sp.membership_hash),
        ("protocol", protocol),
    ):
        if exp.get(key) != expected:
            raise CalibrationError(f"dev record {key} missing or changed; rerun dev before freeze")

    raw_rows = exp.get("variants")
    if not isinstance(raw_rows, list):
        raise CalibrationError("dev record variants is not a list")
    by_name: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, dict) or not isinstance(row.get("variant_id"), str):
            raise CalibrationError("dev record contains an invalid variant row")
        if row["variant_id"] in by_name:
            raise CalibrationError(f"dev record repeats variant {row['variant_id']!r}")
        by_name[row["variant_id"]] = row
    if set(by_name) != set(CANDIDATES):
        raise CalibrationError(
            f"dev record candidate set {sorted(by_name)} does not match the declaration "
            f"{sorted(CANDIDATES)}"
        )

    dev = select(corpus.cases, sp, "dev")
    labels = corpus.resolved_labels()
    cache = _cache()
    usable = []
    for name in CANDIDATES:
        row = by_name[name]
        cfg = _recorded_config(row)
        judgments = [
            judgment
            for case in dev
            if (judgment := cache.get(_cache_key(case, cfg))) is not None
        ]
        paired = pair(dev, labels, judgments, split="dev")
        if paired.completion < 0.98:
            continue
        expected_judge = {(cfg.variant_id, cfg.config_hash)}
        if set(paired.judges) != expected_judge:
            raise CalibrationError(
                f"cached dev judgments for {cfg.variant_id!r} do not all carry the recorded "
                f"variant/config identity {sorted(expected_judge)}"
            )
        try:
            choice = select_threshold(paired, grid=THRESHOLDS)
        except CalibrationError:
            continue
        usable.append((choice.dev_kappa, cfg, paired, choice))
    if not usable:
        print("no candidate reached 98% dev coverage with a defined kappa", file=sys.stderr)
        return 1
    _, cfg, paired, choice = max(usable, key=lambda item: item[0])
    frozen_config = cfg.as_dict()
    # JudgeConfig ignores this audit-only field when reconstructed, while bundle_id hashes the
    # complete mapping. This binds the frozen artifact to the successor experiment protocol
    # without pretending the protocol changes the judge's verdict/config_hash.
    frozen_config["experiment_protocol"] = protocol
    bundle = freeze(
        choice, variant_id=cfg.variant_id, judge_config=frozen_config, dataset_hash=dhash,
        split_seed=sp.seed, dev=paired, min_kappa=args.min_kappa,
        annotations_hash=corpus.annotations_hash,
        split_membership_hash=sp.membership_hash,
        notes=(
            f"Claim-level attribution experiment governed by {protocol['path']} "
            f"(hash {protocol['hash']}). Threshold and variant selected on dev only; test split "
            "untouched at freeze time."
        ),
    )
    bundle.save(path)
    print(f"froze {cfg.variant_id} at threshold {choice.threshold} "
          f"(dev kappa {choice.dev_kappa:.4f}, n={len(paired)})\nwrote {path}")
    return 0


# ---------------------------------------------------------------- test
def cmd_test(args: argparse.Namespace) -> int:
    path = DATA / "calibration_bundle.json"
    if not path.exists():
        print("no bundle; run freeze first", file=sys.stderr)
        return 1
    try:
        bundle = CalibrationBundle.load(path)
    except Exception as exc:  # noqa: BLE001 - malformed protocol state must fail closed
        print(f"cannot read frozen bundle: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if bundle.frozen:
        result = bundle.test_result if isinstance(bundle.test_result, dict) else {}
        completion_row = result.get("completion", {})
        completion = (completion_row.get("completion_rate")
                      if isinstance(completion_row, dict) else None)
        incomplete = isinstance(completion, (int, float)) and completion < 1
        article = "an incomplete" if incomplete else "a"
        print(
            f"REFUSING: this bundle already carries {article} measured test_result. The "
            "held-out split is measured once per frozen bundle. Preserve this artifact; an old "
            "incomplete measurement requires an audited migration or a genuinely new experiment, "
            "not a silent retry/rebind.",
            file=sys.stderr,
        )
        return 1
    report_path = DATA / "calibration_report.json"
    if report_path.exists():
        print(
            f"REFUSING: {report_path} already exists while the bundle is unmeasured. Audit the "
            "orphaned terminal artifact before any inference; it will not be overwritten.",
            file=sys.stderr,
        )
        return 1
    _validate_runtime(args)
    corpus, sp, dhash = _load()
    try:
        protocol = _protocol_binding()
    except CalibrationError as exc:
        print(f"protocol binding invalid: {exc}; refusing before inference", file=sys.stderr)
        return 1
    try:
        recomputed_bundle_id = expected_bundle_id(bundle)
    except Exception as exc:  # noqa: BLE001 - malformed identity must block before inference
        print(f"cannot recompute frozen bundle identity: {exc}", file=sys.stderr)
        return 1
    if not isinstance(bundle.judge_config, dict):
        print("frozen judge_config is not a mapping; refusing before inference", file=sys.stderr)
        return 1
    checks = {
        "bundle_id": (bundle.bundle_id, recomputed_bundle_id),
        "dataset_hash": (bundle.dataset_hash, dhash),
        "annotations_hash": (bundle.annotations_hash, corpus.annotations_hash),
        "split_seed": (bundle.split_seed, sp.seed),
        "split_membership_hash": (bundle.split_membership_hash, sp.membership_hash),
        "rubric_bundle_hash": (bundle.rubric_bundle_hash, bundle_hash()),
        "experiment_protocol": (bundle.judge_config.get("experiment_protocol"), protocol),
    }
    for name, (recorded, current) in checks.items():
        if recorded != current:
            print(
                f"frozen {name} missing or changed ({recorded!r} != {current!r}); refusing "
                "before inference",
                file=sys.stderr,
            )
            return 1

    try:
        cfg = JudgeConfig.from_dict(bundle.judge_config)
    except (TypeError, ValueError) as exc:
        print(f"cannot reconstruct frozen judge config: {exc}", file=sys.stderr)
        return 1
    stored = str(bundle.judge_config.get("config_hash", ""))
    if not stored:
        print("frozen judge config has no config_hash; refusing before inference", file=sys.stderr)
        return 1
    if cfg.config_hash != stored:
        print(f"FROZEN CONFIG MISMATCH: bundle {stored} vs reconstructed {cfg.config_hash}",
              file=sys.stderr)
        return 1
    if cfg.variant_id != bundle.variant_id:
        print(
            f"frozen variant_id {bundle.variant_id!r} != config {cfg.variant_id!r}; refusing "
            "before inference",
            file=sys.stderr,
        )
        return 1

    test = select(corpus.cases, sp, "test")
    labels = corpus.resolved_labels()
    meter = SpendMeter(cap_usd=args.budget)
    print(f"test: {len(test)} cases, frozen {cfg.variant_id} @ threshold {bundle.threshold}, "
          f"cap ${args.budget:.2f}")
    judgments, summary = run_variant(
        test, cfg, cache=_cache(), meter=meter, concurrency=args.concurrency
    )
    if summary.errors.get("quota_exhausted") or summary.errors.get("quota_stopped"):
        print(
            "Provider quota exhausted; cached outputs are retained, but the frozen bundle and "
            "report remain untouched. Resume the same frozen config after quota is restored.",
            file=sys.stderr,
        )
        return 3
    p = pair(test, labels, judgments, split="test")
    if p.completion < 1.0:
        print(
            f"INCOMPLETE held-out attempt: {len(p)} paired, {len(p.unjudged)} unjudged "
            f"({p.completion:.1%} completion; errors={summary.errors}). Cached outputs are "
            "retained, but no terminal bundle result or report was published. Resume the SAME "
            "untouched frozen config.",
            file=sys.stderr,
        )
        return 3
    result = evaluate(bundle, p, bootstrap=args.bootstrap)
    result["run_summary"] = summary.as_dict()
    bundle.test_result = result
    bundle.save(path)
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print()
    for name, row in sorted(result["tracks"].items()):
        if row.get("insufficient"):
            print(f"  {name:34s} n={row['n']:4d}  {row.get('note', '')}")
            continue
        ci = row["interval"]
        b = f"[{ci['low']:.3f}, {ci['high']:.3f}]" if ci.get("usable") else "unavailable"
        print(f"  {name:34s} n={row['n']:4d} kappa={row['kappa']:.4f} 95%CI {b} "
              f"agree={row['observed_agreement']:.3f} pass={row.get('passes_min_kappa')}")
    print(f"\ncompletion {result['completion']['completion_rate']:.1%}  spend ${meter.spent_usd:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("build", cmd_build), ("dev", cmd_dev), ("freeze", cmd_freeze), ("test", cmd_test)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--budget", type=float, default=12.0)
        p.add_argument("--concurrency", type=int, default=3)
        p.add_argument("--min-kappa", type=float, default=ROUTEBENCH_MIN_KAPPA)
        p.add_argument("--bootstrap", type=int, default=2000)
        if name == "build":
            p.add_argument(
                "--protocol",
                help="new predeclared protocol path; the original declaration cannot authorize "
                     "a rebuilt corpus",
            )
    args = ap.parse_args(argv)
    try:
        return int(args.fn(args))
    except CalibrationError as exc:
        print(f"calibration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

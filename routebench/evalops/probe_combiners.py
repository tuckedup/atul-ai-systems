"""Dev-internal probe: would combining the existing judges be worth a paid train run?

    python -m evalops.probe_combiners --out routebench/evalops/data/combiner_probe.json

**This is not the acceptance protocol and its numbers are not results.** The protocol fits a
combiner on TRAIN groups, selects on dev, and measures once on test. This probe fits and selects
inside DEV using grouped k-fold cross-validation, because the judgment cache contains dev
judgments only -- the train split was never judged, so there is nothing to fit on. Every number it
prints is a dev-internal cross-validated estimate, optimistic relative to a genuine held-out
measurement, and it is reported to answer one decision:

    is a combiner worth the provider spend of judging the train split, or does the strongest
    single judge already win?

It makes zero provider calls: it reads `judgment_cache.jsonl` and nothing else. Thresholds and
coefficients are chosen on the training folds of each CV split and applied to the held-out fold,
so the reported kappa is out-of-fold rather than in-sample; that is the weakest form of honesty
available without a train split, not a substitute for one.

Cases are grouped by `group_id` when folding, for the reason `splits.py` gives: sibling claims
about one source document are not independent evidence, and folding on `case_id` would let the
combiner see a near-duplicate of every item it is scored on.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibrate import THRESHOLD_GRID
from .combiners import (
    Combinable,
    CombinerError,
    Contender,
    LogisticSpec,
    MajorityVote,
    MeanScore,
    align,
    build_features,
    compare,
    fit_logistic,
)
from .dataset import Corpus, Judgment
from .metrics import agreement, binarize
from .splits import SplitPlan, select

DATA = Path(__file__).parent / "data"
DEFAULT_FOLDS = 5


# ---------------------------------------------------------------- loading cached judgments


def load_cached(
    cache_path: Path, *, variant_config: dict[str, str]
) -> tuple[list[Judgment], dict[str, int]]:
    """Read judgments from the cache, keeping only the config hash each variant was run at.

    The cache accumulates across corpus and rubric revisions, so a variant id can appear under
    several `config_hash` values. Mixing them would blend two different judges under one name, so
    anything whose hash does not match the recorded run is counted as stale and dropped.
    """
    kept: list[Judgment] = []
    skipped: Counter[str] = Counter()
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line).get("judgment")
        if not raw:
            continue
        variant = str(raw.get("variant_id", ""))
        expected = variant_config.get(variant)
        if expected is None:
            skipped["variant not in the run record"] += 1
            continue
        if str(raw.get("config_hash", "")) != expected:
            skipped["stale config_hash"] += 1
            continue
        kept.append(Judgment(**{k: v for k, v in raw.items() if k in Judgment.model_fields}))
    return kept, dict(skipped)


def group_folds(groups: list[str], n_folds: int) -> list[list[int]]:
    """Assign row indices to folds by group, largest group first, into the smallest fold.

    Deterministic: no RNG, no dependence on dict ordering. Greedy balancing keeps the folds close
    in size even when group sizes are uneven, which matters here because one source document can
    contribute a dozen groundedness claims.
    """
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for _, rows in sorted(by_group.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        target = min(range(n_folds), key=lambda f: (len(folds[f]), f))
        folds[target].extend(rows)
    return [sorted(f) for f in folds]


# ---------------------------------------------------------------- out-of-fold evaluation


@dataclass
class Row:
    case_id: str
    human: int
    group: str
    task: str
    provenance: str


def _best_threshold(human: list[int], scores: list[float]) -> tuple[float, float | None]:
    best_t, best_k = 0.5, None
    for t in THRESHOLD_GRID:
        a = agreement(human, binarize(scores, t), bootstrap=0)
        if a.kappa is not None and (best_k is None or a.kappa > best_k):
            best_k, best_t = a.kappa, t
    return best_t, best_k


def oof_threshold_only(
    rows: list[Row], scores: list[float], folds: list[list[int]]
) -> list[int]:
    """Out-of-fold predictions for a scorer with no fitted parameters but a tuned threshold."""
    predicted = [0] * len(rows)
    for fold in folds:
        held = set(fold)
        train = [i for i in range(len(rows)) if i not in held]
        t, _ = _best_threshold([rows[i].human for i in train], [scores[i] for i in train])
        for i in fold:
            predicted[i] = int(scores[i] >= t)
    return predicted


def oof_logistic(
    rows: list[Row],
    combinable: Combinable,
    index: dict[tuple[str, str], Judgment],
    spec: LogisticSpec,
    folds: list[list[int]],
) -> list[int] | None:
    """Out-of-fold predictions for a fitted combiner.

    The feature matrix is rebuilt from the training rows of each fold, so which criterion columns
    survive the presence filter is itself decided without the held-out fold. Rebuilding it once
    over everything would leak the held-out cases' criterion coverage into the column set.
    """
    predicted = [0] * len(rows)
    for fold in folds:
        held = set(fold)
        train_idx = [i for i in range(len(rows)) if i not in held]
        train_ids = tuple(rows[i].case_id for i in train_idx)
        train_part = Combinable(train_ids, {}, combinable.components)
        matrix = build_features(train_part, index, families=spec.families)
        if not matrix.names:
            return None
        labels = [rows[i].human for i in train_idx]
        groups = [rows[i].group for i in train_idx]
        if len(set(labels)) < 2:
            return None
        try:
            fitted = fit_logistic(spec, matrix, labels, groups, fitted_on="dev")
        except CombinerError:
            return None
        held_part = Combinable(tuple(rows[i].case_id for i in fold), {}, combinable.components)
        held_matrix = build_features(
            held_part, index, families=spec.families, min_criterion_presence=0.0
        )
        # Re-project the held-out rows onto the TRAINING column set: a criterion the held-out
        # cases happen to lack becomes 0, and one they have but training did not is discarded.
        col = {n: k for k, n in enumerate(held_matrix.names)}
        for local, i in enumerate(fold):
            row = [
                held_matrix.rows[local][col[n]] if n in col else 0.0
                for n in matrix.names
            ]
            predicted[i] = int(fitted.score_row(row) >= 0.5)
    return predicted


def score_rows(rows: list[Row], predicted: list[int], name: str, kind: str, threshold: float
               ) -> tuple[Contender, dict[str, Any]]:
    a = agreement(
        [r.human for r in rows], predicted,
        groups=[r.group for r in rows], tasks=[r.task for r in rows], bootstrap=0,
    )
    return (
        Contender(name=name, kappa=a.kappa, threshold=threshold, n=a.n, kind=kind),
        {"name": name, "kind": kind, "kappa": a.kappa, "defined": a.defined,
         "observed_agreement": a.observed_agreement, "n": a.n,
         "confusion": a.confusion.as_dict()},
    )


# ---------------------------------------------------------------- one track


def probe_track(
    title: str,
    rows: list[Row],
    combinable: Combinable,
    index: dict[tuple[str, str], Judgment],
    components: list[str],
    *,
    n_folds: int = DEFAULT_FOLDS,
) -> dict[str, Any]:
    folds = group_folds([r.group for r in rows], n_folds)
    contenders: list[Contender] = []
    detail: list[dict[str, Any]] = []

    def add(name: str, kind: str, predicted: list[int] | None, threshold: float = 0.5) -> None:
        if predicted is None:
            detail.append({"name": name, "kind": kind, "kappa": None,
                           "note": "not fittable on this track"})
            return
        c, d = score_rows(rows, predicted, name, kind, threshold)
        contenders.append(c)
        detail.append(d)

    # 1. every single component, threshold tuned out of fold -- the baselines a combiner must beat
    for v in components:
        scores = [float(index[(v, r.case_id)].raw_score or 0.0) for r in rows]
        add(v, "single", oof_threshold_only(rows, scores, folds))

    # 2. unweighted combiners
    ordered = Combinable(tuple(r.case_id for r in rows), {}, tuple(components))
    add("mean-score", "unweighted",
        oof_threshold_only(rows, MeanScore(tuple(components)).scores(ordered, index), folds))
    add("majority-vote", "unweighted",
        oof_threshold_only(rows, MajorityVote(tuple(components)).scores(ordered, index), folds))

    # 3. learned combiners over the declared feature families
    for families in (("score",), ("criteria",), ("score", "criteria"),
                     ("score", "criteria", "evidence")):
        for c_value in (0.1, 1.0):
            label = "+".join(families)
            add(f"logistic[{label}]C={c_value}", "learned",
                oof_logistic(
                    rows, combinable, index,
                    LogisticSpec(f"probe-{label}-{c_value}", tuple(components),
                                 families=families, C=c_value),
                    folds,
                ))

    try:
        comparison = compare(contenders).as_dict()
    except CombinerError as e:
        comparison = {"error": str(e)}
    return {
        "track": title,
        "n": len(rows),
        "n_groups": len({r.group for r in rows}),
        "human_prevalence": sum(r.human for r in rows) / len(rows) if rows else 0.0,
        "task_mix": dict(sorted(Counter(r.task for r in rows).items())),
        "folds": n_folds,
        "results": detail,
        "comparison": comparison,
    }


# ---------------------------------------------------------------- driver


def run(*, data: Path = DATA, n_folds: int = DEFAULT_FOLDS) -> dict[str, Any]:
    experiments = json.loads((data / "dev_experiments.json").read_text(encoding="utf-8"))
    variant_config = {
        v["variant_id"]: v["config_hash"]
        for v in experiments["variants"]
        if not v.get("skipped") and v.get("config_hash")
    }
    judgments, skipped = load_cached(data / "judgment_cache.jsonl",
                                    variant_config=variant_config)
    corpus = Corpus.load(data)
    plan = SplitPlan.load(data / "splits.json")
    labels = corpus.resolved_labels()
    dev_cases = [c for c in select(corpus.cases, plan, "dev") if c.case_id in labels]
    components = sorted(variant_config)

    combinable, index = align(judgments, components,
                              case_ids=[c.case_id for c in dev_cases])
    by_id = {c.case_id: c for c in dev_cases}
    rows = [
        Row(case_id=cid, human=int(labels[cid].label), group=by_id[cid].group_id,
            task=by_id[cid].task_class, provenance=labels[cid].provenance.value)
        for cid in combinable.case_ids
    ]

    tracks: list[dict[str, Any]] = []
    for title, keep in (
        ("all_admissible_labels", lambda r: True),
        ("human_judgment_of_response", lambda r: r.provenance.startswith("human_expert")
            or r.provenance.startswith("human_local")),
        ("gold_oracle", lambda r: r.provenance == "human_gold_reference_oracle"),
    ):
        subset = [r for r in rows if keep(r)]
        if len(subset) < 40:
            tracks.append({"track": title, "n": len(subset), "insufficient": True,
                           "note": "too few cases on this track to probe"})
            continue
        tracks.append(probe_track(title, subset, combinable, index, components,
                                 n_folds=n_folds))

    # Evidence-validity features can only carry information if the judgments were produced after
    # `Judgment.evidence_location` existed. Before that fix the signals were computed in
    # `judge.parse` and then dropped, so a cache written by the earlier code gives the evidence
    # family all-zero columns -- the hypothesis is untested rather than disproved, and the report
    # has to say which.
    with_evidence = sum(1 for j in judgments if j.evidence_location)
    evidence_note = (
        f"{with_evidence}/{len(judgments)} loaded judgments carry evidence_location. "
        + (
            "The evidence-validity feature family is therefore all-zero here: this probe does NOT "
            "test whether those signals help. Re-judging with the current code is required to "
            "test it."
            if with_evidence == 0 else
            "The evidence-validity feature family is populated for this fraction of cases."
        )
    )

    return {
        "probe": "dev-internal grouped cross-validation; NOT a held-out measurement",
        "evidence_features": {
            "judgments_with_evidence_location": with_evidence,
            "judgments_loaded": len(judgments),
            "note": evidence_note,
        },
        "protocol_deviation": (
            "The acceptance protocol fits on TRAIN and measures once on TEST. The judgment cache "
            "holds dev judgments only -- the train split was never judged -- so this probe fits "
            "and selects inside dev by grouped k-fold CV. Its numbers are optimistic relative to "
            "a held-out measurement and must not be used to freeze a bundle or to claim a kappa."
        ),
        "provider_calls": 0,
        "components": components,
        "coverage": combinable.as_dict(),
        "cache_entries_skipped": skipped,
        "n_dev_cases_with_labels": len(dev_cases),
        "tracks": tracks,
    }


def to_markdown(report: dict[str, Any]) -> str:
    out = [
        "# Combiner probe: dev-internal grouped cross-validation",
        "",
        "**Not a result.** " + report["protocol_deviation"],
        "",
        f"- Components: `{'`, `'.join(report['components'])}`",
        f"- Provider calls: {report['provider_calls']} (reads the judgment cache only)",
    ]
    cov = report["coverage"]
    out += [
        (f"- Joint coverage: {cov['n_complete']}/{cov['n_offered']} dev cases "
         f"({cov['coverage']:.1%}) have a usable judgment from every component"),
        f"- Missing judgments by component: `{cov['missing_by_component']}`",
        f"- Evidence features: {report['evidence_features']['note']}",
        "",
    ]
    for track in report["tracks"]:
        out.append(f"## {track['track']}")
        if track.get("insufficient"):
            out += ["", f"{track['note']} (n={track['n']}).", ""]
            continue
        out += [
            "",
            (f"n={track['n']} over {track['n_groups']} independent groups, "
             f"human prevalence {track['human_prevalence']:.3f}, "
             f"{track['folds']}-fold grouped CV. Task mix: `{track['task_mix']}`."),
            "",
            "| contender | kind | out-of-fold kappa | agreement | confusion |",
            "|---|---|---|---|---|",
        ]
        for r in track["results"]:
            if r.get("kappa") is None:
                out.append(f"| `{r['name']}` | {r['kind']} | — | — | {r.get('note', '')} |")
                continue
            c = r["confusion"]
            out.append(
                f"| `{r['name']}` | {r['kind']} | {r['kappa']:.4f} | "
                f"{r['observed_agreement']:.3f} | tp={c['tp']} fp={c['fp']} "
                f"fn={c['fn']} tn={c['tn']} |"
            )
        cmp_ = track["comparison"]
        if "error" in cmp_:
            out += ["", f"Comparison unavailable: {cmp_['error']}", ""]
            continue
        margin = cmp_["margin_over_best_single"]
        out += [
            "",
            (f"Strongest single component: **`{cmp_['best_single']['name']}`** "
             f"(kappa {cmp_['best_single']['kappa']:.4f}). "
             f"Best overall: **`{cmp_['best_overall']['name']}`** "
             f"(kappa {cmp_['best_overall']['kappa']:.4f}). "
             f"Margin: **{margin:+.4f}**. "
             f"Combiner helps: **{cmp_['combiner_helps']}**."),
            "",
        ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    ap.add_argument("--out", type=Path, default=DATA / "combiner_probe.json")
    ap.add_argument("--markdown", type=Path, default=DATA / "COMBINER_PROBE.md")
    args = ap.parse_args(argv)

    report = run(data=args.data, n_folds=args.folds)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.markdown.write_text(to_markdown(report), encoding="utf-8")

    cov = report["coverage"]
    print(f"components: {len(report['components'])}  joint coverage "
          f"{cov['n_complete']}/{cov['n_offered']} ({cov['coverage']:.1%})")
    for track in report["tracks"]:
        if track.get("insufficient"):
            print(f"  {track['track']:32s} {track['note']}")
            continue
        cmp_ = track["comparison"]
        if "error" in cmp_:
            print(f"  {track['track']:32s} {cmp_['error']}")
            continue
        print(f"  {track['track']:32s} n={track['n']:4d}  "
              f"best single {cmp_['best_single']['name']} "
              f"{cmp_['best_single']['kappa']:.4f}  ->  best overall "
              f"{cmp_['best_overall']['name']} {cmp_['best_overall']['kappa']:.4f}  "
              f"margin {cmp_['margin_over_best_single']:+.4f}")
    print(f"wrote {args.out} and {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

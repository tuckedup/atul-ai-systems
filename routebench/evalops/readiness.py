"""Conservative held-out readiness policy using development evidence only.

These are exploratory diagnostics, NOT a calibrated forecast of test success or a
proof of a model's attainable kappa. Grouped out-of-fold threshold/variant selection
cannot undo earlier prompt design or repeated inspection of these same dev labels.
It assesses a selection procedure, not necessarily the particular frozen winner.

Confusion-table reweighting assumes dev behavior and label prevalence transfer within
each stratum. The two-stage bootstrap further assumes exchangeable documents and
resamples observed within-document items. Its pass fraction is conditional on those
assumptions, with little evidence for strata represented by only a few documents.
Neither the interval nor the fraction includes all model-selection uncertainty.

READY is a conservative project policy, not a statistical theorem or a replacement
for the original acceptance metric. The five-document and pass-fraction cutoffs are
heuristics. Missing evidence can require an explicitly recorded owner override even
for a genuinely good model; NOT_READY does not mean 0.74 is impossible.

The CLI independently recomputes reports from cached dev evidence before test.
Content hashes alone protect against accidental drift, not an author who can edit
the evidence itself. Held-out labels must not be used to improve these diagnostics.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .calibrate import CalibrationError, Paired, select_threshold
from .dataset import CaseRecord, content_hash
from .metrics import MetricInputError, binarize
from .probe_combiners import group_folds

READINESS_PROTOCOL = "heldout-readiness-v2"
READY = "READY"
NOT_READY = "NOT_READY"

#: Fields excluded from `report_id`: when and where a report was computed is commentary, not
#: evidence, and hashing it would make every recomputation look like tampering.
_UNHASHED_FIELDS = frozenset({"report_id", "computed_at", "software"})


class ReadinessError(ValueError):
    """Malformed evidence. Raised, never converted into a verdict -- least of all READY."""


@dataclass(frozen=True)
class ReadinessRule:
    """The declared decision rule. Part of every report's identity; the gate accepts only this.

    `min_pass_probability`: legacy field name for a heuristic bootstrap scenario-fraction
    cutoff, not a calibrated probability. `min_dev_groups` and `min_stratum_weight`: see the module
    docstring. `folds`: document-grouped folds for the honest estimate. The bootstrap is seeded,
    so a report recomputes byte-identically from the same inputs.
    """

    min_pass_probability: float = 0.5
    min_dev_groups: int = 5
    min_stratum_weight: float = 0.10
    folds: int = 5
    replicates: int = 2000
    seed: int = 20260930
    level: float = 0.95

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def rule_hash(self) -> str:
        return content_hash(READINESS_PROTOCOL, self.as_dict())


DEFAULT_RULE = ReadinessRule()


# ---------------------------------------------------------------- label-blind held-out shape


def stratum_of(case: CaseRecord) -> str:
    """The unit the held-out mix is described in: the published subset when a case has one."""
    subset = case.meta.get("subset") if isinstance(case.meta, Mapping) else None
    if isinstance(subset, str) and subset.strip():
        return subset.strip()
    return case.task_class


@dataclass(frozen=True)
class HeldoutComposition:
    """Label-blind shape of the held-out population: stratum -> sizes of its documents."""

    group_sizes: dict[str, tuple[int, ...]]

    @property
    def n(self) -> int:
        return sum(sum(sizes) for sizes in self.group_sizes.values())

    def weights(self) -> dict[str, float]:
        total = self.n
        return {s: sum(sizes) / total for s, sizes in self.group_sizes.items()} if total else {}

    def as_dict(self) -> dict[str, Any]:
        w = self.weights()
        return {
            s: {"items": sum(sizes), "groups": len(sizes), "weight": w[s],
                "group_sizes": sorted(sizes, reverse=True)}
            for s, sizes in sorted(self.group_sizes.items())
        }


def heldout_composition(
    cases: Iterable[CaseRecord], *, stratum: Callable[[CaseRecord], str] = stratum_of
) -> HeldoutComposition:
    """Count held-out items per stratum and document. Takes `CaseRecord`s only.

    A case record carries no label -- labels live in `annotations.jsonl` by design -- so this
    function cannot see a held-out answer however it is called.
    """
    sizes: dict[str, dict[str, int]] = {}
    for case in cases:
        per = sizes.setdefault(stratum(case), {})
        per[case.group_id] = per.get(case.group_id, 0) + 1
    if not sizes:
        raise ReadinessError("the held-out population is empty; there is no measurement to prepare")
    return HeldoutComposition(
        {s: tuple(sorted(groups.values(), reverse=True)) for s, groups in sizes.items()}
    )


# ---------------------------------------------------------------- kappa on fractional tables


def kappa_from_cells(tp: float, fp: float, fn: float, tn: float) -> float | None:
    """Cohen's kappa from (possibly fractional) 2x2 cell masses; None when undefined.

    The same closed form as `metrics.kappa_of`, which takes integer counts. Reweighted tables are
    fractional, hence this twin; a test pins the two together on integer tables.
    """
    n = tp + fp + fn + tn
    if not n > 0:
        return None
    po = (tp + tn) / n
    judge_pos = (tp + fp) / n
    human_pos = (tp + fn) / n
    pe = judge_pos * human_pos + (1 - judge_pos) * (1 - human_pos)
    if math.isclose(pe, 1.0, abs_tol=1e-12):
        return None
    return (po - pe) / (1 - pe)


def _cell(human: int, judge: int) -> int:
    """Index into [tp, fp, fn, tn]. `tp` = both say pass, as in `metrics.Confusion`."""
    if human:
        return 0 if judge else 2
    return 1 if judge else 3


def cells_by_stratum(
    human: Sequence[int], judge: Sequence[int], strata: Sequence[str]
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for h, j, s in zip(human, judge, strata):
        out.setdefault(s, [0, 0, 0, 0])[_cell(h, j)] += 1
    return out


def reweighted_kappa(
    cells: Mapping[str, Sequence[float]], weights: Mapping[str, float]
) -> float | None:
    """Kappa of the mixture: each stratum's cell PROPORTIONS weighted by its held-out share.

    Kappas themselves are never averaged -- kappa does not combine across strata -- the 2x2
    table is reweighted and kappa computed once. None when a weighted stratum has no evidence.
    """
    total = [0.0, 0.0, 0.0, 0.0]
    for stratum, weight in weights.items():
        table = cells.get(stratum)
        n = sum(table) if table is not None else 0
        if table is None or n <= 0:
            return None
        for k in range(4):
            total[k] += weight * table[k] / n
    return kappa_from_cells(*total)


# ---------------------------------------------------------------- the honest estimate


@dataclass(frozen=True)
class Candidate:
    """One variant eligible for selection, with scores aligned to the evidence items."""

    variant_id: str
    scores: tuple[float, ...]


def _select(
    human: Sequence[int], groups: Sequence[str], candidates: Sequence[Candidate], rows: list[int]
) -> tuple[int, float, float]:
    """`cmd_freeze`'s rule on `rows`: `select_threshold` per candidate, then the best candidate.

    Uses `select_threshold` itself rather than a re-implementation, so the tie-breaks (closest to
    0.5, then the smaller threshold) cannot drift from the ones a real freeze applies.
    """
    best: tuple[int, float, float] | None = None
    for index, cand in enumerate(candidates):
        paired = Paired(
            case_ids=[str(i) for i in rows], human=[human[i] for i in rows],
            judge_scores=[cand.scores[i] for i in rows], groups=[groups[i] for i in rows],
            tasks=[""] * len(rows), provenance=[""] * len(rows), split="dev",
        )
        try:
            choice = select_threshold(paired)
        except CalibrationError:
            continue
        if best is None or choice.dev_kappa > best[1]:
            best = (index, choice.dev_kappa, choice.threshold)
    if best is None:
        raise ReadinessError(
            "no candidate produced a defined kappa on a training fold; the selection procedure "
            "cannot be reproduced on this dev set"
        )
    return best


def nested_predictions(
    human: Sequence[int],
    groups: Sequence[str],
    candidates: Sequence[Candidate],
    *,
    folds: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Out-of-fold predictions of the whole selection procedure (variant AND threshold).

    Each document-grouped fold is predicted by the candidate and threshold that the freeze rule
    picks on the other folds. The result estimates what "select on dev, then measure" delivers on
    documents the selection never saw, which is the quantity a held-out measurement tests.
    """
    if not candidates:
        raise ReadinessError("no selection candidates")
    n = len(human)
    for cand in candidates:
        if len(cand.scores) != n:
            raise ReadinessError(
                f"candidate {cand.variant_id!r} has {len(cand.scores)} scores for {n} items"
            )
        try:
            binarize(cand.scores, 0.5)  # finite and in [0, 1], or raise
        except MetricInputError as e:
            raise ReadinessError(f"candidate {cand.variant_id!r}: {e}") from e
    assignment = [f for f in group_folds(list(groups), folds) if f]
    if len(assignment) < 2:
        raise ReadinessError("need at least two non-empty document folds for an out-of-fold estimate")
    predicted = [0] * n
    log: list[dict[str, Any]] = []
    for k, fold in enumerate(assignment):
        held = set(fold)
        train = [i for i in range(n) if i not in held]
        index, train_kappa, threshold = _select(human, groups, candidates, train)
        chosen = candidates[index]
        for i, p in zip(fold, binarize([chosen.scores[i] for i in fold], threshold)):
            predicted[i] = p
        log.append({"fold": k, "n_train": len(train), "n_held": len(fold),
                    "variant_id": chosen.variant_id, "threshold": threshold,
                    "train_kappa": train_kappa})
    return predicted, log


# ---------------------------------------------------------------- assessment


@dataclass
class DevEvidence:
    """Labelled dev items of the gated track and the decisions available for them.

    `in_sample`: the frozen variant at the frozen threshold. `honest`: nested out-of-fold
    predictions of the selection procedure, or None when they could not be computed -- in which
    case the verdict cannot be READY. `unjudged`: labelled items of the track the frozen variant
    has no usable judgment for.
    """

    case_ids: list[str]
    human: list[int]
    groups: list[str]
    strata: list[str]
    in_sample: list[int]
    honest: list[int] | None = None
    unjudged: int = 0
    honest_note: str = ""
    selection: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        n = len(self.case_ids)
        columns: dict[str, Sequence[Any]] = {
            "human": self.human, "groups": self.groups, "strata": self.strata,
            "in_sample": self.in_sample,
        }
        if self.honest is not None:
            columns["honest"] = self.honest
        for name, column in columns.items():
            if len(column) != n:
                raise ReadinessError(f"evidence column {name!r} has {len(column)} rows for {n} items")
        for name in ("human", "in_sample", "honest"):
            binary = columns.get(name)
            if binary is None:
                continue
            bad = [v for v in binary if isinstance(v, bool) or v not in (0, 1)]
            if bad:
                raise ReadinessError(f"evidence column {name!r} holds non-binary values {bad[:3]}")
        if n == 0:
            raise ReadinessError("no judged dev items for the gated track")
        if len(set(self.case_ids)) != n:
            raise ReadinessError("duplicate case IDs in dev evidence")
        if self.unjudged < 0:
            raise ReadinessError("negative unjudged count")


def _percentile(values: list[float], level: float) -> tuple[float, float]:
    ordered = sorted(values)
    lo = max(0, math.floor((1 - level) / 2 * len(ordered)))
    hi = min(len(ordered) - 1, math.ceil((1 - (1 - level) / 2) * len(ordered)) - 1)
    return ordered[lo], ordered[hi]


def _simulate(
    dev_groups: Mapping[str, list[list[tuple[int, int, int | None]]]],
    heldout: HeldoutComposition,
    *,
    min_kappa: float,
    rule: ReadinessRule,
    with_honest: bool,
) -> dict[str, dict[str, Any]]:
    """Stratified two-stage document bootstrap. Common random numbers for both prediction sets."""
    rng = random.Random(rule.seed)
    weights = heldout.weights()
    strata = sorted(heldout.group_sizes)
    kinds = ("in_sample", "honest") if with_honest else ("in_sample",)
    param: dict[str, list[float]] = {k: [] for k in kinds}
    undefined: dict[str, int] = {k: 0 for k in kinds}
    passes: dict[str, int] = {k: 0 for k in kinds}
    for _ in range(rule.replicates):
        cells: dict[str, dict[str, list[int]]] = {k: {} for k in kinds}
        simulated: dict[str, list[int]] = {k: [0, 0, 0, 0] for k in kinds}
        for s in strata:
            pool = dev_groups[s]
            chosen = [pool[rng.randrange(len(pool))] for _ in pool]
            for kind in kinds:
                cells[kind][s] = [0, 0, 0, 0]
            for group in chosen:
                for h, a, b in group:
                    cells["in_sample"][s][_cell(h, a)] += 1
                    if with_honest:
                        cells["honest"][s][_cell(h, int(b or 0))] += 1
            for size in heldout.group_sizes[s]:
                group = chosen[rng.randrange(len(chosen))]
                for _ in range(size):
                    h, a, b = group[rng.randrange(len(group))]
                    simulated["in_sample"][_cell(h, a)] += 1
                    if with_honest:
                        simulated["honest"][_cell(h, int(b or 0))] += 1
        for kind in kinds:
            k = reweighted_kappa(cells[kind], weights)
            if k is None:
                undefined[kind] += 1
            else:
                param[kind].append(k)
            measured = kappa_from_cells(*simulated[kind])
            if measured is not None and measured >= min_kappa:
                passes[kind] += 1
    out: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        usable = bool(param[kind]) and undefined[kind] <= rule.replicates * 0.05
        low, high = _percentile(param[kind], rule.level) if usable else (None, None)
        out[kind] = {
            "interval": {"low": low, "high": high, "level": rule.level, "usable": usable,
                         "undefined_replicates": undefined[kind],
                         "method": "stratified document bootstrap, reweighted to held-out mix"},
            "p_pass": passes[kind] / rule.replicates,
        }
    return out


def assess(
    evidence: DevEvidence,
    heldout: HeldoutComposition,
    *,
    min_kappa: float,
    rule: ReadinessRule = DEFAULT_RULE,
) -> dict[str, Any]:
    """Compute the readiness verdict. Every failed condition is named in `reasons`."""
    evidence.validate()
    if not (isinstance(min_kappa, (int, float)) and math.isfinite(min_kappa)):
        raise ReadinessError(f"min_kappa {min_kappa!r} is not a finite number")
    reasons: list[str] = []
    notes: list[str] = []

    n = len(evidence.case_ids)
    completion = n / (n + evidence.unjudged)
    if evidence.unjudged:
        reasons.append(
            f"dev coverage is {completion:.1%}: {evidence.unjudged} labelled dev items of the gated "
            "track have no usable judgment from the frozen variant, so the evidence is a "
            "self-selected subset"
        )

    dev_groups: dict[str, dict[str, list[tuple[int, int, int | None]]]] = {}
    for i in range(n):
        honest = evidence.honest[i] if evidence.honest is not None else None
        dev_groups.setdefault(evidence.strata[i], {}).setdefault(evidence.groups[i], []).append(
            (evidence.human[i], evidence.in_sample[i], honest))

    weights = heldout.weights()
    missing = sorted(s for s in weights if s not in dev_groups)
    for s in missing:
        reasons.append(f"held-out stratum {s!r} ({weights[s]:.0%} of held-out items) has no dev "
                       "evidence at all")
    for s in sorted(weights):
        g = len(dev_groups.get(s, {}))
        if 0 < g < rule.min_dev_groups:
            message = (f"held-out stratum {s!r} carries {weights[s]:.0%} of the held-out items but "
                       f"rests on {g} dev document(s); at least {rule.min_dev_groups} are required "
                       f"for any stratum carrying >= {rule.min_stratum_weight:.0%}")
            if weights[s] >= rule.min_stratum_weight:
                reasons.append(message)
            else:
                notes.append(message.replace("are required", "would be required") +
                             " (below the weight floor, reported only)")

    in_cells = cells_by_stratum(evidence.human, evidence.in_sample, evidence.strata)
    estimates: dict[str, Any] = {
        "dev_kappa_in_sample": kappa_from_cells(*[sum(t[k] for t in in_cells.values())
                                                  for k in range(4)]),
        "reweighted_kappa_in_sample": reweighted_kappa(in_cells, weights),
        "reweighted_kappa_honest": None,
        "p_pass_in_sample": None,
        "p_pass_honest": None,
    }
    honest_cells: dict[str, list[int]] | None = None
    if evidence.honest is not None:
        honest_cells = cells_by_stratum(evidence.human, evidence.honest, evidence.strata)
        estimates["dev_kappa_honest"] = kappa_from_cells(*[sum(t[k] for t in honest_cells.values())
                                                           for k in range(4)])
        estimates["reweighted_kappa_honest"] = reweighted_kappa(honest_cells, weights)
    else:
        reasons.append("no honest (out-of-fold) estimate of the selection procedure: "
                       + (evidence.honest_note or "not computed"))

    if not missing:
        simulated = _simulate(
            {s: list(groups.values()) for s, groups in dev_groups.items() if s in weights},
            heldout, min_kappa=min_kappa, rule=rule, with_honest=evidence.honest is not None,
        )
        estimates["p_pass_in_sample"] = simulated["in_sample"]["p_pass"]
        estimates["reweighted_kappa_in_sample_interval"] = simulated["in_sample"]["interval"]
        if "honest" in simulated:
            estimates["p_pass_honest"] = simulated["honest"]["p_pass"]
            estimates["reweighted_kappa_honest_interval"] = simulated["honest"]["interval"]

    point = estimates["reweighted_kappa_in_sample"]
    if point is None or point < min_kappa:
        reasons.append(f"frozen-threshold dev kappa reweighted to the held-out mix is {_fmt(point)}, "
                       f"below min_kappa {min_kappa} even before correcting for selection")
    if evidence.honest is not None:
        honest_point = estimates["reweighted_kappa_honest"]
        if honest_point is None or honest_point < min_kappa:
            reasons.append(f"honest (out-of-fold) kappa reweighted to the held-out mix is "
                           f"{_fmt(honest_point)}, below min_kappa {min_kappa}")
        p = estimates["p_pass_honest"]
        if p is None or p < rule.min_pass_probability:
            reasons.append(f"conditional bootstrap fraction reaching "
                           f"{min_kappa} is {_fmt(p)}, below {rule.min_pass_probability}")

    by_stratum: dict[str, Any] = {}
    for s in sorted(set(evidence.strata)):
        row: dict[str, Any] = {
            "items": sum(len(g) for g in dev_groups[s].values()),
            "groups": len(dev_groups[s]),
            "heldout_weight": weights.get(s, 0.0),
            "in_sample_confusion": dict(zip(("tp", "fp", "fn", "tn"), in_cells[s])),
            "in_sample_kappa": kappa_from_cells(*in_cells[s]),
        }
        if honest_cells is not None:
            row["honest_confusion"] = dict(zip(("tp", "fp", "fn", "tn"), honest_cells[s]))
            row["honest_kappa"] = kappa_from_cells(*honest_cells[s])
        by_stratum[s] = row

    return {
        "verdict": NOT_READY if reasons else READY,
        "reasons": reasons,
        "notes": notes,
        "min_kappa": float(min_kappa),
        "estimates": estimates,
        "heldout_composition": heldout.as_dict(),
        "dev_evidence": {
            "n": n, "unjudged": evidence.unjudged, "completion": completion,
            "n_groups": len({g for groups in dev_groups.values() for g in groups}),
            "by_stratum": by_stratum,
        },
        "selection": dict(evidence.selection),
        "interpretation": {
            "kind": "exploratory_policy_diagnostic",
            "calibrated_test_pass_probability": False,
            "selection_bias_removed": False,
            "limitations": [
                "OOF replays bound candidates, not earlier adaptive prompt/model design.",
                "Reweighting assumes within-subset prevalence and error behavior transfer.",
                "Bootstrap resamples fixed predictions and sparse document evidence.",
                "Policy failure does not establish an upper bound on achievable kappa.",
            ],
        },
    }


def _fmt(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


# ---------------------------------------------------------------- the artifact and its gate


def report_digest(report: Mapping[str, Any]) -> str:
    return content_hash({k: v for k, v in report.items() if k not in _UNHASHED_FIELDS})


def finalize_report(
    core: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
    rule: ReadinessRule = DEFAULT_RULE,
    computed_at: str = "",
    software: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind an assessment to the bundle and data it describes, and give it a checkable id."""
    report: dict[str, Any] = {
        "protocol": READINESS_PROTOCOL,
        "rule": rule.as_dict(),
        "rule_hash": rule.rule_hash,
        "bindings": dict(bindings),
        **dict(core),
    }
    report["report_id"] = report_digest(report)
    report["computed_at"] = computed_at
    report["software"] = dict(software or {})
    return report


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def gate_problems(
    report: Any,
    *,
    bindings: Mapping[str, Any],
    min_kappa: float,
    rule: ReadinessRule = DEFAULT_RULE,
) -> list[str]:
    """Local report consistency checks, NOT standalone authorization for held-out inference.

    Treats the report as untrusted, as `validate_bundle` treats a bundle: the verdict is
    re-derived from the recorded estimates, the id is recomputed, and the bindings are compared
    with the bundle and corpus about to be measured. An empty list still requires the CLI's
    independent recomputation from source dev evidence; self-reported summaries can be forged.
    """
    if not isinstance(report, dict):
        return [f"readiness report is a {type(report).__name__}, not a JSON object"]
    problems: list[str] = []
    if report.get("protocol") != READINESS_PROTOCOL:
        problems.append(f"readiness protocol {report.get('protocol')!r} is not {READINESS_PROTOCOL!r}")
    if report.get("rule_hash") != rule.rule_hash or report.get("rule") != rule.as_dict():
        problems.append("report was computed under a different readiness rule than the declared one")
    try:
        recomputed = report_digest(report)
    except Exception as e:  # noqa: BLE001 - an unhashable report is a rejection, not a crash
        problems.append(f"cannot recompute report_id: {type(e).__name__}: {e}")
    else:
        if report.get("report_id") != recomputed:
            problems.append("report_id does not match the report's contents; it was edited after "
                            "it was computed")
    recorded = report.get("bindings")
    if not isinstance(recorded, dict):
        problems.append("report carries no bindings")
        recorded = {}
    for key, expected in bindings.items():
        if recorded.get(key) != expected:
            problems.append(f"report is bound to {key}={recorded.get(key)!r}; the measurement "
                            f"about to run has {expected!r}")
    used = _finite(report.get("min_kappa"))
    if used is None or used < min_kappa:
        problems.append(f"report assessed min_kappa={report.get('min_kappa')!r}, below the "
                        f"bundle's {min_kappa}")
    verdict = report.get("verdict")
    reasons = report.get("reasons")
    if verdict != READY:
        detail = "; ".join(str(r) for r in reasons) if isinstance(reasons, list) else ""
        problems.append(f"readiness verdict is {verdict!r}" + (f": {detail}" if detail else ""))
    else:
        problems += _ready_inconsistencies(report, min_kappa=min_kappa, rule=rule)
        if reasons:
            problems.append("report claims READY while listing reasons against it")
    return problems


def _ready_inconsistencies(
    report: Mapping[str, Any], *, min_kappa: float, rule: ReadinessRule
) -> list[str]:
    """Re-derive a READY verdict from the evidence the report records, condition by condition.

    `report_id` is a content hash, not a signature: it catches an edited report, not one whose id
    was recomputed after the edit. This catches inconsistent summaries only. The CLI must
    separately recompute from cached source predictions before accepting a consistent report.
    """
    out: list[str] = []
    raw_estimates = report.get("estimates")
    estimates: dict[str, Any] = raw_estimates if isinstance(raw_estimates, dict) else {}
    for key, floor in (("reweighted_kappa_in_sample", min_kappa),
                       ("reweighted_kappa_honest", min_kappa),
                       ("p_pass_honest", rule.min_pass_probability)):
        value = _finite(estimates.get(key))
        if value is None or value < floor:
            out.append(f"report claims READY but {key}={estimates.get(key)!r} is below {floor}; "
                       "the report is self-inconsistent")
    raw_dev = report.get("dev_evidence")
    dev: dict[str, Any] = raw_dev if isinstance(raw_dev, dict) else {}
    if dev.get("unjudged") != 0:
        out.append(f"report claims READY but records unjudged={dev.get('unjudged')!r} dev items; "
                   "the report is self-inconsistent")
    raw_strata = dev.get("by_stratum")
    strata: dict[str, Any] = raw_strata if isinstance(raw_strata, dict) else {}
    raw_mix = report.get("heldout_composition")
    mix: dict[str, Any] = raw_mix if isinstance(raw_mix, dict) else {}
    if not mix:
        out.append("report claims READY but records no held-out composition")
    for stratum, row in mix.items():
        weight = _finite(row.get("weight")) if isinstance(row, dict) else None
        evidence = strata.get(stratum)
        groups = evidence.get("groups") if isinstance(evidence, dict) else None
        if weight is None or not isinstance(groups, int) or isinstance(groups, bool) or \
                groups < 1 or (weight >= rule.min_stratum_weight and groups < rule.min_dev_groups):
            out.append(f"report claims READY but held-out stratum {stratum!r} is recorded with "
                       f"{groups!r} dev documents at weight {weight!r}; the report is "
                       "self-inconsistent")
    return out

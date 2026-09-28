"""Combine several judges' structured output into one binary decision.

Three mechanisms, in increasing order of how much they can go wrong:

1.  `MeanScore` / `MajorityVote` -- unweighted. Nothing is fitted, so nothing can be overfitted;
    the only free parameter is the threshold, which comes from dev like any other.
2.  `LogisticCombiner` -- L2-regularised logistic regression over per-criterion verdicts, each
    component's score, and the evidence-validity signals. Coefficients are fitted on TRAIN groups
    only.
3.  Neither, because the strongest single component already wins. `compare()` exists to make that
    outcome visible rather than discoverable, and it is the outcome the first measurement
    produced -- see `data/COMBINER_PROBE.md`.

Three design decisions worth stating, because each of them came out of a measurement rather than
a preference:

**Per task class, not pooled.** Criterion ids belong to a task's rubric, so `sql_result_set_equal`
simply does not exist on a summarization case. Pooling puts a mostly-absent column in the design
matrix for every criterion of every other task, and the fit degenerates: on the first dev probe a
pooled criterion-feature combiner scored kappa = -0.09, worse than chance, while the same features
fitted within the groundedness track scored 0.62. `TaskScoped` is therefore the default.

**No numpy, no sklearn.** `metrics.py` computes kappa in closed form for the same reason: the
production path stays dependency-light and exactly reproducible, and sklearn is used in the tests
as a cross-check so the hand-rolled version cannot drift. `_fit_logistic` minimises the same
objective `sklearn.linear_model.LogisticRegression(penalty="l2", C=...)` does, and
`tests/test_combiners.py` asserts they agree to 3 decimal places.

**Coverage is multiplicative, and that is usually fatal.** A combiner needs every component's
judgment for a case. Components fail independently, so an ensemble's completion rate is roughly
the product of its parts': eight variants at 96-100% each covered only 229 of 371 dev cases
(62%) because two of them lost a fifth of their judgments to provider rate limits. RouteBench's
release gate requires >=98% coverage on the primary track, so a combiner has to *earn* its
components. `Combinable.coverage` reports this and `require_coverage` refuses to hand back a
combiner that cannot meet the gate, instead of letting a good-looking kappa be computed on the
subset of cases where everything happened to succeed.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .dataset import Judgment

__all__ = [
    "Combinable",
    "CombinerError",
    "FeatureMatrix",
    "FittedLogistic",
    "LogisticSpec",
    "MajorityVote",
    "MeanScore",
    "build_features",
    "compare",
    "fit_logistic",
]


class CombinerError(RuntimeError):
    pass


# ---------------------------------------------------------------- assembling the inputs


@dataclass(frozen=True)
class Combinable:
    """Which cases every component judged, and which it did not.

    `coverage` is the number that decides whether a combiner is admissible at all. It is reported
    before anything is fitted, so an ensemble that cannot reach the release gate's coverage
    requirement is rejected on those grounds rather than on a kappa measured over the cases that
    happened to work.
    """

    #: Cases for which EVERY component produced a usable judgment, in sorted order.
    case_ids: tuple[str, ...]
    #: Cases at least one component is missing, mapped to the components that are missing it.
    missing: dict[str, tuple[str, ...]]
    components: tuple[str, ...]

    @property
    def n_offered(self) -> int:
        return len(self.case_ids) + len(self.missing)

    @property
    def coverage(self) -> float:
        return len(self.case_ids) / self.n_offered if self.n_offered else 0.0

    def as_dict(self) -> dict[str, Any]:
        worst: dict[str, int] = {}
        for absent in self.missing.values():
            for c in absent:
                worst[c] = worst.get(c, 0) + 1
        return {
            "components": list(self.components),
            "n_offered": self.n_offered,
            "n_complete": len(self.case_ids),
            "coverage": self.coverage,
            "missing_by_component": dict(sorted(worst.items())),
        }


def align(
    judgments: Iterable[Judgment],
    components: Sequence[str],
    *,
    case_ids: Sequence[str] | None = None,
) -> tuple[Combinable, dict[tuple[str, str], Judgment]]:
    """Index usable judgments by `(variant_id, case_id)` and report per-case completeness.

    `case_ids` is the universe of cases the combiner was asked about. Passing it is what makes
    `coverage` honest: without it, cases no component judged at all would silently vanish from
    the denominator, and an ensemble that answered a third of the split would report 100%
    coverage.
    """
    if not components:
        raise CombinerError("a combiner needs at least one component variant")
    wanted = set(components)
    index: dict[tuple[str, str], Judgment] = {}
    for j in judgments:
        if j.variant_id in wanted and j.usable:
            index[(j.variant_id, j.case_id)] = j

    universe = sorted(set(case_ids)) if case_ids is not None else sorted(
        {cid for (_, cid) in index}
    )
    complete: list[str] = []
    missing: dict[str, tuple[str, ...]] = {}
    for cid in universe:
        absent = tuple(v for v in components if (v, cid) not in index)
        if absent:
            missing[cid] = absent
        else:
            complete.append(cid)
    return Combinable(tuple(complete), missing, tuple(components)), index


def require_coverage(c: Combinable, minimum: float) -> None:
    """Refuse a combiner that cannot meet a coverage requirement.

    Raising here rather than reporting a low-coverage kappa is deliberate: a kappa computed over
    the cases where every component happened to answer is a kappa on a non-random subset, biased
    by whatever made the other cases fail. Long inputs are exactly what times out a judge and
    exactly what is hard to grade.
    """
    if c.coverage + 1e-12 < minimum:
        raise CombinerError(
            f"combiner over {list(c.components)} covers {len(c.case_ids)}/{c.n_offered} cases "
            f"({c.coverage:.1%}), below the required {minimum:.1%}. An ensemble's coverage is "
            "roughly the product of its components' -- drop a component, raise its completion "
            f"rate, or accept fewer. Missing: {c.as_dict()['missing_by_component']}"
        )


# ---------------------------------------------------------------- features


@dataclass(frozen=True)
class FeatureMatrix:
    """A design matrix with named columns. Column order is part of the fitted artifact."""

    case_ids: tuple[str, ...]
    names: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, keep: Sequence[int]) -> FeatureMatrix:
        return FeatureMatrix(
            tuple(self.case_ids[i] for i in keep),
            self.names,
            tuple(self.rows[i] for i in keep),
        )

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.names).encode()).hexdigest()[:16]


#: Feature families, each independently switchable so an ablation can name what it removed.
FEATURE_FAMILIES = ("score", "criteria", "evidence")


def _evidence_features(j: Judgment) -> dict[str, float]:
    """Evidence-validity signals, normalised so they do not scale with response length.

    These are the signals `judge.locate_evidence` produces. They were computed and then dropped
    before `Judgment` carried `evidence_location`, so no calibrator could ever have used them.
    `unverified_support_rate` is the one that encodes the groundedness failure specifically: a
    fact the judge called supported while citing a span that is not in the source document.
    """
    loc = j.evidence_location or {}
    total = float(loc.get("total", 0) or 0)
    checkable = total - float(loc.get("too_short", 0) or 0)
    n_crit = float(len(j.criteria)) or 1.0
    return {
        "ev_source_rate": (float(loc.get("in_source", 0)) / checkable) if checkable else 0.0,
        "ev_unlocated_rate": (float(loc.get("unlocated", 0)) / checkable) if checkable else 0.0,
        "ev_cited_any": 1.0 if total else 0.0,
        "unverified_support_rate": float(j.support_unverified) / n_crit,
    }


def build_features(
    combinable: Combinable,
    index: dict[tuple[str, str], Judgment],
    *,
    families: Sequence[str] = FEATURE_FAMILIES,
    min_criterion_presence: float = 0.95,
) -> FeatureMatrix:
    """Build the design matrix over the cases every component judged.

    `min_criterion_presence` drops criterion columns that are not present on nearly every case.
    Decompose-mode judgments name their criteria `fact_0`, `fact_1`, ... -- per-case identifiers
    that carry no cross-case meaning -- and a rubric criterion only exists for its own task
    class. Without this filter the matrix fills with columns that are zero almost everywhere, and
    the fit chases them. The threshold is high on purpose: a criterion the judge answers on 60% of
    cases is not a feature, it is two different features sharing a name.
    """
    unknown = set(families) - set(FEATURE_FAMILIES)
    if unknown:
        raise CombinerError(f"unknown feature families {sorted(unknown)}")
    ids = combinable.case_ids
    comps = combinable.components
    if not ids:
        return FeatureMatrix((), (), ())

    presence: dict[str, int] = {}
    if "criteria" in families:
        for cid in ids:
            for v in comps:
                for c in index[(v, cid)].criteria:
                    if isinstance(c, dict) and c.get("id"):
                        presence[f"{v}::crit::{c['id']}"] = presence.get(
                            f"{v}::crit::{c['id']}", 0
                        ) + 1
    keep_criteria = tuple(sorted(
        k for k, n in presence.items() if n >= min_criterion_presence * len(ids)
    ))

    names: list[str] = []
    if "score" in families:
        names += [f"{v}::score" for v in comps]
    names += list(keep_criteria)
    if "evidence" in families:
        names += [f"{v}::{k}" for v in comps for k in sorted(_evidence_features(Judgment(
            case_id="", variant_id="", model_requested="")))]

    rows: list[tuple[float, ...]] = []
    for cid in ids:
        row: dict[str, float] = {}
        for v in comps:
            j = index[(v, cid)]
            if "score" in families:
                row[f"{v}::score"] = float(j.raw_score or 0.0)
            if "criteria" in families:
                for c in j.criteria:
                    if isinstance(c, dict) and c.get("id"):
                        key = f"{v}::crit::{c['id']}"
                        if key in presence:
                            row[key] = 1.0 if c.get("verdict") else 0.0
            if "evidence" in families:
                for k, val in _evidence_features(j).items():
                    row[f"{v}::{k}"] = val
        rows.append(tuple(row.get(n, 0.0) for n in names))
    return FeatureMatrix(ids, tuple(names), tuple(rows))


# ---------------------------------------------------------------- unweighted combiners


@dataclass(frozen=True)
class MeanScore:
    """Mean of the components' scores. No fitting; the threshold still comes from dev."""

    components: tuple[str, ...]

    def scores(self, combinable: Combinable, index: dict[tuple[str, str], Judgment]) -> list[float]:
        out = []
        for cid in combinable.case_ids:
            vals = [float(index[(v, cid)].raw_score or 0.0) for v in self.components]
            out.append(sum(vals) / len(vals))
        return out


@dataclass(frozen=True)
class MajorityVote:
    """Fraction of components whose score clears `member_threshold`.

    Returned as a fraction rather than a hard verdict so the same dev threshold machinery that
    calibrates a single judge also chooses the vote rule: a threshold of 0.5 is "most of them",
    1.0 is unanimity, and anything in between is an m-of-k rule -- selected on dev, not asserted.
    """

    components: tuple[str, ...]
    member_threshold: float = 0.5

    def scores(self, combinable: Combinable, index: dict[tuple[str, str], Judgment]) -> list[float]:
        out = []
        for cid in combinable.case_ids:
            votes = sum(
                1 for v in self.components
                if float(index[(v, cid)].raw_score or 0.0) >= self.member_threshold
            )
            out.append(votes / len(self.components))
        return out


# ---------------------------------------------------------------- logistic combiner


@dataclass(frozen=True)
class LogisticSpec:
    """A predeclared combiner candidate. Declared before fitting, like the judge variant grid."""

    combiner_id: str
    components: tuple[str, ...]
    families: tuple[str, ...] = FEATURE_FAMILIES
    #: Inverse regularisation strength, matching sklearn's `C`. Small C = strong shrinkage, which
    #: is what a few hundred grouped training items wants.
    C: float = 1.0
    task_class: str = ""
    notes: str = ""


@dataclass
class FittedLogistic:
    """Fitted coefficients plus everything needed to reproduce and to refuse misuse."""

    combiner_id: str
    components: list[str]
    feature_names: list[str]
    coefficients: list[float]
    intercept: float
    C: float
    task_class: str
    n_train: int
    n_train_groups: int
    feature_spec_hash: str
    #: Which split the coefficients were fitted on. Recorded so an artifact fitted on dev -- or,
    #: worse, on test -- is identifiable after the fact rather than indistinguishable.
    fitted_on: str = "train"
    converged: bool = True
    iterations: int = 0
    notes: str = ""

    def score_row(self, row: Sequence[float]) -> float:
        if len(row) != len(self.coefficients):
            raise CombinerError(
                f"combiner {self.combiner_id} expects {len(self.coefficients)} features, got "
                f"{len(row)}"
            )
        z = self.intercept + sum(c * x for c, x in zip(self.coefficients, row, strict=True))
        return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z))))

    def scores(self, matrix: FeatureMatrix) -> list[float]:
        if matrix.names and tuple(matrix.names) != tuple(self.feature_names):
            raise CombinerError(
                f"combiner {self.combiner_id} was fitted on a different feature set "
                f"(spec {self.feature_spec_hash}); refusing to score a matrix whose columns do "
                "not match, because the coefficients would be applied to the wrong features"
            )
        return [self.score_row(r) for r in matrix.rows]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FittedLogistic:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. Small, dense, and exactly reproducible."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise CombinerError(
                "the feature matrix is singular even after ridge regularisation; two feature "
                "columns are identical, or a column is constant across every training row"
            )
        a[col], a[pivot] = a[pivot], a[col]
        inv = 1.0 / a[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col] * inv
            if factor:
                for c in range(col, n + 1):
                    a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def _fit_logistic(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    C: float,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[list[float], float, bool, int]:
    """L2-regularised logistic regression by IRLS / Newton-Raphson.

    Minimises the same objective as `sklearn.linear_model.LogisticRegression(penalty="l2", C=C)`:

        0.5 * w'w  +  C * sum_i log(1 + exp(-y_i (x_i'w + b)))

    with the intercept unpenalised. The cross-check against sklearn lives in the tests; this is
    here so `evalops` keeps no numpy/sklearn dependency on the production path, exactly as
    `metrics.py` hand-rolls kappa for the same reason.
    """
    n = len(rows)
    if n == 0:
        raise CombinerError("no training rows")
    d = len(rows[0])
    if any(len(r) != d for r in rows):
        raise CombinerError("ragged feature matrix")
    if len(labels) != n:
        raise CombinerError(f"{n} rows but {len(labels)} labels")
    if len(set(labels)) < 2:
        raise CombinerError(
            "training labels are single-class; a combiner fitted on one class predicts that "
            "class everywhere and its agreement is an artefact of prevalence"
        )

    # Design matrix with an intercept column appended (index d), which is not penalised.
    x = [list(r) + [1.0] for r in rows]
    p_dim = d + 1
    w = [0.0] * p_dim
    converged = False
    used = 0
    for used in range(1, max_iter + 1):
        mu = []
        for row in x:
            z = sum(wi * xi for wi, xi in zip(w, row, strict=True))
            mu.append(1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z)))))
        # gradient of the objective, negated: C * X'(y - mu) - penalty * w
        grad = [0.0] * p_dim
        for i, row in enumerate(x):
            resid = C * (labels[i] - mu[i])
            if resid:
                for k in range(p_dim):
                    grad[k] += resid * row[k]
        for k in range(d):  # intercept (index d) is unpenalised
            grad[k] -= w[k]
        # Hessian: penalty * I + C * X' W X, W = mu(1-mu), floored for numerical stability
        hess = [[0.0] * p_dim for _ in range(p_dim)]
        for i, row in enumerate(x):
            weight = C * max(mu[i] * (1.0 - mu[i]), 1e-10)
            for a_i in range(p_dim):
                if row[a_i] == 0.0:
                    continue
                wa = weight * row[a_i]
                for b_i in range(a_i, p_dim):
                    hess[a_i][b_i] += wa * row[b_i]
        for a_i in range(p_dim):
            for b_i in range(a_i + 1, p_dim):
                hess[b_i][a_i] = hess[a_i][b_i]
        for k in range(d):
            hess[k][k] += 1.0
        hess[d][d] += 1e-10  # keep the unpenalised intercept row from going singular
        step = _solve(hess, grad)
        w = [wi + s for wi, s in zip(w, step, strict=True)]
        if max(abs(s) for s in step) < tol:
            converged = True
            break
    return w[:d], w[d], converged, used


def fit_logistic(
    spec: LogisticSpec,
    matrix: FeatureMatrix,
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    fitted_on: str = "train",
) -> FittedLogistic:
    """Fit `spec` on `matrix`. `fitted_on` must name the split the rows came from.

    Passing `fitted_on="test"` raises. There is no legitimate reason to fit a combiner on held-out
    data, and a fitted artifact that does not record which split it saw is indistinguishable from
    one that does.
    """
    if fitted_on == "test":
        raise CombinerError(
            "refusing to fit a combiner on the test split. Coefficients fitted on held-out data "
            "make the held-out measurement a training score."
        )
    if not matrix.rows:
        raise CombinerError(f"combiner {spec.combiner_id}: no rows to fit")
    if len(labels) != len(matrix):
        raise CombinerError(f"{len(matrix)} rows but {len(labels)} labels")
    coefs, intercept, converged, iterations = _fit_logistic(matrix.rows, labels, C=spec.C)
    return FittedLogistic(
        combiner_id=spec.combiner_id,
        components=list(spec.components),
        feature_names=list(matrix.names),
        coefficients=coefs,
        intercept=intercept,
        C=spec.C,
        task_class=spec.task_class,
        n_train=len(matrix),
        n_train_groups=len(set(groups)),
        feature_spec_hash=matrix.spec_hash,
        fitted_on=fitted_on,
        converged=converged,
        iterations=iterations,
        notes=spec.notes,
    )


# ---------------------------------------------------------------- comparison


@dataclass(frozen=True)
class Contender:
    name: str
    kappa: float | None
    threshold: float
    n: int
    kind: str  # "single" | "unweighted" | "learned"


@dataclass(frozen=True)
class Comparison:
    """A combiner is only worth its cost if it beats its strongest single component.

    The research summary that prompted this work is explicit that ensembles improve some datasets
    and not others, so "the ensemble won" has to be a measured claim about a named baseline. This
    type exists so the baseline cannot be omitted from the report.
    """

    contenders: tuple[Contender, ...]
    best_single: Contender
    best_overall: Contender

    @property
    def margin(self) -> float | None:
        if self.best_overall.kappa is None or self.best_single.kappa is None:
            return None
        return self.best_overall.kappa - self.best_single.kappa

    @property
    def combiner_helps(self) -> bool:
        return self.best_overall.kind != "single" and (self.margin or 0.0) > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "contenders": [asdict(c) for c in self.contenders],
            "best_single": asdict(self.best_single),
            "best_overall": asdict(self.best_overall),
            "margin_over_best_single": self.margin,
            "combiner_helps": self.combiner_helps,
        }


def compare(contenders: Sequence[Contender]) -> Comparison:
    usable = [c for c in contenders if c.kappa is not None]
    if not usable:
        raise CombinerError("no contender produced a defined kappa")
    singles = [c for c in usable if c.kind == "single"]
    if not singles:
        raise CombinerError(
            "no single-component contender was supplied. A combiner's kappa is uninterpretable "
            "without the strongest individual judge to compare it against."
        )
    # Deterministic tie-breaking: higher kappa, then prefer the simpler kind, then the name. A
    # tie must not be resolved in the combiner's favour -- "as good as the single judge, for eight
    # times the cost" is a loss.
    rank = {"single": 0, "unweighted": 1, "learned": 2}

    def key(c: Contender) -> tuple[float, int, str]:
        return (-(c.kappa or 0.0), rank[c.kind], c.name)

    return Comparison(
        contenders=tuple(contenders),
        best_single=min(singles, key=key),
        best_overall=min(usable, key=key),
    )

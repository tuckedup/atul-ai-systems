"""Agreement metrics for judge calibration.

Design notes
------------
1. Cohen's kappa is computed from the 2x2 table in closed form rather than read off
   `sklearn.metrics.cohen_kappa_score`, because the degenerate case matters more than the
   happy path. When both raters are constant and agree, expected agreement `pe` is 1.0 and
   kappa is 0/0. sklearn returns `nan`; `nan >= 0.74` is False but `nan < 0.74` is *also*
   False, so a release gate written as `if kappa < threshold: reject` lets `nan` straight
   through. That was the live bug in `evalops/calibrate.py`. Here `defined` is an explicit
   field and `Agreement.passes()` requires it.

2. The confidence interval resamples *groups*, not items. Several candidate responses can
   share one source prompt; resampling items would treat those as independent evidence and
   report an interval that is too narrow.

3. sklearn is still used as a cross-check in the tests, so the closed form cannot drift.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

Label = int  # 0 or 1


@dataclass(frozen=True)
class Confusion:
    """Judge-vs-human 2x2 table. `tp` = both say pass."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def as_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn, "n": self.n}


@dataclass(frozen=True)
class Interval:
    low: float | None = None
    high: float | None = None
    level: float = 0.95
    method: str = "group-bootstrap-percentile"
    replicates: int = 0
    undefined_replicates: int = 0
    usable: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Agreement:
    """Full agreement record. `defined is False` means kappa does not exist for this table."""

    kappa: float | None
    defined: bool
    observed_agreement: float
    expected_agreement: float
    confusion: Confusion
    human_prevalence: float
    judge_prevalence: float
    n: int
    n_groups: int
    reason: str = ""
    interval: Interval = field(default_factory=Interval)
    per_task: dict[str, dict[str, Any]] = field(default_factory=dict)

    def passes(self, min_kappa: float) -> bool:
        """Fail closed: an undefined or missing kappa never passes a release gate."""
        return self.defined and self.kappa is not None and self.kappa >= min_kappa

    def as_dict(self) -> dict[str, Any]:
        return {
            "kappa": self.kappa,
            "defined": self.defined,
            "observed_agreement": self.observed_agreement,
            "expected_agreement": self.expected_agreement,
            "confusion": self.confusion.as_dict(),
            "human_prevalence": self.human_prevalence,
            "judge_prevalence": self.judge_prevalence,
            "n": self.n,
            "n_groups": self.n_groups,
            "reason": self.reason,
            "interval": self.interval.as_dict(),
            "per_task": self.per_task,
        }


class MetricInputError(ValueError):
    """Raised for malformed inputs. Never coerced into a score."""


def _validate(human: Sequence[Any], judge: Sequence[Any]) -> tuple[list[Label], list[Label]]:
    if len(human) != len(judge):
        raise MetricInputError(f"length mismatch: human={len(human)} judge={len(judge)}")
    if not human:
        raise MetricInputError("no labelled items")
    h: list[Label] = []
    j: list[Label] = []
    for index, (a, b) in enumerate(zip(human, judge)):
        for name, value in (("human", a), ("judge", b)):
            if isinstance(value, bool):
                continue
            if not isinstance(value, int) or value not in (0, 1):
                raise MetricInputError(
                    f"{name} label at index {index} is {value!r}; binary 0/1 labels are required"
                )
        h.append(int(a))
        j.append(int(b))
    return h, j


def confusion_of(human: Sequence[Any], judge: Sequence[Any]) -> Confusion:
    h, j = _validate(human, judge)
    return Confusion(
        tp=sum(1 for a, b in zip(h, j) if a == 1 and b == 1),
        fp=sum(1 for a, b in zip(h, j) if a == 0 and b == 1),
        fn=sum(1 for a, b in zip(h, j) if a == 1 and b == 0),
        tn=sum(1 for a, b in zip(h, j) if a == 0 and b == 0),
    )


def kappa_of(c: Confusion) -> tuple[float | None, float, float, str]:
    """Closed-form Cohen's kappa. Returns (kappa or None, p_observed, p_expected, reason)."""
    n = c.n
    if n == 0:
        return None, 0.0, 0.0, "empty table"
    po = (c.tp + c.tn) / n
    judge_pos = (c.tp + c.fp) / n
    human_pos = (c.tp + c.fn) / n
    pe = judge_pos * human_pos + (1 - judge_pos) * (1 - human_pos)
    if math.isclose(pe, 1.0, abs_tol=1e-12):
        return (
            None,
            po,
            pe,
            (
                "kappa undefined: expected agreement is 1.0 because both raters are constant and "
                "identical (single-class table); chance-corrected agreement does not exist"
            ),
        )
    return (po - pe) / (1 - pe), po, pe, ""


def _bootstrap(
    items: list[tuple[str, Label, Label]],
    *,
    replicates: int,
    seed: int,
    level: float,
) -> Interval:
    """Percentile CI over groups resampled with replacement."""
    groups: dict[str, list[tuple[Label, Label]]] = {}
    for group_id, h, j in items:
        groups.setdefault(group_id, []).append((h, j))
    keys = sorted(groups)
    if len(keys) < 10:
        return Interval(
            None,
            None,
            level,
            replicates=0,
            usable=False,
            note=(
                "insufficient independent groups for a bootstrap interval "
                f"(n_groups={len(keys)}, need >=10)"
            ),
        )
    rng = random.Random(seed)
    values: list[float] = []
    undefined = 0
    for _ in range(replicates):
        tp = fp = fn = tn = 0
        for _ in keys:
            for h, j in groups[keys[rng.randrange(len(keys))]]:
                if h and j:
                    tp += 1
                elif not h and j:
                    fp += 1
                elif h and not j:
                    fn += 1
                else:
                    tn += 1
        k, _po, _pe, _reason = kappa_of(Confusion(tp, fp, fn, tn))
        if k is None:
            undefined += 1
        else:
            values.append(k)
    if not values or undefined > replicates * 0.05:
        return Interval(
            None,
            None,
            level,
            replicates=replicates,
            undefined_replicates=undefined,
            usable=False,
            note=(
                f"{undefined}/{replicates} replicates had undefined kappa; the sample is too "
                "degenerate to support an interval"
            ),
        )
    values.sort()
    lo_index = max(0, math.floor((1 - level) / 2 * len(values)))
    hi_index = min(len(values) - 1, math.ceil((1 - (1 - level) / 2) * len(values)) - 1)
    return Interval(
        values[lo_index],
        values[hi_index],
        level,
        replicates=replicates,
        undefined_replicates=undefined,
        usable=True,
    )


def agreement(
    human: Sequence[Any],
    judge: Sequence[Any],
    *,
    groups: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    bootstrap: int = 2000,
    seed: int = 20260927,
    level: float = 0.95,
) -> Agreement:
    """Compute the full agreement record.

    `groups` identifies independent sampling units (source document / prompt family). Items
    sharing a group are resampled together. Defaults to one group per item.
    """
    h, j = _validate(human, judge)
    group_ids = list(groups) if groups is not None else [f"i{n}" for n in range(len(h))]
    if len(group_ids) != len(h):
        raise MetricInputError(f"groups length {len(group_ids)} != labels length {len(h)}")
    c = confusion_of(h, j)
    k, po, pe, reason = kappa_of(c)
    items = list(zip(group_ids, h, j))
    interval = (
        _bootstrap(items, replicates=bootstrap, seed=seed, level=level)
        if bootstrap and k is not None
        else Interval(None, None, level, usable=False, note="interval not requested or kappa undefined")
    )

    per_task: dict[str, dict[str, Any]] = {}
    if tasks is not None:
        if len(tasks) != len(h):
            raise MetricInputError(f"tasks length {len(tasks)} != labels length {len(h)}")
        for task in sorted(set(tasks)):
            idx = [n for n, t in enumerate(tasks) if t == task]
            tc = confusion_of([h[n] for n in idx], [j[n] for n in idx])
            tk, tpo, _tpe, treason = kappa_of(tc)
            per_task[task] = {
                "n": tc.n,
                "kappa": tk,
                "defined": tk is not None,
                "observed_agreement": tpo,
                "confusion": tc.as_dict(),
                "reason": treason
                or ("" if tc.n >= 20 else f"small stratum (n={tc.n}); indicative only"),
            }

    return Agreement(
        kappa=k,
        defined=k is not None,
        observed_agreement=po,
        expected_agreement=pe,
        confusion=c,
        human_prevalence=(c.tp + c.fn) / c.n,
        judge_prevalence=(c.tp + c.fp) / c.n,
        n=c.n,
        n_groups=len(set(group_ids)),
        reason=reason,
        interval=interval,
        per_task=per_task,
    )


def binarize(scores: Sequence[float], threshold: float) -> list[Label]:
    """Turn continuous judge scores into pass/fail. Applies to JUDGE scores only.

    Human labels are collected as binary and are never re-thresholded; doing so would move
    the target while tuning the predictor, which is the defect this module exists to fix.
    """
    out: list[Label] = []
    for index, s in enumerate(scores):
        value = float(s)
        if not math.isfinite(value):
            raise MetricInputError(f"judge score at index {index} is {s!r}; scores must be finite")
        if not 0.0 <= value <= 1.0:
            raise MetricInputError(f"judge score at index {index} is {value}; expected range [0, 1]")
        out.append(int(value >= threshold))
    return out

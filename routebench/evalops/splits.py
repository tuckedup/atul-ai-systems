"""Deterministic, group-aware train/dev/test splits.

Why groups: several cases can derive from one source document or one prompt family. If a
CNN article appears in train and a different claim about the same article appears in test,
the judge has effectively seen the test material. Splitting on `group_id` instead of
`case_id` closes that path, at the cost of less exact split sizes.

Why deterministic: each group's split is a pure function of `(seed, task_class, group_id)` --
its own hash bucket compared against fixed cutoffs -- with no dependence on how many other
groups exist. Adding cases to the corpus therefore cannot reassign a group that was already
assigned, so the frozen test set stays frozen as the corpus grows. The cost is that split
proportions are only approximate for small strata; `plan().stats` reports what was realised.

The test split is treated as write-once. `verify()` is what CI runs; it asserts no group and
no content fingerprint crosses a split boundary.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .dataset import CaseRecord

Split = Literal["train", "dev", "test"]
SPLITS: tuple[Split, ...] = ("train", "dev", "test")


class SplitError(ValueError):
    pass


def _bucket(seed: int, group_id: str) -> float:
    """Map a group to [0, 1) stably. Independent of corpus size and insertion order."""
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass
class SplitPlan:
    assignment: dict[str, Split]           # group_id -> split
    weights: dict[str, float]
    seed: int
    policy: str = "sha256(seed:group_id) bucketed; stratified by task_class"
    stats: dict[str, Any] = field(default_factory=dict)

    def of_case(self, case: CaseRecord) -> Split:
        try:
            return self.assignment[case.group_id]
        except KeyError as e:
            raise SplitError(
                f"case {case.case_id} has unassigned group {case.group_id!r}; "
                "regenerate the split plan rather than defaulting it into a split"
            ) from e

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "seed": self.seed,
                    "policy": self.policy,
                    "weights": self.weights,
                    "assignment": self.assignment,
                    "stats": self.stats,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> SplitPlan:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            assignment={k: v for k, v in raw["assignment"].items()},
            weights=raw["weights"],
            seed=raw["seed"],
            policy=raw.get("policy", ""),
            stats=raw.get("stats", {}),
        )


def plan(
    cases: Iterable[CaseRecord],
    *,
    seed: int = 20260927,
    weights: dict[str, float] | None = None,
) -> SplitPlan:
    """Assign every group to a split, stratified within each task class.

    Stratifying per task class stops a whole task family landing in one split, which would
    make per-task kappa uncomputable for the others.
    """
    w = weights or {"train": 0.34, "dev": 0.33, "test": 0.33}
    if abs(sum(w.values()) - 1.0) > 1e-9:
        raise SplitError(f"split weights must sum to 1.0, got {w} summing to {sum(w.values())}")
    if set(w) != set(SPLITS):
        raise SplitError(f"split weights must cover exactly {SPLITS}, got {sorted(w)}")

    cases = list(cases)
    # A group may (rarely) carry cases from more than one task class; stratify by the task
    # class of its first case so the group stays intact.
    group_task: dict[str, str] = {}
    for c in cases:
        group_task.setdefault(c.group_id, c.task_class)

    assignment: dict[str, Split] = {}
    train_cut = w["train"]
    dev_cut = w["train"] + w["dev"]
    for task in sorted(set(group_task.values())):
        for g in sorted(gid for gid, t in group_task.items() if t == task):
            # Compare each group's own bucket against fixed cumulative cutoffs. The task class is
            # folded into the hash so strata are independently balanced.
            #
            # An earlier version ranked a task's groups by bucket and cut the ranked list at
            # `round(n * weight)`. That gives exact proportions on small strata, but the cut index
            # is a function of `n`, so adding groups to an existing task shifts the boundary and
            # can move a previously-assigned group into a different split even though its own hash
            # never changed. Reproduced: 30 code + 30 sql groups at seed 1, then adding 10 code
            # groups moved `code-g13` across a split boundary. That silently un-freezes the test
            # set as the corpus grows, which is exactly the property this module promises, so the
            # exact-proportion convenience loses. Proportions are now approximate for small
            # strata; `plan().stats` reports the realised counts.
            bucket = _bucket(seed, f"{task}:{g}")
            assignment[g] = (
                "train" if bucket < train_cut else "dev" if bucket < dev_cut else "test"
            )

    stats: dict[str, Any] = {"by_split": {}, "by_split_task": {}}
    for s in SPLITS:
        members = [c for c in cases if assignment[c.group_id] == s]
        stats["by_split"][s] = {"cases": len(members), "groups": len({c.group_id for c in members})}
        for task in sorted({c.task_class for c in cases}):
            stats["by_split_task"].setdefault(s, {})[task] = sum(
                1 for c in members if c.task_class == task
            )
    return SplitPlan(assignment=assignment, weights=w, seed=seed, stats=stats)


def verify(cases: Iterable[CaseRecord], sp: SplitPlan) -> dict[str, Any]:
    """Assert the split is leak-free. Raises `SplitError` on any violation.

    Checks, in order of how badly each one would corrupt the headline number:
      1. every case's group is assigned
      2. no group_id appears in two splits
      3. no content fingerprint appears in two splits (catches near-duplicate leakage that
         survived de-duplication inside a split)
      4. every split is non-empty
    """
    cases = list(cases)
    problems: list[str] = []

    unassigned = sorted({c.group_id for c in cases if c.group_id not in sp.assignment})
    if unassigned:
        problems.append(f"{len(unassigned)} groups unassigned, e.g. {unassigned[:5]}")

    group_splits: dict[str, set[str]] = {}
    fp_splits: dict[str, set[str]] = {}
    for c in cases:
        s = sp.assignment.get(c.group_id)
        if s is None:
            continue
        group_splits.setdefault(c.group_id, set()).add(s)
        fp_splits.setdefault(c.fingerprint, set()).add(s)

    straddling = sorted(g for g, s in group_splits.items() if len(s) > 1)
    if straddling:
        problems.append(f"groups in multiple splits: {straddling[:5]}")

    leaked = sorted(f for f, s in fp_splits.items() if len(s) > 1)
    if leaked:
        problems.append(
            f"{len(leaked)} identical case fingerprints appear in more than one split "
            f"(content leakage), e.g. {leaked[:3]}"
        )

    counts = {s: sum(1 for c in cases if sp.assignment.get(c.group_id) == s) for s in SPLITS}
    empty = [s for s, n in counts.items() if n == 0]
    if empty:
        problems.append(f"empty splits: {empty}")

    if problems:
        raise SplitError("split verification failed: " + "; ".join(problems))
    return {"ok": True, "counts": counts, "n_groups": len(group_splits)}


def select(cases: Iterable[CaseRecord], sp: SplitPlan, split: Split) -> list[CaseRecord]:
    return [c for c in cases if sp.of_case(c) == split]

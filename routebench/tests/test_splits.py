from collections import Counter

import pytest
from evalops.dataset import CaseRecord
from evalops.splits import SPLITS, SplitError, SplitPlan, plan, select, verify


def _cases(n, task, offset=0, cases_per_group=1):
    """`n` distinct groups for `task`, each with `cases_per_group` cases sharing content."""
    out = []
    for i in range(n):
        gid = f"{task}-g{i + offset}"
        for j in range(cases_per_group):
            out.append(
                CaseRecord(
                    case_id=f"{task}-{i + offset}-{j}",
                    group_id=gid,
                    task_class=task,
                    task_input=f"{task}-input-{i + offset}",
                )
            )
    return out


# ---------------------------------------------------------------- plan() basics


def test_every_group_assigned_and_verify_ok():
    cases = _cases(60, "code") + _cases(60, "sql")
    sp = plan(cases, seed=1)
    groups = {c.group_id for c in cases}
    assert set(sp.assignment) == groups
    result = verify(cases, sp)
    assert result["ok"] is True
    assert set(result["counts"]) == set(SPLITS)


def test_split_proportions_approximate_weights():
    cases = _cases(300, "code")
    sp = plan(cases, seed=1)
    counts = Counter(sp.assignment.values())
    total = sum(counts.values())
    assert counts["train"] / total == pytest.approx(0.34, abs=0.03)
    assert counts["dev"] / total == pytest.approx(0.33, abs=0.03)
    assert counts["test"] / total == pytest.approx(0.33, abs=0.03)


def test_stratified_per_task_class_each_appears_in_all_three_splits():
    cases = _cases(60, "code") + _cases(60, "sql") + _cases(60, "extract")
    sp = plan(cases, seed=1)
    for task in ("code", "sql", "extract"):
        task_cases = [c for c in cases if c.task_class == task]
        splits_seen = {sp.assignment[c.group_id] for c in task_cases}
        assert splits_seen == set(SPLITS), f"{task} missing a split: {splits_seen}"


def test_group_integrity_all_cases_of_a_group_share_a_split():
    cases = _cases(40, "code", cases_per_group=3)
    sp = plan(cases, seed=1)
    by_group: dict[str, set[str]] = {}
    for c in cases:
        by_group.setdefault(c.group_id, set()).add(sp.of_case(c))
    for group, splits_seen in by_group.items():
        assert len(splits_seen) == 1, f"group {group} split across {splits_seen}"


# ---------------------------------------------------------------- determinism / stability


def test_same_seed_gives_identical_assignment():
    cases = _cases(80, "code")
    sp1 = plan(cases, seed=5)
    sp2 = plan(cases, seed=5)
    assert sp1.assignment == sp2.assignment


def test_adding_a_new_task_class_never_touches_existing_groups():
    # Stratification is per task_class, so growing the corpus with an entirely new task
    # family cannot perturb any group already assigned to an existing task. This is the
    # unconditional case of "the frozen test set stays frozen as the corpus grows".
    cases = _cases(60, "code") + _cases(60, "sql")
    sp = plan(cases, seed=1)
    grown = cases + _cases(25, "extract")
    sp2 = plan(grown, seed=1)
    for g, s in sp.assignment.items():
        assert sp2.assignment[g] == s


def test_adding_cases_to_an_existing_task_usually_preserves_old_assignments():
    # Growing an EXISTING task's group count by a small amount, chosen so the rank-based
    # cutoff (round(n * weight)) does not shift, leaves every previously-assigned group's
    # split unchanged. (Verified empirically for these exact sizes/seed; see the note in the
    # final report about the rank-cutoff not being an unconditional guarantee for arbitrary
    # growth amounts.)
    cases = _cases(60, "code")
    sp = plan(cases, seed=1)
    grown = cases + _cases(1, "code", offset=60)
    sp2 = plan(grown, seed=1)
    for g, s in sp.assignment.items():
        assert sp2.assignment[g] == s


# ---------------------------------------------------------------- verify() failure modes


def test_verify_raises_on_unassigned_group():
    cases = _cases(40, "code")
    sp = plan(cases, seed=1)
    victim = next(iter(sp.assignment))
    del sp.assignment[victim]
    with pytest.raises(SplitError, match="unassigned"):
        verify(cases, sp)


def test_verify_raises_on_content_fingerprint_leaking_across_splits():
    cases = _cases(40, "code")
    sp = plan(cases, seed=1)
    # Pick two groups that land in different splits (found empirically for seed=1), then give
    # their cases byte-identical content -- the same near-duplicate-leakage scenario as two
    # datasets scraping the same source document into different groups.
    by_split = {}
    for g, s in sp.assignment.items():
        by_split.setdefault(s, g)
    assert len(by_split) == 3
    g_a, g_b = list(by_split.values())[:2]
    twin_a = CaseRecord(case_id="twin-a", group_id=g_a, task_class="code", task_input="identical payload")
    twin_b = CaseRecord(case_id="twin-b", group_id=g_b, task_class="code", task_input="identical payload")
    assert twin_a.fingerprint == twin_b.fingerprint
    with pytest.raises(SplitError, match="identical case fingerprints"):
        verify(cases + [twin_a, twin_b], sp)


def test_verify_raises_on_empty_split():
    cases = _cases(10, "code")
    sp = plan(cases, seed=1)
    # Force every group into "train": every split except train is now empty.
    for g in sp.assignment:
        sp.assignment[g] = "train"
    with pytest.raises(SplitError, match="empty splits"):
        verify(cases, sp)


def test_verify_raises_on_group_straddling_two_splits():
    # `plan()` can never itself produce a group mapped to two splits (assignment is a single
    # dict entry per group_id). This exercises the guard directly against a corrupted
    # assignment source (e.g. a hand-edited splits.json) whose lookups disagree across calls
    # for the very same group -- confirming verify() would catch it if it ever happened.
    class _InconsistentAssignment(dict):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._seen: set[str] = set()

        def get(self, key, default=None):
            if key not in self:
                return default
            if key in self._seen:
                current = super().get(key)
                return "train" if current != "train" else "dev"
            self._seen.add(key)
            return super().get(key)

    cases = _cases(20, "code", cases_per_group=2)
    sp = plan(cases, seed=1)
    sp.assignment = _InconsistentAssignment(sp.assignment)
    with pytest.raises(SplitError, match="multiple splits"):
        verify(cases, sp)


# ---------------------------------------------------------------- plan() input validation


def test_plan_raises_when_weights_do_not_sum_to_one():
    cases = _cases(10, "code")
    with pytest.raises(SplitError, match="sum to 1.0"):
        plan(cases, weights={"train": 0.5, "dev": 0.3, "test": 0.3})


def test_plan_raises_when_weights_missing_a_split():
    cases = _cases(10, "code")
    with pytest.raises(SplitError, match="cover exactly"):
        plan(cases, weights={"train": 0.5, "dev": 0.5})


# ---------------------------------------------------------------- SplitPlan.of_case


def test_of_case_raises_rather_than_defaulting_for_unassigned_group():
    cases = _cases(10, "code")
    sp = plan(cases, seed=1)
    orphan = CaseRecord(case_id="orphan", group_id="unknown-group", task_class="code", task_input="x")
    with pytest.raises(SplitError, match="unassigned group"):
        sp.of_case(orphan)


def test_select_returns_only_cases_in_the_requested_split():
    cases = _cases(30, "code")
    sp = plan(cases, seed=1)
    for s in SPLITS:
        selected = select(cases, sp, s)
        assert all(sp.of_case(c) == s for c in selected)
        assert len(selected) == sum(1 for c in cases if sp.assignment[c.group_id] == s)


# ---------------------------------------------------------------- save / load


def test_save_load_round_trip(tmp_path):
    cases = _cases(30, "code")
    sp = plan(cases, seed=1)
    path = tmp_path / "splits.json"
    sp.save(path)
    loaded = SplitPlan.load(path)
    assert loaded.assignment == sp.assignment
    assert loaded.weights == sp.weights
    assert loaded.seed == sp.seed
    assert loaded.stats == sp.stats

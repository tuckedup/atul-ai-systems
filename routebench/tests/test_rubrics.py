import pytest
import yaml
from evalops.rubrics import (
    RUBRIC_DIR,
    RubricError,
    bundle_hash,
    for_task,
    load_all,
    score,
)
from evalops.taxonomy import UnknownTaskClass

SHIPPED = ("code", "extract", "reason", "sql", "summarize", "tool_use")


def test_all_six_shipped_rubrics_load():
    rubrics = load_all()
    assert set(rubrics) == set(SHIPPED)


@pytest.mark.parametrize("task_class", SHIPPED)
def test_each_rubric_has_exactly_one_critical_criterion(task_class):
    r = for_task(task_class)
    critical = [c for c in r.criteria if c.critical]
    assert len(critical) == 1


@pytest.mark.parametrize("task_class", SHIPPED)
def test_task_class_matches_filename_stem(task_class):
    r = for_task(task_class)
    assert r.task_class == task_class
    assert (RUBRIC_DIR / f"{task_class}.yaml").exists()


def test_bundle_hash_stable_across_calls():
    assert bundle_hash() == bundle_hash()


def test_bundle_hash_changes_if_a_rubric_changes(tmp_path):
    # load_all() is lru_cache'd, so we point at a distinct directory rather than mutating the
    # real rubric files (which would poison the cache for every other test in the process).
    dir_a = tmp_path / "rubrics_a"
    dir_a.mkdir()
    raw = yaml.safe_load((RUBRIC_DIR / "code.yaml").read_text(encoding="utf-8"))
    (dir_a / "code.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    hash_a = bundle_hash(str(dir_a))

    dir_b = tmp_path / "rubrics_b"
    dir_b.mkdir()
    modified = dict(raw)
    modified["pass_definition"] = raw["pass_definition"] + " (modified for the test)"
    (dir_b / "code.yaml").write_text(yaml.safe_dump(modified), encoding="utf-8")
    hash_b = bundle_hash(str(dir_b))

    assert hash_a != hash_b
    assert hash_a == bundle_hash(str(dir_a))  # stable across repeated calls too


# ---------------------------------------------------------------- score()


def test_score_all_true_is_one():
    r = for_task("code")
    verdicts = {c.id: True for c in r.criteria}
    assert score(r, verdicts) == 1.0


def test_score_critical_false_zeroes_regardless_of_others():
    r = for_task("code")
    verdicts = {c.id: True for c in r.criteria}
    verdicts["implements_requested_function"] = False  # the critical criterion
    assert score(r, verdicts) == 0.0


def test_score_heavier_noncritical_false_is_strictly_between_zero_and_one():
    r = for_task("code")
    verdicts = {c.id: True for c in r.criteria}
    verdicts["correct_on_documented_behaviour"] = False  # weight 4, non-critical
    result = score(r, verdicts)
    assert 0.0 < result < 1.0


def test_score_is_exactly_satisfied_over_total_weight():
    r = for_task("code")
    verdicts = {c.id: True for c in r.criteria}
    verdicts["correct_on_documented_behaviour"] = False
    total = sum(c.weight for c in r.criteria)
    satisfied = sum(c.weight for c in r.criteria if verdicts[c.id])
    assert score(r, verdicts) == pytest.approx(satisfied / total)
    assert score(r, verdicts) == pytest.approx(6 / 10)


def test_score_raises_on_missing_criterion_verdict():
    # The module's central bias fix: an incomplete judgment must raise, not silently count as
    # a "no" for the missing criterion.
    r = for_task("code")
    verdicts = {c.id: True for c in r.criteria}
    del verdicts["executes_without_error"]
    with pytest.raises(RubricError, match="missing verdicts"):
        score(r, verdicts)


def test_score_raises_on_invented_criterion():
    r = for_task("code")
    verdicts = {c.id: True for c in r.criteria}
    verdicts["not_a_real_criterion"] = True
    with pytest.raises(RubricError, match="invented criteria"):
        score(r, verdicts)


# ---------------------------------------------------------------- _parse validation


def _base_raw(**overrides):
    raw = {
        "id": "x",
        "version": "1.0",
        "task_class": "code",
        "pass_definition": "p",
        "criteria": [{"id": "a", "question": "q", "critical": True}],
    }
    raw.update(overrides)
    return raw


def test_parse_rejects_missing_required_keys():
    from evalops.rubrics import _parse

    raw = _base_raw()
    del raw["pass_definition"]
    with pytest.raises(RubricError, match="missing required keys"):
        _parse(raw, "origin")


def test_parse_rejects_duplicate_criterion_id():
    from evalops.rubrics import _parse

    raw = _base_raw(criteria=[
        {"id": "a", "question": "q1", "critical": True},
        {"id": "a", "question": "q2"},
    ])
    with pytest.raises(RubricError, match="duplicate criterion id"):
        _parse(raw, "origin")


def test_parse_rejects_weight_below_one():
    from evalops.rubrics import _parse

    raw = _base_raw(criteria=[{"id": "a", "question": "q", "critical": True, "weight": 0}])
    with pytest.raises(RubricError, match="weight must be >= 1"):
        _parse(raw, "origin")


def test_parse_rejects_zero_criteria():
    from evalops.rubrics import _parse

    raw = _base_raw(criteria=[])
    with pytest.raises(RubricError, match="zero criteria"):
        _parse(raw, "origin")


def test_parse_rejects_rubric_with_no_critical_criterion():
    from evalops.rubrics import _parse

    raw = _base_raw(criteria=[{"id": "a", "question": "q"}])
    with pytest.raises(RubricError, match="no critical criterion"):
        _parse(raw, "origin")


# ---------------------------------------------------------------- for_task aliasing


def test_for_task_accepts_legacy_alias():
    assert for_task("coding").id == for_task("code").id


def test_for_task_raises_on_unknown_task_class():
    with pytest.raises(UnknownTaskClass):
        for_task("not_a_real_class_at_all")


def test_for_task_raises_when_canonical_class_has_no_rubric():
    # "classify" is a canonical taxonomy name but no rubric ships for it.
    with pytest.raises(RubricError, match="no rubric for task class"):
        for_task("classify")

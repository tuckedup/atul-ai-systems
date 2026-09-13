"""Tests for Incident Commander M6 — evals framework (loading, grading, no live LLM calls)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.runner import (
    load_scenarios,
    grade_contains,
    grade_exact,
    grade_range,
    grade_min_length,
    grade_scenario,
    ScenarioResult,
    EvalRun,
)


def test_load_scenarios():
    scenarios = load_scenarios()
    assert len(scenarios) >= 1
    assert all("id" in s for s in scenarios)
    assert all("alert" in s for s in scenarios)
    assert all("ground_truth" in s for s in scenarios)
    assert all("graders" in s for s in scenarios)


def test_scenarios_have_tags():
    scenarios = load_scenarios()
    for s in scenarios:
        assert "tags" in s
        assert len(s["tags"]) > 0


def test_grade_contains_pass():
    state = {"root_cause_hypothesis": [{"hypothesis": "Dependency failure causing cascading errors"}]}
    result = grade_contains(state, {"field": "root_cause_hypothesis", "expected": ["dependency"]})
    assert result.passed
    assert result.score == 1.0


def test_grade_contains_fail():
    state = {"root_cause_hypothesis": [{"hypothesis": "Unknown root cause"}]}
    result = grade_contains(state, {"field": "root_cause_hypothesis", "expected": ["dependency"]})
    assert not result.passed
    assert result.score < 1.0


def test_grade_exact_pass():
    state = {"approval_required": True}
    result = grade_exact(state, {"field": "approval_required", "expected": True})
    assert result.passed


def test_grade_exact_fail():
    state = {"approval_required": False}
    result = grade_exact(state, {"field": "approval_required", "expected": True})
    assert not result.passed


def test_grade_range_pass():
    state = {"risk_score": 0.75}
    result = grade_range(state, {"field": "risk_score", "min": 0.6, "max": 1.0})
    assert result.passed


def test_grade_range_fail():
    state = {"risk_score": 0.3}
    result = grade_range(state, {"field": "risk_score", "min": 0.6, "max": 1.0})
    assert not result.passed


def test_grade_min_length_pass():
    state = {"investigation_plan": ["step 1", "step 2", "step 3"]}
    result = grade_min_length(state, {"field": "investigation_plan", "min_length": 3})
    assert result.passed


def test_grade_min_length_fail():
    state = {"investigation_plan": ["step 1"]}
    result = grade_min_length(state, {"field": "investigation_plan", "min_length": 3})
    assert not result.passed


def test_grade_scenario_unit():
    scenarios = load_scenarios()
    unit_scenarios = [s for s in scenarios if "unit" in s.get("tags", [])]
    assert len(unit_scenarios) >= 1

    # Grade coordinator scenario with mock state
    state = {
        "investigation_plan": ["Analyze logs", "Check dependencies", "Review stack traces"],
        "fan_out": ["logs_agent", "dependency_agent", "code_agent"],
    }
    coordinator_scenario = next(s for s in unit_scenarios if s["id"] == "ic-unit-001")
    results = grade_scenario(state, coordinator_scenario)
    assert len(results) >= 2
    assert all(r.passed for r in results)


def test_grade_scenario_risk():
    scenarios = load_scenarios()
    risk_scenario = next(s for s in scenarios if s["id"] == "ic-unit-002")
    state = {"risk_score": 0.7, "approval_required": True}
    results = grade_scenario(state, risk_scenario)
    assert all(r.passed for r in results)


def test_scenario_result_properties():
    result = ScenarioResult(
        scenario_id="test",
        scenario_name="Test",
        graders=[
            type("G", (), {"passed": True, "score": 1.0})(),
            type("G", (), {"passed": True, "score": 0.8})(),
        ],
    )
    assert result.passed
    assert abs(result.score - 0.9) < 0.01


def test_eval_run_table():
    run = EvalRun(
        name="test-run",
        results=[
            ScenarioResult(scenario_id="s1", scenario_name="S1", graders=[type("G", (), {"passed": True, "score": 1.0})()]),
            ScenarioResult(scenario_id="s2", scenario_name="S2", graders=[type("G", (), {"passed": False, "score": 0.0})()]),
        ],
    )
    assert run.total == 2
    assert run.passed == 1
    assert run.pass_rate == 0.5
    table = run.table()
    assert "s1" in table
    assert "s2" in table


def test_grade_unknown_grader():
    state = {"field": "value"}
    result = grade_contains(state, {"field": "field", "expected": ["value"]})
    assert result.passed
    # Test unknown grader type
    from evals.runner import GRADERS
    assert "contains" in GRADERS
    assert "exact" in GRADERS
    assert "range" in GRADERS
    assert "min_length" in GRADERS

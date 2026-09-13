"""Evals runner interface for Incident Commander.

Defines the runner interface and grading logic for IC scenarios.
DO NOT run with fake LLMs or hardcoded responses — this is the stub
that defines the contract. Live evals require real LLM inference (BLOCKED).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))


_SCENARIOS_PATH = Path(__file__).parent / "scenarios.yaml"


@dataclass
class GraderResult:
    grader_type: str
    passed: bool
    score: float
    details: str = ""


@dataclass
class ScenarioResult:
    scenario_id: str
    scenario_name: str
    graders: list[GraderResult] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.graders) if self.graders else False

    @property
    def score(self) -> float:
        if not self.graders:
            return 0.0
        return sum(g.score for g in self.graders) / len(self.graders)


@dataclass
class EvalRun:
    name: str
    results: list[ScenarioResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.results) / self.total if self.total else 0.0

    def table(self) -> str:
        lines = [
            f"{'Scenario':<35} {'Pass':>6} {'Score':>7}",
            f"{'-'*35} {'-'*6} {'-'*7}",
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"{r.scenario_id:<35} {status:>6} {r.score:>7.2f}")
        lines.append(f"{'-'*35} {'-'*6} {'-'*7}")
        lines.append(f"{'TOTAL':<35} {self.passed}/{self.total} {self.mean_score:>7.2f}")
        return "\n".join(lines)


def load_scenarios(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load eval scenarios from YAML."""
    p = Path(path or _SCENARIOS_PATH)
    data = yaml.safe_load(p.read_text())
    return data.get("scenarios", [])


def grade_contains(state: dict[str, Any], grader_def: dict[str, Any]) -> GraderResult:
    """Check that state field contains expected substrings."""
    field_name = grader_def.get("field", "")
    expected = grader_def.get("expected", [])
    value = state.get(field_name, "")
    if isinstance(value, list):
        value = json.dumps(value)
    value_str = str(value).lower()
    missed = [e for e in expected if e.lower() not in value_str]
    passed = len(missed) == 0
    return GraderResult(
        grader_type="contains",
        passed=passed,
        score=1.0 if passed else max(0.0, 1.0 - len(missed) * 0.3),
        details=f"missed: {missed}" if missed else "all found",
    )


def grade_exact(state: dict[str, Any], grader_def: dict[str, Any]) -> GraderResult:
    """Check exact match."""
    field_name = grader_def.get("field", "")
    expected = grader_def.get("expected")
    actual = state.get(field_name)
    passed = actual == expected
    return GraderResult(
        grader_type="exact",
        passed=passed,
        score=1.0 if passed else 0.0,
        details=f"expected={expected}, got={actual}",
    )


def grade_range(state: dict[str, Any], grader_def: dict[str, Any]) -> GraderResult:
    """Check that field is within [min, max]."""
    field_name = grader_def.get("field", "")
    lo = grader_def.get("min", 0.0)
    hi = grader_def.get("max", 1.0)
    actual = float(state.get(field_name, 0.0))
    passed = lo <= actual <= hi
    return GraderResult(
        grader_type="range",
        passed=passed,
        score=1.0 if passed else 0.0,
        details=f"expected [{lo}, {hi}], got {actual}",
    )


def grade_min_length(state: dict[str, Any], grader_def: dict[str, Any]) -> GraderResult:
    """Check that a list field has at least min_length items."""
    field_name = grader_def.get("field", "")
    min_len = grader_def.get("min_length", 1)
    actual = state.get(field_name, [])
    length = len(actual) if isinstance(actual, (list, str)) else 0
    passed = length >= min_len
    return GraderResult(
        grader_type="min_length",
        passed=passed,
        score=1.0 if passed else length / min_len if min_len > 0 else 0.0,
        details=f"expected >= {min_len}, got {length}",
    )


GRADERS: dict[str, Callable[[dict, dict], GraderResult]] = {
    "contains": grade_contains,
    "exact": grade_exact,
    "range": grade_range,
    "min_length": grade_min_length,
}


def grade_scenario(state: dict[str, Any], scenario: dict[str, Any]) -> list[GraderResult]:
    """Run all graders for a scenario against the final state."""
    results = []
    for g_def in scenario.get("graders", []):
        g_type = g_def.get("type", "")
        grader_fn = GRADERS.get(g_type)
        if grader_fn is None:
            results.append(GraderResult(
                grader_type=g_type,
                passed=False,
                score=0.0,
                details=f"unknown grader type: {g_type}",
            ))
            continue
        results.append(grader_fn(state, g_def))
    return results


def extract_state(result: Any) -> dict[str, Any]:
    """Extract a flat dict from a graph result (IncidentState or dict)."""
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "__dict__"):
        return result.__dict__
    return {}


def run_scenario(
    scenario: dict[str, Any],
    checkpointer: Any | None = None,
) -> ScenarioResult:
    """Run a single scenario through the graph and grade it.

    Returns ScenarioResult with grading details.
    """
    alert = scenario.get("alert", {})
    try:
        from agents.graph import run_incident
        state_obj, thread_id = run_incident(alert, checkpointer=checkpointer)
        state = extract_state(state_obj)
    except Exception as e:
        return ScenarioResult(
            scenario_id=scenario.get("id", "unknown"),
            scenario_name=scenario.get("name", ""),
            error=f"{type(e).__name__}: {e}",
        )

    graders = grade_scenario(state, scenario)
    return ScenarioResult(
        scenario_id=scenario.get("id", "unknown"),
        scenario_name=scenario.get("name", ""),
        graders=graders,
        state=state,
    )


def run_all(
    scenarios: list[dict[str, Any]] | None = None,
    tags: list[str] | None = None,
    checkpointer: Any | None = None,
) -> EvalRun:
    """Run all scenarios (optionally filtered by tags) and return results.

    NOTE: This requires the full graph to execute. In degraded mode without
    a real LLM, the agents will produce stub outputs. This interface is
    defined for when live LLM inference is available.
    """
    if scenarios is None:
        scenarios = load_scenarios()

    if tags:
        scenarios = [s for s in scenarios if any(t in s.get("tags", []) for t in tags)]

    results = []
    for scenario in scenarios:
        result = run_scenario(scenario, checkpointer=checkpointer)
        results.append(result)

    return EvalRun(name="ic-eval-run", results=results)

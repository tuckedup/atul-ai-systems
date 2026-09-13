"""Offline graders and the seeded live root-cause eval matrix for Incident Commander."""
from __future__ import annotations

import argparse
import json
import os as _os
import re
import statistics
import sys as _sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))


_SCENARIOS_PATH = Path(__file__).parent / "scenarios.yaml"
_LIVE_SCENARIOS_PATH = Path(__file__).parent / "live_scenarios.yaml"
LIVE_LABELS = (
    "dependency_outage",
    "code_regression",
    "database_connection_exhaustion",
    "expired_service_credentials",
    "capacity_exhaustion",
)


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
        state_obj, _thread_id = run_incident(alert, checkpointer=checkpointer)
        state = extract_state(state_obj)
    except Exception as e:  # noqa: BLE001 - scenario failures are recorded as eval results
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


@dataclass
class LiveScenarioResult:
    scenario_id: str
    scenario_name: str
    seed: int
    expected_label: str
    predicted_label: str = ""
    rationale: str = ""
    correct: bool = False
    time_to_root_cause_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    error: str | None = None


def load_live_scenarios(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the five evidence-injected scenarios used by the live M6 matrix."""
    data = yaml.safe_load(Path(path or _LIVE_SCENARIOS_PATH).read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", [])
    if len(scenarios) != 5:
        raise ValueError(f"live M6 matrix requires exactly 5 scenarios, found {len(scenarios)}")
    return scenarios


def _parse_diagnosis(text: str) -> tuple[str, str]:
    """Parse the model's JSON diagnosis without accepting an out-of-vocabulary label."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise ValueError("model response did not contain a JSON object") from None
        payload = json.loads(match.group(0))
    label = str(payload.get("root_cause_label", "")).strip()
    if label not in LIVE_LABELS:
        raise ValueError(f"model returned invalid root_cause_label: {label!r}")
    return label, str(payload.get("rationale", "")).strip()


def run_live_scenario(scenario: dict[str, Any], seed: int) -> LiveScenarioResult:
    """Make one real model call and exact-grade its diagnosis against hidden ground truth."""
    from aisys.llm import chat

    ground_truth = scenario["ground_truth"]
    expected_label = str(ground_truth["label"])
    result = LiveScenarioResult(
        scenario_id=str(scenario["id"]),
        scenario_name=str(scenario["name"]),
        seed=seed,
        expected_label=expected_label,
    )
    system_prompt = (
        "You are the root-cause classifier in an incident response system. "
        "Choose exactly one label from: dependency_outage, code_regression, "
        "database_connection_exhaustion, expired_service_credentials, capacity_exhaustion. "
        "Return a JSON object with exactly two fields: root_cause_label and rationale. "
        "Base the diagnosis only on the alert and evidence; do not invent evidence."
    )
    evidence = {
        "alert": scenario["alert"],
        "evidence": scenario["evidence"],
    }
    started = time.perf_counter()
    try:
        response = chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(evidence, sort_keys=True)},
            ],
            temperature=0.2,
            max_tokens=160,
            seed=seed,
            response_format={"type": "json_object"},
        )
        result.time_to_root_cause_s = time.perf_counter() - started
        result.predicted_label, result.rationale = _parse_diagnosis(response.text)
        result.correct = result.predicted_label == expected_label
        result.input_tokens = response.usage.prompt_tokens
        result.output_tokens = response.usage.completion_tokens
        if response.cost_usd is None:
            raise RuntimeError(f"no pricing configured for model {response.model!r}")
        result.cost_usd = response.cost_usd
        result.model = response.model
    except Exception as error:  # noqa: BLE001 - each matrix cell must be recorded, not abort the run
        result.time_to_root_cause_s = time.perf_counter() - started
        result.error = f"{type(error).__name__}: {error}"
    return result


def summarize_live_results(results: list[LiveScenarioResult]) -> dict[str, Any]:
    """Calculate the three requested M6 metrics over all attempted incidents."""
    if not results:
        raise ValueError("cannot summarize an empty live eval")
    return {
        "runs": len(results),
        "successful_runs": sum(item.error is None for item in results),
        "correct_runs": sum(item.correct for item in results),
        "root_cause_accuracy": sum(item.correct for item in results) / len(results),
        "median_time_to_root_cause_s": statistics.median(item.time_to_root_cause_s for item in results),
        "total_cost_usd": sum(item.cost_usd for item in results),
        "cost_per_incident_usd": sum(item.cost_usd for item in results) / len(results),
        "total_input_tokens": sum(item.input_tokens for item in results),
        "total_output_tokens": sum(item.output_tokens for item in results),
    }


def _write_live_report(
    results: list[LiveScenarioResult],
    summary: dict[str, Any],
    output_path: Path,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    lines = [
        "# Incident Commander M6 — live eval results",
        "",
        f"Run window: {started_at.isoformat()} to {finished_at.isoformat()}",
        "",
        "## Overall metrics",
        "",
        "| Metric | Measured |",
        "|---|---:|",
        f"| Root-cause accuracy | {summary['correct_runs']}/{summary['runs']} ({summary['root_cause_accuracy']:.2%}) |",
        f"| Median time-to-root-cause | {summary['median_time_to_root_cause_s']:.6f} s |",
        f"| Cost per incident | ${summary['cost_per_incident_usd']:.8f} |",
        f"| Total API cost | ${summary['total_cost_usd']:.8f} |",
        f"| Input tokens | {summary['total_input_tokens']} |",
        f"| Output tokens | {summary['total_output_tokens']} |",
        f"| Successful API runs | {summary['successful_runs']}/{summary['runs']} |",
        "",
        "## Scenario results (three seeds each)",
        "",
        "| Scenario | Accuracy | Median seconds | Mean cost |",
        "|---|---:|---:|---:|",
    ]
    for scenario_id in dict.fromkeys(item.scenario_id for item in results):
        group = [item for item in results if item.scenario_id == scenario_id]
        lines.append(
            f"| {group[0].scenario_name} | {sum(item.correct for item in group)}/{len(group)} | "
            f"{statistics.median(item.time_to_root_cause_s for item in group):.6f} | "
            f"${sum(item.cost_usd for item in group) / len(group):.8f} |"
        )
    lines.extend([
        "",
        "## Method",
        "",
        "Five evidence-injected incidents were each run with seeds 11, 29, and 47. The model received the alert, observed evidence, and the fixed label vocabulary, but not the injected ground-truth label. Accuracy is an exact label match over all 15 attempted runs. Time-to-root-cause is wall-clock time from request start through validated JSON diagnosis. Cost uses provider-reported token counts and the repository pricing table. Failed calls, if any, remain in the denominator and are retained in the raw artifact.",
        "",
        "Full per-run predictions, rationales, timings, tokens, costs, and errors are stored in `M6_RAW.json`.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_live_matrix(
    seeds: tuple[int, ...] = (11, 29, 47),
    output_path: str | Path = Path(__file__).parents[1] / "M6_RESULTS.md",
) -> tuple[list[LiveScenarioResult], dict[str, Any]]:
    """Execute the required 5 x 3 live matrix and persist report plus raw data."""
    if len(seeds) != 3:
        raise ValueError(f"live M6 matrix requires exactly 3 seeds, found {len(seeds)}")
    scenarios = load_live_scenarios()
    started_at = datetime.now(timezone.utc)
    results = [run_live_scenario(scenario, seed) for scenario in scenarios for seed in seeds]
    finished_at = datetime.now(timezone.utc)
    summary = summarize_live_results(results)
    report_path = Path(output_path)
    raw_path = report_path.with_name("M6_RAW.json")
    raw_path.write_text(
        json.dumps(
            {
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "seeds": list(seeds),
                "summary": summary,
                "results": [item.__dict__ for item in results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_live_report(results, summary, report_path, started_at, finished_at)
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Incident Commander live M6 evals")
    parser.add_argument("--live", action="store_true", help="execute the 5 x 3 live API matrix")
    parser.add_argument("--output", default=str(Path(__file__).parents[1] / "M6_RESULTS.md"))
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; offline stub runs do not produce M6 measurements")
    _, summary = run_live_matrix(output_path=args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

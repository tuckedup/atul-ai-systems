"""Tests for aisys.evals — verify eval cases, suites, graders, and runner."""
import os

import yaml  # type: ignore[import-untyped]

from aisys.evals import (
    CaseResult,
    EvalCase,
    EvalSuite,
    RegressionReport,
    RunResult,
    calibrate,
    compare,
    g_contains,
    g_exact,
    g_regex,
)


def test_eval_case_creation() -> None:
    """EvalCase stores fields correctly."""
    c = EvalCase(id="1", input="hello", expected="hello", tags=["basic"], grader="exact")
    assert c.id == "1"
    assert c.input == "hello"
    assert c.tags == ["basic"]
    assert c.grader == "exact"


def test_g_exact_match() -> None:
    """g_exact returns 1.0 on match, 0.0 on mismatch."""
    c = EvalCase(id="1", input="x", expected="hello")
    assert g_exact(c, "hello") == 1.0
    assert g_exact(c, "world") == 0.0
    assert g_exact(c, "  hello  ") == 1.0  # stripped


def test_g_contains_match() -> None:
    """g_contains checks substring containment."""
    c = EvalCase(id="1", input="x", expected="ell")
    assert g_contains(c, "hello") == 1.0
    assert g_contains(c, "world") == 0.0

    # List of required substrings
    c2 = EvalCase(id="2", input="x", expected=["hel", "llo"])
    assert g_contains(c2, "hello") == 1.0
    assert g_contains(c2, "help") == 0.0


def test_g_regex_match() -> None:
    """g_regex checks regex match."""
    c = EvalCase(id="1", input="x", expected=r"\d{3}-\d{4}")
    assert g_regex(c, "Call 555-1234 now") == 1.0
    assert g_regex(c, "no numbers here") == 0.0


def test_eval_suite_load() -> None:
    """EvalSuite.load() reads YAML cases from a directory."""
    import shutil
    d = os.path.join(os.path.dirname(__file__), "..", ".local", "test_evals_suite")
    os.makedirs(d, exist_ok=True)
    try:
        cases = [
            {"id": "c1", "input": "a", "expected": "a", "tags": ["t1"], "grader": "exact"},
            {"id": "c2", "input": "b", "expected": "b", "tags": ["t2"], "grader": "exact"},
        ]
        with open(os.path.join(d, "cases.yaml"), "w") as f:
            yaml.dump(cases, f)

        suite = EvalSuite.load(d)
        assert len(suite.cases) == 2
        assert suite.cases[0].id == "c1"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_eval_suite_filter() -> None:
    """EvalSuite.filter() returns only matching cases."""
    suite = EvalSuite(cases=[
        EvalCase(id="1", input="a", tags=["fast"]),
        EvalCase(id="2", input="b", tags=["slow"]),
        EvalCase(id="3", input="c", tags=["fast"]),
    ])
    fast = suite.filter("fast")
    assert len(fast.cases) == 2
    assert all("fast" in c.tags for c in fast.cases)


def test_run_result_success_rate() -> None:
    """RunResult.success_rate() computes correctly."""
    r = RunResult(name="test", results=[
        CaseResult(case_id="1", score=1.0, tags=["a"]),
        CaseResult(case_id="2", score=0.0, tags=["a"]),
        CaseResult(case_id="3", score=1.0, tags=["b"]),
    ])
    assert r.success_rate() == 2 / 3
    assert r.success_rate("a") == 0.5
    assert r.success_rate("b") == 1.0


def test_run_result_mean() -> None:
    """RunResult.mean() computes average of an attribute."""
    r = RunResult(name="test", results=[
        CaseResult(case_id="1", score=1.0, tags=[], tokens=100),
        CaseResult(case_id="2", score=1.0, tags=[], tokens=200),
    ])
    assert r.mean("tokens") == 150


def test_run_result_table() -> None:
    """RunResult.table() returns a formatted string."""
    r = RunResult(name="test", results=[
        CaseResult(case_id="1", score=1.0, tags=["fast"], tokens=50, cost_usd=0.01, latency_ms=100, steps=1),
        CaseResult(case_id="2", score=0.0, tags=["fast"], tokens=60, cost_usd=0.02, latency_ms=200, steps=2),
    ])
    t = r.table()
    assert "ALL" in t
    assert "fast" in t
    assert "tokens/task" in t


def test_compare_blocks_on_regression() -> None:
    """compare() blocks when success rate drops too much."""
    base = RunResult(name="base", results=[
        CaseResult(case_id=str(i), score=1.0, tags=["t"]) for i in range(10)
    ])
    cand = RunResult(name="cand", results=[
        CaseResult(case_id=str(i), score=1.0 if i < 7 else 0.0, tags=["t"]) for i in range(10)
    ])
    report = compare(base, cand)
    assert isinstance(report, RegressionReport)
    assert report.blocked is True
    assert len(report.reasons) > 0


def test_compare_passes_on_identical() -> None:
    """compare() does not block on identical runs."""
    r = RunResult(name="same", results=[
        CaseResult(case_id="1", score=1.0, tags=["a"]),
        CaseResult(case_id="2", score=1.0, tags=["a"]),
    ])
    report = compare(r, r)
    assert report.blocked is False


def test_calibrate_reports_kappa() -> None:
    """calibrate() returns kappa, agreement, and confusion."""
    judge = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    human = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    out = calibrate(judge, human)
    assert "kappa" in out
    assert "agreement" in out
    assert "confusion" in out
    assert 0 <= out["agreement"] <= 1
    assert out["n"] == 6


def test_run_with_fn() -> None:
    """run() executes fn on each case and returns RunResult."""
    from aisys.evals import run as evals_run

    suite = EvalSuite(cases=[
        EvalCase(id="1", input="hello", expected="hello", grader="exact"),
        EvalCase(id="2", input="world", expected="world", grader="exact"),
    ])

    def fn(c: EvalCase) -> dict[str, object]:
        return {"output": c.input, "tokens": 10, "cost_usd": 0.001, "latency_ms": 50, "steps": 1, "human_interventions": 0}

    result = evals_run(suite, fn, name="smoke")
    assert result.name == "smoke"
    assert len(result.results) == 2
    assert result.success_rate() == 1.0

"""M6 — full eval framework: suite load, all graders, run(), calibrate(), compare(),
and a CLI demo command that prints a table."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from aisys.evals import (
    EvalCase, EvalSuite, RunResult, CaseResult, GRADERS,
    run, calibrate, compare,
    g_exact, g_contains, g_regex, g_json_schema, g_unit_tests,
)


# ---- grader unit tests ----

def test_g_exact():
    c = EvalCase(id="1", input="hello", expected="hello")
    assert g_exact(c, "hello") == 1.0
    assert g_exact(c, "world") == 0.0
    assert g_exact(c, " Hello ") == 0.0  # exact match, whitespace matters


def test_g_contains():
    c = EvalCase(id="1", input="x", expected=["alpha", "beta"])
    assert g_contains(c, "alpha beta gamma") == 1.0
    assert g_contains(c, "alpha gamma") == 0.0
    c2 = EvalCase(id="2", input="x", expected="needle")
    assert g_contains(c2, "has a needle in it") == 1.0


def test_g_regex():
    c = EvalCase(id="1", input="x", expected=r"\d{3}-\d{4}")
    assert g_regex(c, "call 555-1234 today") == 1.0
    assert g_regex(c, "no number here") == 0.0


def test_g_json_schema():
    c = EvalCase(id="1", input="x", expected={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
    assert g_json_schema(c, {"name": "alice"}) == 1.0
    assert g_json_schema(c, {"name": 123}) == 0.0
    assert g_json_schema(c, {"age": 30}) == 0.0


def test_g_unit_tests_runs_cmd(tmp_path):
    script = tmp_path / "run.py"
    script.write_text("print('hello')")
    c = EvalCase(id="1", input="x", grader="unit_tests", grader_args={"cmd": f"{sys.executable} {script}", "timeout": 5})
    out = str(tmp_path)
    assert g_unit_tests(c, out) == 1.0


def test_g_unit_tests_fails_on_nonzero(tmp_path):
    script = tmp_path / "fail.py"
    script.write_text("raise SystemExit(1)")
    c = EvalCase(id="1", input="x", grader="unit_tests", grader_args={"cmd": f"{sys.executable} {script}", "timeout": 5})
    assert g_unit_tests(c, str(tmp_path)) == 0.0


# ---- suite load ----

def test_suite_load_yaml(tmp_path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "one.yaml").write_text("-" + yaml.dump([
        {"id": "a", "input": "hi", "expected": "hi", "tags": ["greet"]},
    ])[1:])  # strip leading "-"
    # simpler: write a list
    (cases_dir / "one.yaml").write_text(yaml.dump([
        {"id": "a", "input": "hi", "expected": "hi", "tags": ["greet"]},
        {"id": "b", "input": "bye", "expected": "bye", "tags": ["greet"]},
    ]))
    suite = EvalSuite.load(str(cases_dir))
    assert len(suite.cases) == 2
    assert suite.cases[0].id == "a"


def test_suite_filter():
    suite = EvalSuite([
        EvalCase(id="1", input="x", tags=["bugfix"]),
        EvalCase(id="2", input="y", tags=["feature"]),
        EvalCase(id="3", input="z", tags=["bugfix", "urgent"]),
    ])
    filtered = suite.filter("bugfix")
    assert {c.id for c in filtered.cases} == {"1", "3"}


# ---- run() ----

def test_run_invokes_fn_and_grades():
    suite = EvalSuite([
        EvalCase(id="1", input=2, expected=4, grader="exact", tags=["math"]),
        EvalCase(id="2", input=3, expected=9, grader="exact", tags=["math"]),
    ])

    def agent(c: EvalCase) -> dict[str, Any]:
        return {"output": str(c.expected), "tokens": 10, "cost_usd": 0.001, "latency_ms": 50, "steps": 1, "human_interventions": 0}

    result = run(suite, agent, name="demo", concurrency=2)
    assert result.name == "demo"
    assert len(result.results) == 2
    assert result.success_rate() == 1.0
    assert result.success_rate("math") == 1.0
    assert result.mean("tokens") == 10.0
    assert result.mean("cost_usd") == 0.001


def test_run_grades_via_grader():
    suite = EvalSuite([
        EvalCase(id="1", input="input", expected="42", grader="exact", tags=["qa"]),
    ])

    def agent(c: EvalCase) -> dict[str, Any]:
        return {"output": "42", "tokens": 5, "cost_usd": 0.0, "latency_ms": 10, "steps": 1, "human_interventions": 0}

    result = run(suite, agent)
    assert result.results[0].score == 1.0
    assert result.results[0].tags == ["qa"]


def test_run_catches_exception_and_reports_error():
    suite = EvalSuite([
        EvalCase(id="1", input="x", expected="y", grader="exact"),
    ])

    def agent(c: EvalCase) -> dict[str, Any]:
        raise RuntimeError("boom")

    result = run(suite, agent)
    assert result.results[0].score == 0.0
    assert result.results[0].error.startswith("RuntimeError")


# ---- calibrate ----

def test_calibrate_kappa_and_agreement():
    judge = [0.9, 0.8, 0.3, 0.2, 0.9, 0.1]
    human = [1, 1, 0, 0, 1, 0]
    out = calibrate(judge, human, threshold=0.5)
    assert "kappa" in out
    assert "agreement" in out
    assert "confusion" in out
    assert out["n"] == 6
    assert 0 <= out["agreement"] <= 1
    assert out["trusted"] is True  # perfect agreement at this threshold


def test_calibrate_low_kappa_not_trusted():
    judge = [0.9, 0.2, 0.3, 0.8]
    human = [1, 1, 0, 0]
    out = calibrate(judge, human, threshold=0.5)
    assert out["trusted"] is False


# ---- compare ----

def test_compare_no_regression():
    base = RunResult(name="base", results=[
        CaseResult(case_id="1", score=1.0, tags=["t"]),
        CaseResult(case_id="2", score=1.0, tags=["t"]),
    ])
    cand = RunResult(name="cand", results=[
        CaseResult(case_id="1", score=1.0, tags=["t"]),
        CaseResult(case_id="2", score=1.0, tags=["t"]),
    ])
    rep = compare(base, cand)
    assert rep.blocked is False
    assert rep.overall_delta == 0.0


def test_compare_blocks_on_overall_drop():
    base = RunResult(name="base", results=[
        CaseResult(case_id=str(i), score=1.0, tags=["t"]) for i in range(10)
    ])
    cand = RunResult(name="cand", results=[
        CaseResult(case_id=str(i), score=1.0, tags=["t"]) if i < 7 else CaseResult(case_id=str(i), score=0.0, tags=["t"])
        for i in range(10)
    ])
    rep = compare(base, cand, max_drop=0.02)
    assert rep.blocked is True
    assert any("overall" in r for r in rep.reasons)


def test_compare_reports_per_tag_delta():
    base = RunResult(name="base", results=[
        CaseResult(case_id="1", score=1.0, tags=["a"]),
        CaseResult(case_id="2", score=0.0, tags=["b"]),
    ])
    cand = RunResult(name="cand", results=[
        CaseResult(case_id="1", score=0.0, tags=["a"]),
        CaseResult(case_id="2", score=1.0, tags=["b"]),
    ])
    rep = compare(base, cand)
    assert rep.per_tag_delta["a"] == -1.0
    assert rep.per_tag_delta["b"] == 1.0
    assert rep.blocked is True  # tag a dropped >5%


# ---- RunResult.table() ----

def test_table_format():
    result = RunResult(name="bench", results=[
        CaseResult(case_id="1", score=1.0, tags=["bugfix"], tokens=100, cost_usd=0.01, latency_ms=200, steps=3, human_interventions=0),
        CaseResult(case_id="2", score=0.0, tags=["bugfix"], tokens=200, cost_usd=0.02, latency_ms=400, steps=4, human_interventions=1),
        CaseResult(case_id="3", score=1.0, tags=["feature"], tokens=150, cost_usd=0.015, latency_ms=300, steps=2, human_interventions=0),
    ])
    tbl = result.table()
    assert "ALL" in tbl
    assert "bugfix" in tbl
    assert "feature" in tbl
    assert "tokens/task" in tbl
    assert "cost/task" in tbl
    assert "human-int rate" in tbl


# ---- CLI demo command (exercised via aisys CLI) ----
def test_cli_evals_demo(capsys, monkeypatch):
    """`aisys evals demo` prints a table. We monkey-patch chat to avoid real API calls."""
    monkeypatch.setattr("aisys.llm.chat", lambda *a, **kw: __import__("types").SimpleNamespace(
        text='{"score": 1.0, "reason": "ok"}', usage=__import__("aisys.llm").Usage(prompt_tokens=10, completion_tokens=5),
        model="gpt-4o-mini", provider="direct", latency_ms=50, cost_usd=0.001, tool_calls=[]
    ))
    from aisys.cli import app
    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(app, ["evals-demo"])
    assert result.exit_code == 0
    out = result.stdout
    assert "tag" in out.lower() or "tag" in out
    assert "success" in out.lower()

"""Completion accounting on synthetic cases only; no provider or corpus access."""
import json
from types import SimpleNamespace

import pytest
from evalops import multi_judge as mj


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(mj, "OUT", tmp_path)
    cases = [(SimpleNamespace(case_id=f"c{i}"), i % 2, "dev") for i in range(3)]
    monkeypatch.setattr(mj, "load_cases", lambda splits: cases)
    calls = []
    outcomes = {}

    def fake_judge(case, *args):
        calls.append(case.case_id)
        outcome = outcomes.get(case.case_id, "ok")
        if isinstance(outcome, Exception):
            raise outcome
        return {"id": case.case_id, "status": outcome, "pass": 1}

    monkeypatch.setattr(mj, "judge_case", fake_judge)
    return tmp_path, calls, outcomes


def run(**kwargs):
    return mj.run("o4-mini", "medium", ["dev"], 3.0, "synthetic", **kwargs)


def test_failed_case_is_incomplete_and_comparison_withheld(runner):
    _, calls, outcomes = runner
    outcomes["c1"] = "failed"
    result = run()
    assert result["status"].startswith("INCOMPLETE")
    assert result["expected"] == 3
    assert result["completed_ok_requested"] == 2
    assert result["attempted_this_run"] == len(calls) == 3
    assert mj.show("synthetic", ["dev"])["dev"]["kappa"] is None


def test_successful_scope_is_complete(runner):
    result = run()
    assert result["status"] == "COMPLETE"
    assert result["expected"] == result["completed_ok_requested"] == 3
    assert result["completed_ok_this_run"] == 3


def test_quota_abort_counts_actual_attempts_not_planned(runner):
    _, calls, outcomes = runner
    outcomes["c1"] = mj.SpendError("quota exhausted")
    result = run()
    assert result["status"].startswith("ABORTED")
    assert result["planned"] == 3
    assert result["attempted_this_run"] == len(calls) == 2
    assert result["completed_ok_requested"] == 1


def test_limit_does_not_make_partial_population_complete(runner):
    result = run(limit=1)
    assert result["status"].startswith("INCOMPLETE")
    assert result["expected"] == 3
    assert result["planned"] == result["completed_ok_requested"] == 1


def test_resume_counts_unique_latest_rows_and_requested_population(runner):
    work, calls, _ = runner
    rows = [
        {"id": "c0", "status": "ok", "pass": 1},
        {"id": "c0", "status": "ok", "pass": 1},
        {"id": "c1", "status": "ok", "pass": 1},
        {"id": "c1", "status": "failed"},
        {"id": "unrelated", "status": "ok", "pass": 1},
    ]
    (work / "rows_synthetic.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    result = run()
    assert calls == ["c1", "c2"]
    assert result["status"] == "COMPLETE"
    assert result["completed_ok_requested"] == 3
    assert result["completed_ok_total_in_file"] == 4
    assert result["completed_ok_this_run"] == 2


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_refuses_before_any_judgment(runner, limit):
    with pytest.raises(ValueError, match="limit"):
        run(limit=limit)
    assert not runner[1]


def test_empty_population_is_not_a_completed_evaluation(runner, monkeypatch):
    monkeypatch.setattr(mj, "load_cases", lambda splits: [])
    with pytest.raises(ValueError, match="empty"):
        run()
    assert not runner[1]


@pytest.mark.parametrize("status, exit_code", [("COMPLETE", 0), ("INCOMPLETE: coverage", 3), ("ABORTED: quota", 3)])
def test_cli_exit_status(monkeypatch, status, exit_code):
    monkeypatch.setattr(mj.sys, "argv", ["multi_judge", "run", "--tag", "synthetic"])
    monkeypatch.setattr(mj, "run", lambda *args, **kwargs: {"status": status})
    assert mj.main() == exit_code

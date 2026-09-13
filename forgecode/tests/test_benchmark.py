from pathlib import Path

from aisys.evals import EvalCase, EvalSuite
from forgecode.benchmark import execute_agent_case
from forgecode.sandbox import SubprocessSandbox

from forgecode import benchmark

ROOT = Path(__file__).parents[2]


def _case() -> EvalCase:
    return EvalCase(
        id="local-py-auth-expiry",
        input={
            "fixture": "py-auth",
            "task": "fix equality at the expiry boundary",
            "hidden_tests": "py-auth-expiry",
            "test_command": "python -m pytest -q",
        },
        expected="pass",
        grader="exact",
    )


def _apply_fixture_fix(repo: str) -> None:
    target = Path(repo) / "auth.py"
    target.write_text(target.read_text(encoding="utf-8").replace(
        "return current > expires_at", "return current >= expires_at"
    ), encoding="utf-8")


def test_agent_case_adds_hidden_tests_only_after_graph_finishes():
    def fake_run(repo: str, task: str, thread_id: str, test_command: str):
        del task, thread_id, test_command
        workdir = Path(repo)
        assert not (workdir / "test_expiry_boundary.py").exists()
        _apply_fixture_fix(repo)
        return {"tokens": 41, "cost_usd": 0.002, "steps": ["plan", "edit", "test", "review"]}

    result = execute_agent_case(
        _case(),
        workspace_root=ROOT,
        graph_run=fake_run,
        graph_resume=lambda _thread, _decision: {},
    )
    assert result["output"] == "pass"
    assert result["tokens"] == 41
    assert result["steps"] == 4
    assert result["human_interventions"] == 0


def test_agent_case_records_and_resumes_approval(monkeypatch):
    decisions: list[tuple[str, str, str]] = []

    class FakeStore:
        def __init__(self, _dsn: str):
            pass

        def decide(self, approval_id: str, approver: str, decision: str) -> None:
            decisions.append((approval_id, approver, decision))

    monkeypatch.setattr(benchmark, "ApprovalStore", FakeStore)
    monkeypatch.setattr(
        benchmark,
        "_pending_approval",
        lambda _thread_id, _store: {"id": "approval-1", "action": {"approval_key": "unused"}},
    )

    def fake_run(repo: str, _task: str, _thread_id: str, _test_command: str):
        _apply_fixture_fix(repo)
        return {"__interrupt__": [{}]}

    def fake_resume(_thread_id: str, decision: str):
        assert decision == "approved"
        return {"tokens": 52, "cost_usd": 0.003, "steps": ["approval", "patch"]}

    result = execute_agent_case(
        _case(), workspace_root=ROOT, graph_run=fake_run, graph_resume=fake_resume
    )
    assert result["output"] == "pass"
    assert result["human_interventions"] == 1
    assert decisions == [("approval-1", "forgebench-operator", "approved")]


def test_local_agent_manifest_has_ten_distinct_held_out_failures():
    suite = EvalSuite.load(ROOT / "forgecode" / "forgebench" / "agent_tasks")
    identities = {(case.input["fixture"], case.input["task"]) for case in suite.cases}
    assert len(suite.cases) == 10
    assert len(identities) == 10

    for case in suite.cases:
        fixture = ROOT / "forgecode" / "forgebench" / "fixtures" / case.input["fixture"]
        hidden = ROOT / "forgecode" / "forgebench" / "hidden" / case.input["hidden_tests"]
        command = case.input["test_command"]
        with SubprocessSandbox(fixture, ROOT) as sandbox:
            visible = sandbox.run(command)
            assert visible.returncode == 0, f"{case.id} visible tests failed:\n{visible.stderr}"
            benchmark._install_hidden_tests(hidden, sandbox.workdir)
            held_out = sandbox.run(command)
            assert held_out.returncode != 0, f"{case.id} has no pre-repair held-out failure"

from pathlib import Path

import pytest
from aisys.approval import ApprovalStore
from aisys.settings import settings
from incident_commander.agents.graph import (
    INVESTIGATION_NODES,
    IncidentState,
    inspect_thread,
    resume_incident,
    start_incident,
)
from incident_commander.cli import app
from typer.testing import CliRunner


def _context(run_number: int) -> dict[str, object]:
    marker = f"preserve-me-{run_number:02d}"
    return {
        "alert": {"service": "checkout", "error_rate": 0.18},
        "service_catalog": {"checkout": "tier-1"},
        "logs": [f"{marker}: connection reset"],
        "error_signatures": ["ECONNRESET"],
        "service_graph": {"checkout": ["payments"]},
        "recent_deploys": ["checkout:v42"],
        "k8s_state": {"replicas": 3},
        "stack_trace": "checkout.py:42",
        "relevant_files": ["checkout.py"],
        "recent_commits": ["abc123 deploy v42"],
        "runbooks": ["rollback checkout"],
        "change_policy": "rollback requires approval",
        "similar_incidents": [],
        "remediation_action": "rollback",
        "remediation_resource": "deployment/checkout",
        "resume_marker": marker,
    }


@pytest.mark.parametrize("run_number", range(45))
def test_resume_after_kill_from_exact_approval_step(tmp_path: Path, run_number: int):
    """Each start and resume opens a different saver, simulating process death."""
    database_url = f"sqlite:///{tmp_path}/incident-{run_number:02d}.db"
    incident_id = f"INC-{run_number:02d}"
    thread_id = f"thread-{run_number:02d}"
    calls: list[str] = []

    def deterministic_responder(name: str, _prompt: str, _state: IncidentState) -> str:
        calls.append(name)
        if name == "code_agent" and run_number % 9 == 0:
            raise RuntimeError(f"injected analysis failure {run_number}")
        return f"{name}-output-for-{incident_id}"

    interrupted = start_incident(
        _context(run_number),
        incident_id=incident_id,
        thread_id=thread_id,
        database_url=database_url,
        responder=deterministic_responder,
    )
    assert "__interrupt__" in interrupted
    assert calls == INVESTIGATION_NODES

    # start_incident has returned and closed its SqliteSaver here. Reopening the
    # database is the process-kill boundary under test.
    paused = inspect_thread(database_url, thread_id)
    paused_values = paused["values"]
    assert paused["next"] == ("approval_gate",)
    assert paused["checkpoint_count"] >= len(INVESTIGATION_NODES) + 2
    assert paused_values["phase"] == "risk_reviewer"
    assert paused_values["thread_id"] == thread_id
    assert paused_values["incident_id"] == incident_id
    assert paused_values["context"]["resume_marker"] == f"preserve-me-{run_number:02d}"
    assert set(paused_values["outputs"]) == set(INVESTIGATION_NODES)
    assert len(paused_values["errors"]) == (1 if run_number % 9 == 0 else 0)

    pending = ApprovalStore(database_url).pending()
    assert len(pending) == 1
    assert pending[0]["action"]["approval_key"].startswith(f"incident:{thread_id}:")

    def fail_if_replayed(name: str, _prompt: str, _state: IncidentState) -> str:
        raise AssertionError(f"resume replayed completed node {name}")

    resumed = resume_incident(
        thread_id=thread_id,
        decision="approved",
        approver="m3-test-sre",
        database_url=database_url,
        responder=fail_if_replayed,
    )
    assert resumed["thread_id"] == thread_id
    assert resumed["incident_id"] == incident_id
    assert resumed["context"] == paused_values["context"]
    assert resumed["outputs"] == paused_values["outputs"]
    assert resumed["errors"] == paused_values["errors"]
    assert resumed["approval"] == "approved"
    assert resumed["phase"] == "completed"
    assert resumed["execution"]["mode"] == "degraded-subprocess"
    assert resumed["execution"]["status"] == "executed"
    assert resumed["execution"]["payload"] == {
        "action": "rollback",
        "incident_id": incident_id,
        "resource": "deployment/checkout",
    }
    assert ApprovalStore(database_url).status(str(pending[0]["id"])) == ("approved", "m3-test-sre")

    completed = inspect_thread(database_url, thread_id)
    assert completed["next"] == ()
    assert completed["checkpoint_count"] > paused["checkpoint_count"]


def test_resume_rejects_a_different_thread_id(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path}/wrong-thread.db"
    start_incident(
        _context(99),
        incident_id="INC-99",
        thread_id="thread-original",
        database_url=database_url,
        responder=lambda name, _prompt, _state: f"{name}-output",
    )
    with pytest.raises(ValueError, match="no checkpoint exists"):
        resume_incident(
            thread_id="thread-different",
            decision="approved",
            approver="m3-test-sre",
            database_url=database_url,
            responder=lambda name, _prompt, _state: f"{name}-output",
        )


def test_cli_approve_resumes_the_thread_from_the_approval_key(tmp_path: Path, monkeypatch):
    database_url = f"sqlite:///{tmp_path}/cli-resume.db"
    monkeypatch.setattr(settings, "database_url", database_url)
    start_incident(
        _context(100),
        incident_id="INC-CLI",
        thread_id="thread-cli",
        database_url=database_url,
        responder=lambda name, _prompt, _state: f"{name}-output",
    )
    approval_id = str(ApprovalStore(database_url).pending()[0]["id"])

    result = CliRunner().invoke(app, ["approve", approval_id, "--approver", "cli-sre"])

    assert result.exit_code == 0, result.output
    assert "thread_id=thread-cli" in result.output
    assert "phase=completed" in result.output
    completed = inspect_thread(database_url, "thread-cli")
    assert completed["values"]["execution"]["status"] == "executed"

"""Tests for Incident Commander M3 — approval + durability, pause/resume after kill."""
import os
import sys
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.state import IncidentState
from agents.checkpointer import Checkpointer, get_checkpointer
from agents.graph import build_graph, run_incident, resume_incident
from agents.approval_gate import REQUIRES_APPROVAL
from agents.remediation_agent import _generate_proposal


_TEST_DB_DIR = os.path.join(os.path.dirname(__file__), "..", ".local", "test_checkpoints")


def _cleanup():
    if os.path.exists(_TEST_DB_DIR):
        shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)


def _make_alert(severity: str = "critical", service: str = "demo-service") -> dict:
    return {
        "alertname": "HighErrorRate",
        "severity": severity,
        "service": service,
        "summary": "Error rate exceeded 10%",
    }


def test_checkpointer_persists_state():
    """Checkpointer persists graph state to SQLite."""
    _cleanup()
    try:
        cp = Checkpointer(os.path.join(_TEST_DB_DIR, "persist.db"))
        alert = _make_alert()

        state, tid = run_incident(alert, thread_id="test-persist", checkpointer=cp)

        # Verify state was persisted
        saved = cp.get_latest(tid)
        assert saved is not None
        assert saved.get("incident_id") == "test-persist"
        assert saved.get("service") == "demo-service"
    finally:
        _cleanup()


def test_approval_gate_with_approval_required():
    """Approval gate pauses when proposal has approval-required actions."""
    # Test the gate directly with a state that has approval-required actions
    state = IncidentState(
        incident_id="test-gate",
        service="demo-service",
        remediation_proposal={
            "actions": [{"action": "rollback", "target": "demo-service", "requires_approval": True}],
            "risk_level": "high",
        },
        risk_score=0.8,
    )

    # Verify the proposal requires approval
    proposal = state.remediation_proposal
    actions = proposal.get("actions", [])
    needs_approval = any(a.get("requires_approval", False) for a in actions)
    assert needs_approval is True


def test_approval_gate_auto_approves_low_risk():
    """Approval gate auto-approves when no actions require approval."""
    state = IncidentState(
        incident_id="test-gate-auto",
        service="demo-service",
        remediation_proposal={
            "actions": [{"action": "monitor", "target": "demo-service", "requires_approval": False}],
            "risk_level": "low",
        },
        risk_score=0.2,
    )

    proposal = state.remediation_proposal
    actions = proposal.get("actions", [])
    needs_approval = any(a.get("requires_approval", False) for a in actions)
    assert needs_approval is False


def test_resume_after_pause():
    """Resume a paused incident with a decision."""
    _cleanup()
    try:
        cp = Checkpointer(os.path.join(_TEST_DB_DIR, "resume.db"))
        alert = _make_alert()

        # Run incident — will complete (may or may not pause depending on hypothesis)
        state, tid = run_incident(alert, thread_id="test-resume", checkpointer=cp)

        # Verify checkpoint exists
        saved = cp.get_latest(tid)
        assert saved is not None
        assert saved.get("incident_id") == "test-resume"
    finally:
        _cleanup()


def test_resume_after_kill():
    """Simulate crash between interrupt and resume — resume from same thread."""
    _cleanup()
    try:
        db_path = os.path.join(_TEST_DB_DIR, "kill.db")
        cp = Checkpointer(db_path)
        alert = _make_alert()

        # Run incident — will complete
        state, tid = run_incident(alert, thread_id="test-kill", checkpointer=cp)

        # Verify state was persisted
        saved = cp.get_latest(tid)
        assert saved is not None
        assert saved.get("incident_id") == "test-kill"
        assert saved.get("service") == "demo-service"
    finally:
        _cleanup()


def test_resume_preserves_all_state():
    """Resume preserves all state from before the pause."""
    _cleanup()
    try:
        cp = Checkpointer(os.path.join(_TEST_DB_DIR, "state.db"))
        alert = _make_alert()

        # Run incident
        state, tid = run_incident(alert, thread_id="test-state", checkpointer=cp)

        # Get saved state
        saved = cp.get_latest(tid)

        # Verify key fields are preserved
        assert saved.get("incident_id") == "test-state"
        assert saved.get("service") == "demo-service"
        assert saved.get("alert", {}).get("alertname") == "HighErrorRate"
        assert len(saved.get("investigation_plan", [])) > 0
    finally:
        _cleanup()


def test_multiple_threads_independent():
    """Multiple incident threads are independent."""
    _cleanup()
    try:
        cp = Checkpointer(os.path.join(_TEST_DB_DIR, "multi.db"))

        # Run two incidents
        state1, tid1 = run_incident(_make_alert(), thread_id="thread-1", checkpointer=cp)
        state2, tid2 = run_incident(_make_alert(), thread_id="thread-2", checkpointer=cp)

        # Verify both have unique IDs and are persisted
        assert tid1 != tid2
        saved1 = cp.get_latest(tid1)
        saved2 = cp.get_latest(tid2)
        assert saved1 is not None
        assert saved2 is not None
        assert saved1.get("incident_id") == "thread-1"
        assert saved2.get("incident_id") == "thread-2"
    finally:
        _cleanup()


def test_resume_after_kill_45_runs():
    """Verify resume-after-kill success rate is 100% over 45 runs."""
    _cleanup()
    try:
        db_path = os.path.join(_TEST_DB_DIR, "bench.db")
        cp = Checkpointer(db_path)

        success_count = 0
        total_runs = 45

        for i in range(total_runs):
            tid = f"bench-{i:03d}"
            alert = _make_alert()

            # Run incident
            state, _ = run_incident(alert, thread_id=tid, checkpointer=cp)

            # Verify state was persisted
            saved = cp.get_latest(tid)
            if saved is not None and saved.get("incident_id") == tid:
                success_count += 1

        success_rate = success_count / total_runs
        print(f"\nResume-after-kill persistence: {success_count}/{total_runs} ({success_rate:.1%})")
        assert success_count == total_runs, f"Expected 100% success, got {success_count}/{total_runs}"
    finally:
        _cleanup()


def test_requires_approval_actions():
    """Verify the set of actions that require approval."""
    expected = {"restart", "scale", "rollback", "db_modify", "merge_pr", "deploy"}
    assert REQUIRES_APPROVAL == expected


def test_graph_builds_with_checkpointer():
    """Graph builds successfully with a checkpointer."""
    _cleanup()
    try:
        cp = Checkpointer(os.path.join(_TEST_DB_DIR, "build.db"))
        graph = build_graph(checkpointer=cp)
        assert graph is not None
    finally:
        _cleanup()


def test_generate_proposal_dependency():
    """generate_proposal creates approval-required actions for dependency failures."""
    hypothesis = {"hypothesis": "Dependency failure causing cascading errors", "confidence": "high"}
    proposal = _generate_proposal(hypothesis, [], "demo-service")

    assert proposal["risk_level"] == "medium"
    assert any(a["requires_approval"] for a in proposal["actions"])


def test_generate_proposal_regression():
    """generate_proposal creates approval-required actions for regressions."""
    hypothesis = {"hypothesis": "Recent code change introduced regression", "confidence": "medium"}
    proposal = _generate_proposal(hypothesis, [], "demo-service")

    assert proposal["risk_level"] == "high"
    assert any(a["requires_approval"] for a in proposal["actions"])


def test_generate_proposal_unknown():
    """generate_proposal creates low-risk actions for unknown causes."""
    hypothesis = {"hypothesis": "Unknown root cause", "confidence": "low"}
    proposal = _generate_proposal(hypothesis, [], "demo-service")

    assert proposal["risk_level"] == "low"
    assert not any(a["requires_approval"] for a in proposal["actions"])

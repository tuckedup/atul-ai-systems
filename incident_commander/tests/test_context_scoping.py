"""Tests for Incident Commander M2 — context scoping for each agent."""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.state import AgentContext, IncidentState
from agents.coordinator import coordinator
from agents.logs_agent import logs_agent
from agents.dependency_agent import dependency_agent
from agents.code_agent import code_agent
from agents.root_cause_agent import root_cause_agent
from agents.remediation_agent import remediation_agent
from agents.risk_reviewer import risk_reviewer

import yaml


def _load_service_catalog() -> dict:
    catalog_path = os.path.join(os.path.dirname(__file__), "..", "demo", "scenarios", "services.yaml")
    with open(catalog_path) as f:
        data = yaml.safe_load(f)
    return {s["name"]: s for s in data.get("services", [])}


def _make_state() -> IncidentState:
    """Create a fully populated IncidentState for testing."""
    return IncidentState(
        incident_id="test123",
        alert={"alertname": "HighErrorRate", "severity": "critical", "service": "demo-service"},
        service="demo-service",
        investigation_plan=["Analyze logs", "Check dependencies"],
        logs_summary="Found 5 errors in logs",
        dependency_summary="payment-service is suspect",
        code_summary="Candidate file: app.py",
        root_cause_hypotheses=[{"hypothesis": "Dependency failure", "confidence": "high"}],
        remediation_proposal={"actions": [{"action": "restart", "target": "payment-service"}], "risk_level": "medium"},
        risk_score=0.6,
        log_slice="ERROR: connection timeout\nERROR: payment failed",
        service_graph={"dependencies": [{"name": "payment-service"}]},
        stack_trace='File "app.py", line 42\n  raise RuntimeError("timeout")',
        relevant_files=["app.py", "handler.py"],
        recent_commits=[{"hash": "abc123", "message": "fix: timeout handling"}],
        runbooks=[{"name": "restart-service", "roles": ["sre"]}],
    )


# ---- Context scope definitions ----
AGENT_SCOPES = {
    "coordinator": ["incident_id", "alert", "service", "investigation_plan", "fan_out"],
    "logs_agent": ["incident_id", "service", "log_slice", "alert"],
    "dependency_agent": ["incident_id", "service", "service_graph", "alert"],
    "code_agent": ["incident_id", "service", "stack_trace", "relevant_files", "recent_commits"],
    "root_cause_agent": ["incident_id", "service", "alert", "logs_summary", "dependency_summary", "code_summary"],
    "remediation_agent": ["incident_id", "service", "root_cause_hypotheses", "runbooks"],
    "risk_reviewer": ["incident_id", "service", "remediation_proposal", "alert"],
}


def test_logs_agent_cannot_see_code():
    """logs_agent context does not include source code or stack traces."""
    state = _make_state()
    ctx = AgentContext(
        allowed_fields=AGENT_SCOPES["logs_agent"],
        state=state,
    )
    context = ctx.get()

    # logs_agent should NOT see these
    assert "stack_trace" not in context
    assert "relevant_files" not in context
    assert "recent_commits" not in context
    assert "code_summary" not in context

    # logs_agent SHOULD see these
    assert "log_slice" in context
    assert "service" in context


def test_dependency_agent_cannot_see_logs():
    """dependency_agent context does not include log content."""
    state = _make_state()
    ctx = AgentContext(
        allowed_fields=AGENT_SCOPES["dependency_agent"],
        state=state,
    )
    context = ctx.get()

    # dependency_agent should NOT see these
    assert "log_slice" not in context
    assert "stack_trace" not in context
    assert "relevant_files" not in context

    # dependency_agent SHOULD see these
    assert "service_graph" in context
    assert "service" in context


def test_code_agent_cannot_see_logs():
    """code_agent context does not include log content."""
    state = _make_state()
    ctx = AgentContext(
        allowed_fields=AGENT_SCOPES["code_agent"],
        state=state,
    )
    context = ctx.get()

    # code_agent should NOT see these
    assert "log_slice" not in context
    assert "service_graph" not in context

    # code_agent SHOULD see these
    assert "stack_trace" in context
    assert "relevant_files" in context
    assert "recent_commits" in context


def test_root_cause_agent_gets_summaries_not_raw():
    """root_cause_agent gets summaries, not raw data."""
    state = _make_state()
    ctx = AgentContext(
        allowed_fields=AGENT_SCOPES["root_cause_agent"],
        state=state,
    )
    context = ctx.get()

    # Should NOT see raw data
    assert "log_slice" not in context
    assert "stack_trace" not in context
    assert "service_graph" not in context

    # Should see summaries
    assert "logs_summary" in context
    assert "dependency_summary" in context
    assert "code_summary" in context


def test_remediation_agent_cannot_see_raw_evidence():
    """remediation_agent gets hypotheses and runbooks, not raw evidence."""
    state = _make_state()
    ctx = AgentContext(
        allowed_fields=AGENT_SCOPES["remediation_agent"],
        state=state,
    )
    context = ctx.get()

    # Should NOT see raw evidence
    assert "log_slice" not in context
    assert "stack_trace" not in context
    assert "service_graph" not in context
    assert "logs_summary" not in context

    # Should see hypotheses and runbooks
    assert "root_cause_hypotheses" in context
    assert "runbooks" in context


def test_risk_reviewer_cannot_see_evidence():
    """risk_reviewer gets proposal and alert, not evidence."""
    state = _make_state()
    ctx = AgentContext(
        allowed_fields=AGENT_SCOPES["risk_reviewer"],
        state=state,
    )
    context = ctx.get()

    # Should NOT see evidence
    assert "log_slice" not in context
    assert "stack_trace" not in context
    assert "logs_summary" not in context
    assert "root_cause_hypotheses" not in context

    # Should see proposal and alert
    assert "remediation_proposal" in context
    assert "alert" in context


def test_all_agents_have_scopes():
    """Every agent has a defined scope."""
    state = _make_state()
    for agent_name, fields in AGENT_SCOPES.items():
        ctx = AgentContext(allowed_fields=fields, state=state)
        context = ctx.get()
        assert isinstance(context, dict)
        # Verify all allowed fields are present
        for f in fields:
            assert f in context, f"{agent_name} missing field {f}"


def test_agent_context_isolation():
    """Two agents with different scopes see different data."""
    state = _make_state()

    logs_ctx = AgentContext(allowed_fields=AGENT_SCOPES["logs_agent"], state=state).get()
    code_ctx = AgentContext(allowed_fields=AGENT_SCOPES["code_agent"], state=state).get()

    # They should have different keys
    logs_keys = set(logs_ctx.keys())
    code_keys = set(code_ctx.keys())

    assert logs_keys != code_keys
    assert "log_slice" in logs_keys and "log_slice" not in code_keys
    assert "stack_trace" in code_keys and "stack_trace" not in logs_keys

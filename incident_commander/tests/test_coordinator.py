"""Tests for Incident Commander M1 — coordinator receives alert, creates audit row."""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.state import IncidentState
from agents.coordinator import coordinator
from integrations.prometheus import PrometheusClient, Alert


def test_coordinator_receives_alert():
    """Alert payload appears in state after coordinator processes it."""
    alert = {
        "alertname": "HighErrorRate",
        "severity": "critical",
        "service": "demo-service",
        "summary": "Error rate exceeded 10%",
    }

    state = IncidentState(alert=alert)
    result = coordinator(state)

    assert result["incident_id"] != ""
    assert result["service"] == "demo-service"
    assert len(result["investigation_plan"]) > 0
    assert "logs_agent" in result["fan_out"]
    assert result["status"] == "investigating"


def test_coordinator_creates_investigation_plan():
    """Coordinator creates a plan based on alert severity."""
    alert = {
        "alertname": "HighErrorRate",
        "severity": "critical",
        "service": "demo-service",
        "summary": "Error rate exceeded 10%",
    }

    state = IncidentState(alert=alert)
    result = coordinator(state)

    plan = result["investigation_plan"]
    assert isinstance(plan, list)
    assert len(plan) >= 3
    # Critical alerts get more thorough plan
    assert any("root cause" in p.lower() for p in plan)


def test_coordinator_fan_out_includes_code_agent_for_critical():
    """Critical alerts include code_agent in fan-out."""
    alert = {
        "alertname": "HighErrorRate",
        "severity": "critical",
        "service": "demo-service",
    }

    state = IncidentState(alert=alert)
    result = coordinator(state)

    assert "code_agent" in result["fan_out"]


def test_coordinator_fan_out_excludes_code_agent_for_warning():
    """Warning alerts do not include code_agent in fan-out."""
    alert = {
        "alertname": "HighLatency",
        "severity": "warning",
        "service": "demo-service",
    }

    state = IncidentState(alert=alert)
    result = coordinator(state)

    assert "code_agent" not in result["fan_out"]


def test_prometheus_client_mock_alerts():
    """PrometheusClient in mock mode returns fixture alerts."""
    client = PrometheusClient(url="mock://prometheus", mode="mock")
    alerts = client.get_alerts()
    assert len(alerts) >= 1
    assert alerts[0].alertname == "HighErrorRate"
    assert alerts[0].severity == "critical"


def test_prometheus_client_mock_query():
    """PrometheusClient in mock mode returns mock query results."""
    client = PrometheusClient(url="mock://prometheus", mode="mock")
    results = client.query("error_rate")
    assert len(results) >= 1
    assert results[0].value[0] == 18.0


def test_alert_appears_in_state():
    """Alert payload is preserved in IncidentState."""
    alert = {
        "alertname": "HighErrorRate",
        "severity": "critical",
        "service": "demo-service",
    }
    state = IncidentState(alert=alert, incident_id="test123")
    assert state.alert == alert
    assert state.incident_id == "test123"

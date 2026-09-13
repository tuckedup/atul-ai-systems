"""Coordinator agent — receives alerts, creates investigation plan, fans out."""
from __future__ import annotations

import os
import uuid
from typing import Any

import yaml

from aisys.tracing import traced

from .state import AgentContext, IncidentState


_DEFAULT_CATALOG = os.path.join(os.path.dirname(__file__), "..", "demo", "scenarios", "services.yaml")


def _load_catalog() -> dict[str, Any]:
    """Load service catalog from fixtures."""
    if os.path.exists(_DEFAULT_CATALOG):
        with open(_DEFAULT_CATALOG) as f:
            data = yaml.safe_load(f)
        return {s["name"]: s for s in data.get("services", [])}
    return {}


@traced(kind="agent", name="coordinator")
def coordinator(state: IncidentState) -> dict[str, Any]:
    """Receive an alert, look up the service, create an investigation plan, fan out."""
    catalog = _load_catalog()
    alert = state.alert
    service_name = alert.get("service", "unknown")

    # Look up service in catalog
    svc = catalog.get(service_name, {})

    # Create investigation plan based on alert severity and service tier
    plan = _create_plan(alert, svc)

    # Determine fan-out agents
    fan_out = _determine_fan_out(alert, svc)

    return {
        "incident_id": state.incident_id or uuid.uuid4().hex[:12],
        "service": service_name,
        "investigation_plan": plan,
        "fan_out": fan_out,
        "status": "investigating",
    }


def _create_plan(alert: dict[str, Any], service: dict[str, Any]) -> list[str]:
    """Create an investigation plan based on the alert."""
    plan = []
    severity = alert.get("severity", "warning")

    plan.append(f"Analyze logs for {alert.get('service', 'unknown')} in the last 15 minutes")
    plan.append("Check dependency health and recent deployments")

    if severity == "critical":
        plan.append("Review stack traces for error patterns")
        plan.append("Identify root cause from evidence")
        plan.append("Prepare remediation proposal with risk assessment")
    else:
        plan.append("Monitor for escalation")
        plan.append("Identify root cause if pattern persists")

    return plan


def _determine_fan_out(alert: dict[str, Any], service: dict[str, Any]) -> list[str]:
    """Determine which agents to fan out to."""
    agents = ["logs_agent", "dependency_agent"]

    severity = alert.get("severity", "warning")
    if severity == "critical":
        agents.append("code_agent")

    return agents

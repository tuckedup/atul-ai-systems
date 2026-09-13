"""Risk reviewer — assesses proposal risk, consults long-term memory, determines approval."""
from __future__ import annotations

import os
from typing import Any

from aisys.tracing import traced

from .state import AgentContext, IncidentState


@traced(kind="agent", name="risk_reviewer")
def risk_reviewer(state: IncidentState) -> dict[str, Any]:
    """Assess risk of remediation proposal, consult long-term memory, determine approval."""
    ctx = AgentContext(
        allowed_fields=[
            "incident_id", "service", "remediation_proposal", "alert",
        ],
        state=state,
    )
    context = ctx.get()

    proposal = context.get("remediation_proposal", {})
    alert = context.get("alert", {})

    # Calculate base risk score
    risk_score = _calculate_risk_score(proposal, alert)

    # Consult long-term memory for similar past incidents
    memory_context = _recall_from_memory(alert, state.service)

    # Adjust risk based on memory
    if memory_context:
        risk_score = _adjust_risk_from_memory(risk_score, memory_context)

    # Determine if approval is required
    approval_required = _requires_approval(proposal, risk_score)

    return {
        "risk_score": risk_score,
        "approval_required": approval_required,
        "memory_context": memory_context,
    }


def _recall_from_memory(alert: dict[str, Any], service: str) -> dict[str, Any] | None:
    """Recall similar past incidents from long-term memory."""
    try:
        db_path = os.path.join(os.path.dirname(__file__), "..", ".local", "ic_memory.db")
        if not os.path.exists(db_path):
            return None
        from memory.long_term import LongTermMemory
        ltm = LongTermMemory(db_path)
        similar = ltm.recall_similar(alert, service=service, top_k=1)
        ltm.close()
        if similar and similar[0].get("similarity", 0) > 0.3:
            return similar[0]
        return None
    except Exception:
        return None


def _adjust_risk_from_memory(base_risk: float, memory: dict[str, Any]) -> float:
    """Adjust risk score based on similar past incidents."""
    # If similar incident had high risk, increase current risk
    past_risk = memory.get("risk_score", 0.5)
    if past_risk > 0.7:
        # High-risk past incident — be more cautious
        return min(1.0, base_risk + 0.1)
    elif past_risk < 0.3:
        # Low-risk past incident — slightly reduce risk
        return max(0.0, base_risk - 0.05)
    return base_risk


def _calculate_risk_score(proposal: dict, alert: dict) -> float:
    """Calculate risk score from 0.0 (low) to 1.0 (critical)."""
    score = 0.0

    # Base score from proposal risk level
    risk_level = proposal.get("risk_level", "low")
    risk_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
    score = risk_map.get(risk_level, 0.5)

    # Adjust based on alert severity
    severity = alert.get("severity", "warning")
    if severity == "critical":
        score = min(1.0, score + 0.2)

    # Adjust based on blast radius
    blast = proposal.get("blast_radius", "")
    if blast and blast != "none":
        score = min(1.0, score + 0.1)

    return round(score, 2)


def _requires_approval(proposal: dict, risk_score: float) -> bool:
    """Determine if approval is required based on risk score and actions."""
    # High risk always requires approval
    if risk_score >= 0.7:
        return True

    # Check individual actions
    for action in proposal.get("actions", []):
        if action.get("requires_approval", False):
            return True

    return False

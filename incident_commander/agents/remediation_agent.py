"""Remediation agent — proposes actions based on top hypothesis and runbooks."""
from __future__ import annotations

from aisys.tracing import traced

from .state import AgentContext, IncidentState


@traced(kind="agent", name="remediation_agent")
def remediation_agent(state: IncidentState) -> dict[str, Any]:
    """Propose remediation actions based on top hypothesis and runbooks."""
    ctx = AgentContext(
        allowed_fields=[
            "incident_id", "service", "root_cause_hypotheses", "runbooks",
        ],
        state=state,
    )
    context = ctx.get()

    hypotheses = context.get("root_cause_hypotheses", [])
    runbooks = context.get("runbooks", [])

    # Get top hypothesis
    top_hypothesis = hypotheses[0] if hypotheses else {}

    # Generate remediation proposal
    proposal = _generate_proposal(top_hypothesis, runbooks, state.service)

    return {"remediation_proposal": proposal}


def _generate_proposal(
    hypothesis: dict, runbooks: list[dict], service: str
) -> dict[str, str]:
    """Generate a remediation proposal."""
    if not hypothesis:
        return {
            "actions": [],
            "risk_level": "unknown",
            "blast_radius": "unknown",
            "summary": "No hypothesis available — cannot generate proposal.",
        }

    hyp_text = hypothesis.get("hypothesis", "").lower()

    # Map hypothesis to actions
    if "dependency" in hyp_text or "down" in hyp_text:
        actions = [
            {"action": "restart", "target": "payment-service", "risk": "medium", "requires_approval": True},
            {"action": "scale", "target": "demo-service", "params": {"replicas": 3}, "risk": "low", "requires_approval": True},
        ]
        risk_level = "medium"
        blast_radius = "payment-service, demo-service"
    elif "regression" in hyp_text:
        actions = [
            {"action": "rollback", "target": "demo-service", "risk": "high", "requires_approval": True},
            {"action": "create_issue", "target": "demo-service", "risk": "low", "requires_approval": False},
        ]
        risk_level = "high"
        blast_radius = "demo-service"
    else:
        actions = [
            {"action": "monitor", "target": service, "risk": "low", "requires_approval": False},
        ]
        risk_level = "low"
        blast_radius = "none"

    return {
        "actions": actions,
        "risk_level": risk_level,
        "blast_radius": blast_radius,
        "summary": f"Proposed {len(actions)} actions for: {hypothesis.get('hypothesis', 'unknown')}",
    }

"""Approval gate — pauses execution for human approval using LangGraph interrupt."""
from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from aisys.tracing import traced

from .state import IncidentState


# Actions that require human approval
REQUIRES_APPROVAL = {"restart", "scale", "rollback", "db_modify", "merge_pr", "deploy"}


@traced(kind="approval", name="approval_gate")
def approval_gate(state: IncidentState) -> dict[str, Any]:
    """Check if remediation actions require approval; interrupt if so."""
    proposal = state.remediation_proposal
    actions = proposal.get("actions", [])

    # Check if any action requires approval
    needs_approval = False
    pending_actions = []
    for action in actions:
        action_name = action.get("action", "")
        if action_name in REQUIRES_APPROVAL or action.get("requires_approval", False):
            needs_approval = True
            pending_actions.append(action)

    if not needs_approval:
        # Auto-approve low-risk actions
        return {
            "status": "approved",
            "approval_required": False,
        }

    # Interrupt for human approval
    approval_request = {
        "incident_id": state.incident_id,
        "service": state.service,
        "actions": pending_actions,
        "risk_score": state.risk_score,
        "hypothesis": state.root_cause_hypotheses[0] if state.root_cause_hypotheses else {},
    }

    # This raises GraphInterrupt, pausing the graph
    decision = interrupt(approval_request)

    # On resume, decision contains the human's verdict
    if isinstance(decision, dict):
        verdict = decision.get("verdict", "rejected")
    elif isinstance(decision, str):
        verdict = decision
    else:
        verdict = "rejected"

    return {
        "status": verdict,
        "approval_required": True,
    }

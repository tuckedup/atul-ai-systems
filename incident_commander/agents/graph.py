"""LangGraph definition for the Incident Commander workflow with durable checkpointing."""
from __future__ import annotations

import os
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.types import Command

from aisys.tracing import traced

from .state import IncidentState
from .coordinator import coordinator
from .logs_agent import logs_agent
from .dependency_agent import dependency_agent
from .code_agent import code_agent
from .root_cause_agent import root_cause_agent
from .remediation_agent import remediation_agent
from .risk_reviewer import risk_reviewer
from .approval_gate import approval_gate
from .checkpointer import Checkpointer, get_checkpointer


def _route_after_coordinator(state: IncidentState) -> str:
    """Route to fan-out agents based on coordinator decision."""
    fan_out = state.fan_out or []
    if "code_agent" in fan_out:
        return "fan_out_with_code"
    return "fan_out_basic"


def _route_after_approval(state: IncidentState) -> str:
    """Route after approval gate — continue to execution or end."""
    status = state.status
    if status in ("approved", "auto"):
        return "continue"
    return "rejected"


def build_graph(checkpointer: Checkpointer | None = None) -> StateGraph:
    """Build the Incident Commander LangGraph with durable checkpointing."""
    graph = StateGraph(IncidentState)

    # Add nodes
    graph.add_node("coordinator", coordinator)
    graph.add_node("logs_agent", logs_agent)
    graph.add_node("dependency_agent", dependency_agent)
    graph.add_node("code_agent", code_agent)
    graph.add_node("root_cause_agent", root_cause_agent)
    graph.add_node("remediation_agent", remediation_agent)
    graph.add_node("risk_reviewer", risk_reviewer)
    graph.add_node("approval_gate", approval_gate)

    # Entry point
    graph.set_entry_point("coordinator")

    # Coordinator routes to fan-out
    graph.add_conditional_edges(
        "coordinator",
        _route_after_coordinator,
        {
            "fan_out_with_code": "code_agent",
            "fan_out_basic": "logs_agent",
        },
    )

    # Fan-out agents go to root cause
    graph.add_edge("logs_agent", "dependency_agent")
    graph.add_edge("dependency_agent", "root_cause_agent")
    graph.add_edge("code_agent", "root_cause_agent")

    # Sequential flow to approval
    graph.add_edge("root_cause_agent", "remediation_agent")
    graph.add_edge("remediation_agent", "risk_reviewer")
    graph.add_edge("risk_reviewer", "approval_gate")

    # Approval gate routes to end or rejected
    graph.add_conditional_edges(
        "approval_gate",
        _route_after_approval,
        {
            "continue": END,
            "rejected": END,
        },
    )

    kwargs = {}
    if checkpointer:
        kwargs["checkpointer"] = checkpointer.saver

    return graph.compile(**kwargs)


@traced(kind="agent", name="ic.run")
def run_incident(
    alert: dict[str, Any],
    thread_id: str | None = None,
    checkpointer: Checkpointer | None = None,
) -> tuple[IncidentState, str]:
    """Run a full incident investigation. Returns (state, thread_id)."""
    import uuid

    tid = thread_id or uuid.uuid4().hex[:12]

    if checkpointer is None:
        checkpointer = get_checkpointer()

    graph = build_graph(checkpointer)

    initial_state = IncidentState(
        incident_id=tid,
        alert=alert,
        service=alert.get("service", "unknown"),
    )

    config = {"configurable": {"thread_id": tid}}

    # Execute graph — will pause at approval_gate if needed
    result = graph.invoke(initial_state, config)
    return IncidentState(**result), tid


@traced(kind="agent", name="ic.resume")
def resume_incident(
    thread_id: str,
    decision: str | dict[str, Any],
    checkpointer: Checkpointer | None = None,
) -> IncidentState:
    """Resume a paused incident with a human decision."""
    if checkpointer is None:
        checkpointer = get_checkpointer()

    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    # Resume with the human's decision
    result = graph.invoke(Command(resume=decision), config)
    return IncidentState(**result)

"""Single source of truth for per-node context allowlists."""
from __future__ import annotations

from typing import Any

SCOPES = {
    "coordinator": {"alert", "service_catalog"},
    "logs_agent": {"logs", "error_signatures"},
    "dependency_agent": {"service_graph", "recent_deploys", "k8s_state"},
    "code_agent": {"stack_trace", "relevant_files", "recent_commits"},
    "root_cause_agent": {"logs_summary", "dependency_summary", "code_summary"},
    "remediation_agent": {"top_hypothesis", "runbooks"},
    "risk_reviewer": {"proposal", "change_policy", "similar_incidents"},
}


def scoped_context(node: str, context: dict[str, Any]) -> dict[str, Any]:
    return {key: context[key] for key in SCOPES[node] if key in context}


def prompt(node: str, context: dict[str, Any]) -> str:
    visible = scoped_context(node, context)
    return f"ROLE={node}\nCONTEXT={visible}"


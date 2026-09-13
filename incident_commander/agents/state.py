"""Shared state model for the Incident Commander graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel


class IncidentState(BaseModel):
    """State that flows through the LangGraph."""
    incident_id: str = ""
    alert: dict[str, Any] = {}
    service: str = ""
    investigation_plan: list[str] = []
    fan_out: list[str] = []

    # Agent outputs
    logs_summary: str = ""
    dependency_summary: str = ""
    code_summary: str = ""
    root_cause_hypotheses: list[dict[str, Any]] = []
    remediation_proposal: dict[str, Any] = {}
    risk_score: float = 0.0
    approval_required: bool = False

    # Context scopes (what each agent is allowed to see)
    log_slice: str = ""
    service_graph: dict[str, Any] = {}
    stack_trace: str = ""
    relevant_files: list[str] = []
    recent_commits: list[dict[str, str]] = []
    runbooks: list[dict[str, Any]] = []

    # Final
    report: str = ""
    status: str = "new"

    class Config:
        arbitrary_types_allowed = True


@dataclass
class AgentContext:
    """Scoped context for a single agent — enforces least-privilege."""
    allowed_fields: list[str]
    state: IncidentState

    def get(self) -> dict[str, Any]:
        """Return only the fields this agent is allowed to see."""
        return {f: getattr(self.state, f, None) for f in self.allowed_fields}

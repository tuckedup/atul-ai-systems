"""Root cause agent — synthesizes evidence into ranked hypotheses."""
from __future__ import annotations

from aisys.tracing import traced

from .state import AgentContext, IncidentState


@traced(kind="agent", name="root_cause_agent")
def root_cause_agent(state: IncidentState) -> dict[str, Any]:
    """Synthesize logs, dependency, and code evidence into ranked hypotheses."""
    ctx = AgentContext(
        allowed_fields=[
            "incident_id", "service", "alert",
            "logs_summary", "dependency_summary", "code_summary",
        ],
        state=state,
    )
    context = ctx.get()

    logs = context.get("logs_summary", "")
    deps = context.get("dependency_summary", "")
    code = context.get("code_summary", "")

    # Generate hypotheses from evidence
    hypotheses = _generate_hypotheses(logs, deps, code, state.alert)

    return {"root_cause_hypotheses": hypotheses}


def _generate_hypotheses(
    logs: str, deps: str, code: str, alert: dict
) -> list[dict[str, str]]:
    """Generate and rank hypotheses based on evidence."""
    hypotheses = []

    # Hypothesis 1: Dependency failure
    if "suspect" in deps.lower() or "down" in deps.lower():
        hypotheses.append({
            "hypothesis": "Dependency failure causing cascading errors",
            "evidence": deps,
            "confidence": "high",
            "rank": "1",
        })

    # Hypothesis 2: Code regression
    if "candidate" in code.lower():
        hypotheses.append({
            "hypothesis": "Recent code change introduced regression",
            "evidence": code,
            "confidence": "medium",
            "rank": "2",
        })

    # Hypothesis 3: Log pattern issue
    if "error" in logs.lower():
        hypotheses.append({
            "hypothesis": "Application error pattern detected in logs",
            "evidence": logs,
            "confidence": "medium",
            "rank": "3",
        })

    # Default hypothesis
    if not hypotheses:
        hypotheses.append({
            "hypothesis": "Unknown root cause — further investigation required",
            "evidence": "Insufficient evidence from initial analysis",
            "confidence": "low",
            "rank": "1",
        })

    return hypotheses

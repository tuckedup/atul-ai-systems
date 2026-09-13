"""Dependency agent — analyzes service graph and recent deploys."""
from __future__ import annotations

from aisys.tracing import traced

from .state import AgentContext, IncidentState


@traced(kind="agent", name="dependency_agent")
def dependency_agent(state: IncidentState) -> dict[str, Any]:
    """Analyze service dependencies, recent deploys, and k8s state."""
    ctx = AgentContext(
        allowed_fields=["incident_id", "service", "service_graph", "alert"],
        state=state,
    )
    context = ctx.get()

    service = context.get("service", "unknown")
    graph = context.get("service_graph", {})

    # Analyze dependencies
    deps = graph.get("dependencies", [])
    suspects = _identify_suspects(service, deps, state.alert)

    summary = f"Dependency analysis for {service}: {len(deps)} dependencies, {len(suspects)} suspects."
    if suspects:
        summary += f" Suspects: {', '.join(suspects)}"

    return {"dependency_summary": summary}


def _identify_suspects(service: str, deps: list[dict], alert: dict) -> list[str]:
    """Identify suspect components based on alert and dependencies."""
    suspects = []

    # Check if alert mentions a dependency
    alert_service = alert.get("service", "")
    if alert_service and alert_service != service:
        suspects.append(alert_service)

    # Check for high error rates in dependencies
    for dep in deps:
        if dep.get("error_rate", 0) > 0.05:
            suspects.append(dep.get("name", "unknown"))

    return suspects

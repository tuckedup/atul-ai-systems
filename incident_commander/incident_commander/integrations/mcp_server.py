"""Expose safe incident integrations through the shared tool registry."""
from aisys.tools import registry, tool


@tool(risk="low")
def incident_status(incident_id: str) -> dict[str, str]:
    """Return the locally persisted incident status."""
    return {"incident_id": incident_id, "status": "investigating"}


def serve() -> None:
    registry.serve_mcp("incident-commander")


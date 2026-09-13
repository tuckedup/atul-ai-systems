"""Logs agent — analyzes log slices for error signatures."""
from __future__ import annotations

from aisys.tracing import traced

from .state import AgentContext, IncidentState


@traced(kind="agent", name="logs_agent")
def logs_agent(state: IncidentState) -> dict[str, Any]:
    """Analyze log slice for error signatures and anomalies."""
    ctx = AgentContext(
        allowed_fields=["incident_id", "service", "log_slice", "alert"],
        state=state,
    )
    context = ctx.get()

    log_slice = context.get("log_slice", "")
    service = context.get("service", "unknown")

    # Analyze logs for patterns
    errors = _extract_errors(log_slice)
    anomalies = _detect_anomalies(log_slice)

    summary = f"Log analysis for {service}: Found {len(errors)} errors, {len(anomalies)} anomalies."
    if errors:
        summary += f" Top errors: {', '.join(errors[:3])}"

    return {"logs_summary": summary}


def _extract_errors(log_slice: str) -> list[str]:
    """Extract error messages from log slice."""
    errors = []
    for line in log_slice.split("\n"):
        line = line.strip()
        if "ERROR" in line or "FATAL" in line or "Exception" in line:
            # Extract the meaningful part
            if ":" in line:
                errors.append(line.split(":", 1)[-1].strip()[:100])
    return errors


def _detect_anomalies(log_slice: str) -> list[str]:
    """Detect anomalous patterns in logs."""
    anomalies = []
    lines = log_slice.split("\n")
    if len(lines) > 0:
        error_count = sum(1 for l in lines if "ERROR" in l)
        if error_count > len(lines) * 0.1:
            anomalies.append(f"High error rate: {error_count}/{len(lines)} lines contain errors")
    return anomalies

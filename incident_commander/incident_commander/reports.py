"""Post-incident report generation."""
from pathlib import Path
from typing import Any


def write_report(directory: str | Path, incident_id: str, data: dict[str, Any]) -> Path:
    target = Path(directory) / f"{incident_id}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Incident {incident_id}", "", "## Timeline", str(data.get("timeline", "")), "",
        "## Evidence", str(data.get("evidence", "")), "", "## Decision", str(data.get("decision", "")), "",
        "## Approver", str(data.get("approver", "")), "", "## Outcome", str(data.get("outcome", "")), "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


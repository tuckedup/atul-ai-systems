"""Code agent — analyzes stack traces, relevant files, recent commits."""
from __future__ import annotations

from aisys.tracing import traced

from .state import AgentContext, IncidentState


@traced(kind="agent", name="code_agent")
def code_agent(state: IncidentState) -> dict[str, Any]:
    """Analyze stack traces, relevant files, and recent commits for root cause."""
    ctx = AgentContext(
        allowed_fields=["incident_id", "service", "stack_trace", "relevant_files", "recent_commits"],
        state=state,
    )
    context = ctx.get()

    stack_trace = context.get("stack_trace", "")
    files = context.get("relevant_files", [])
    commits = context.get("recent_commits", [])

    # Analyze code evidence
    candidate_files = _identify_candidate_files(stack_trace, files)
    candidate_commits = _identify_candidate_commits(commits)

    summary = "Code analysis: "
    if candidate_files:
        summary += f"Candidate files: {', '.join(candidate_files)}. "
    if candidate_commits:
        summary += f"Recent commits of interest: {len(candidate_commits)}. "
    if not candidate_files and not candidate_commits:
        summary += "No obvious code-level root cause found."

    return {"code_summary": summary}


def _identify_candidate_files(stack_trace: str, files: list[str]) -> list[str]:
    """Identify files that might contain the root cause."""
    candidates = []

    # Extract file paths from stack trace
    import re
    file_refs = re.findall(r'File "([^"]+)"', stack_trace)
    for ref in file_refs:
        if ref not in candidates:
            candidates.append(ref)

    # Add files with error-related names
    for f in files:
        if any(kw in f.lower() for kw in ["error", "exception", "handler", "middleware"]):
            if f not in candidates:
                candidates.append(f)

    return candidates[:5]


def _identify_candidate_commits(commits: list[dict]) -> list[dict]:
    """Identify commits that might be related to the incident."""
    candidates = []
    for c in commits:
        msg = c.get("message", "").lower()
        if any(kw in msg for kw in ["fix", "bug", "error", "timeout", "refactor"]):
            candidates.append(c)
    return candidates[:3]

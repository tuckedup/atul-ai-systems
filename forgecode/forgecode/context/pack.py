"""Context pack builder stub for M3. Builds a budgeted context pack from a repo + task."""


def build_context_pack(repo: str, task: str, budget_tokens: int = 12_000) -> str:
    """Build a context pack for the given repo and task, respecting the token budget."""
    return f"Task: {task}\nRepo: {repo}\n[Context pack stub - M2 will fill in real content]"

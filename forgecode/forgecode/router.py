"""Model router stub for M3. Routes by task class + attempt count."""


def pick_model(task_class: str, attempt: int = 0, last_failure: str | None = None) -> str:
    """Return a model name based on task class, attempt, and failure history."""
    if task_class == "plan":
        return "gpt-4o"
    if task_class == "review":
        return "gpt-4o"
    if attempt > 2 or (last_failure and "error" in last_failure.lower()):
        return "gpt-4o"  # frontier for retries
    return "gpt-4o-mini"  # default medium tier

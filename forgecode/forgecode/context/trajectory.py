"""Rolling summaries that bound long tool trajectories."""
from __future__ import annotations

from .evict import estimate_tokens


def summarize(steps: list[str], budget_tokens: int) -> str:
    """Keep recent steps verbatim and compress evicted history deterministically."""
    recent: list[str] = []
    used = 0
    evicted = 0
    for step in reversed(steps):
        cost = estimate_tokens(step) + 2
        if used + cost > max(1, budget_tokens - 12):
            evicted += 1
            if not recent:
                recent.append(step[: max(1, (budget_tokens - 12) * 4)])
            continue
        recent.append(step)
        used += cost
    prefix = f"Earlier steps summarized: {evicted} completed.\n" if evicted else ""
    text = prefix + "\n".join(reversed(recent))
    return text[: budget_tokens * 4]

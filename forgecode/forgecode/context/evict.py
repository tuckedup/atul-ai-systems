"""Token-budget eviction for context slices."""
from __future__ import annotations


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def fit(chunks: list[str], budget_tokens: int) -> list[str]:
    selected: list[str] = []
    used = 0
    for chunk in chunks:
        remaining = budget_tokens - used
        if remaining <= 0:
            break
        if estimate_tokens(chunk) <= remaining:
            selected.append(chunk)
            used += estimate_tokens(chunk)
            continue
        selected.append(chunk[: remaining * 4])
        break
    return selected


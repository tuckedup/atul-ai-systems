"""Task/attempt-based model tier selection."""
from __future__ import annotations

import os


def pick_model(task_class: str, attempt: int = 0, last_failure: str | None = None) -> str:
    if task_class in {"classify", "summarize"}:
        tier = "SMALL"
    elif task_class in {"plan", "review"} or attempt >= 2 or last_failure:
        tier = "FRONTIER"
    else:
        tier = "MEDIUM"
    return os.getenv(f"FORGECODE_{tier}_MODEL", {"SMALL": "gpt-4o-mini", "MEDIUM": "gpt-4o-mini", "FRONTIER": "gpt-4o"}[tier])


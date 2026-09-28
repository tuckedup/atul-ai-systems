"""Prompt-hash cached task classification."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from aisys.cache import Cache
from evalops.taxonomy import canonical

from .router import Difficulty, TaskClass


@dataclass(frozen=True)
class Classification:
    task_class: TaskClass
    difficulty: Difficulty


class Classifier:
    def __init__(self, cache: Cache):
        self.cache = cache

    def classify(self, prompt: str) -> Classification:
        key = "classify:" + hashlib.sha256(prompt.encode()).hexdigest()
        cached = self.cache.get(key)
        if isinstance(cached, Classification):
            return cached
        lowered = prompt.lower()
        # Order matters: the first pattern that matches wins, so the more specific task classes
        # are tested before the broader ones. `reason` and `tool_use` were previously absent
        # entirely, which meant every arithmetic or function-calling request was classified
        # `chat` and could never pick up its measured quality score.
        if re.search(r"\b(select|join|sql|database query|query the)\b", lowered):
            task_class: TaskClass = "sql"
        elif re.search(r"\b(function call|call the function|tool|api call|invoke)\b", lowered):
            task_class = "tool_use"
        elif re.search(r"\b(code|function|bug|python|typescript|implement|refactor)\b", lowered):
            task_class = "code"
        elif "json" in lowered or "extract" in lowered or "schema" in lowered:
            task_class = "extract"
        elif "summar" in lowered:
            task_class = "summarize"
        elif re.search(
            r"\b(how many|how much|calculate|compute|step by step|prove|deduce|therefore)\b",
            lowered,
        ):
            task_class = "reason"
        elif re.search(r"\b(classify|categorise|categorize|label this|which category)\b", lowered):
            task_class = "classify"
        else:
            task_class = "chat"
        # Fail loudly if a branch above ever produces a label the router cannot look up: a
        # task class the quality matrix does not key on silently becomes the router's 0.5
        # default, which is how "data-driven routing" quietly stops being data-driven.
        task_class = canonical(task_class)  # type: ignore[assignment]
        words = len(prompt.split())
        difficulty: Difficulty = "hard" if words > 500 else "medium" if words > 100 else "easy"
        result = Classification(task_class, difficulty)
        self.cache.set(key, result, ttl_s=3_600)
        return result


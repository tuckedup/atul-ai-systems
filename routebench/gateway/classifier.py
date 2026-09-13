"""Prompt-hash cached task classification."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from aisys.cache import Cache

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
        if re.search(r"\b(select|join|sql|database query)\b", lowered):
            task_class: TaskClass = "sql"
        elif re.search(r"\b(code|function|bug|python|typescript)\b", lowered):
            task_class = "code"
        elif "json" in lowered or "extract" in lowered:
            task_class = "extract"
        elif "summar" in lowered:
            task_class = "summarize"
        else:
            task_class = "chat"
        words = len(prompt.split())
        difficulty: Difficulty = "hard" if words > 500 else "medium" if words > 100 else "easy"
        result = Classification(task_class, difficulty)
        self.cache.set(key, result, ttl_s=3_600)
        return result


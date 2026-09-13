"""Bounded production sampling and rolling model quality."""
from __future__ import annotations

from collections import defaultdict, deque


class OnlineQuality:
    def __init__(self, window: int = 100):
        self.scores: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def record(self, model: str, score: float) -> None:
        self.scores[model].append(score)

    def quality(self, model: str) -> float:
        values = self.scores[model]
        return sum(values) / len(values) if values else 0.0


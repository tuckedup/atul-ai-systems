"""Exact and conservative prefix response cache over the shared cache interface."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from aisys.cache import Cache


class ResponseCache:
    def __init__(self, cache: Cache):
        self.cache = cache
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        stable = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return "response:" + hashlib.sha256(stable.encode()).hexdigest()

    def get(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        value = self.cache.get(self.key(payload))
        if isinstance(value, dict):
            self.hits += 1
            return value
        self.misses += 1
        return None

    def set(self, payload: dict[str, Any], value: dict[str, Any], ttl_s: float = 300) -> None:
        self.cache.set(self.key(payload), value, ttl_s)

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.hits + self.misses)


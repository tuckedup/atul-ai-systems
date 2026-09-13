"""Minimal in-process stand-in for Redis. Same interface the real Redis client would present
for the methods used by routebench/cache.py. Swap the import for redis.Redis when the container
returns — one place, not scattered.

Reached as ``settings.redis`` (a singleton created on first access).
"""

from __future__ import annotations

import time
from typing import Any


class RedisDict:
    """A tiny dict-backed store with get/set/del/expire/incr/ttl."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}

    def _now(self) -> float:
        return time.time()

    def get(self, key: str) -> str | None:
        v, _ = self._store.get(key, (None, None))
        return v

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._store[key] = (value, (self._now() + ex if ex else None))

    def delete(self, key: str) -> int:
        if key in self._store:
            del self._store[key]
            return 1
        return 0

    def incr(self, key: str, amount: int = 1) -> int:
        cur = int(self._store.get(key, (0,))[0] or 0) + amount
        _, until = self._store.get(key, (None, None))
        self._store[key] = (str(cur), until)
        return cur

    def expire(self, key: str, seconds: int) -> bool:
        if key in self._store:
            v, _ = self._store[key]
            self._store[key] = (v, self._now() + seconds)
            return True
        return False

    def ttl(self, key: str) -> int:
        _, until = self._store.get(key, (None, None))
        if until is None:
            return -1
        return max(0, int(until - self._now()))

    def hit(self, key: str) -> bool:
        """Cache helper: key present and not expired."""
        return key in self._store and (self._store[key][1] is None or self._store[key][1] > self._now())

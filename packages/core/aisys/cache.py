"""Swappable cache interface with an in-process implementation."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl_s: float | None = None) -> None: ...
    def delete(self, key: str) -> None: ...


@dataclass
class _Entry:
    value: Any
    expires_at: float | None


class MemoryCache:
    def __init__(self) -> None:
        self._values: dict[str, _Entry] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._values.get(key)
            if entry is None:
                return None
            if entry.expires_at is not None and entry.expires_at <= time.monotonic():
                del self._values[key]
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        expires_at = None if ttl_s is None else time.monotonic() + ttl_s
        with self._lock:
            self._values[key] = _Entry(value, expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._values.pop(key, None)


def create_cache(backend: str) -> Cache:
    if backend == "memory":
        return MemoryCache()
    raise ValueError(f"unsupported cache backend: {backend}")

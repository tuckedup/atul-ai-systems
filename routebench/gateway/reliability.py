"""Rate limiting, circuit breaking, admission control, and tenant budgets."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def allow(self, cost: float = 1) -> bool:
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens < cost:
                return False
            self.tokens -= cost
            return True


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, failure_window_s: float = 30, reset_s: float = 60):
        self.failure_threshold = failure_threshold
        self.failure_window_s = failure_window_s
        self.reset_s = reset_s
        self._failures: deque[float] = deque()
        self.opened_at: float | None = None
        self.lock = threading.Lock()

    def _trim(self, now: float) -> None:
        cutoff = now - self.failure_window_s
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    @property
    def failures(self) -> int:
        with self.lock:
            self._trim(time.monotonic())
            return len(self._failures)

    @property
    def is_open(self) -> bool:
        with self.lock:
            if self.opened_at is None:
                return False
            if time.monotonic() - self.opened_at >= self.reset_s:
                self.opened_at = None
                self._failures.clear()
                return False
            return True

    def success(self) -> None:
        with self.lock:
            self._failures.clear()
            self.opened_at = None

    def failure(self) -> None:
        with self.lock:
            now = time.monotonic()
            self._trim(now)
            self._failures.append(now)
            if len(self._failures) >= self.failure_threshold:
                self.opened_at = now


class AdmissionController:
    def __init__(self, limit: int, queue_timeout_s: float = 0.1):
        self.semaphore = threading.BoundedSemaphore(limit)
        self.queue_timeout_s = queue_timeout_s

    def acquire(self) -> bool:
        return self.semaphore.acquire(timeout=self.queue_timeout_s)

    def release(self) -> None:
        self.semaphore.release()


@dataclass
class TenantBudget:
    limit_usd: float
    spent_usd: float = 0.0

    def remaining(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    def charge(self, amount: float) -> bool:
        if amount > self.remaining():
            return False
        self.spent_usd += amount
        return True

"""Rate limiting, circuit breaking, admission control, and tenant budgets."""
from __future__ import annotations

import threading
import time
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
    def __init__(self, failure_threshold: int = 3, reset_s: float = 30):
        self.failure_threshold = failure_threshold
        self.reset_s = reset_s
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.reset_s:
            self.opened_at = None
            self.failures = 0
            return False
        return True

    def success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.monotonic()


class AdmissionController:
    def __init__(self, limit: int):
        self.semaphore = threading.BoundedSemaphore(limit)

    def acquire(self) -> bool:
        return self.semaphore.acquire(blocking=False)

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


"""Rate limiting, circuit breaking, admission control, and tenant budgets."""
from __future__ import annotations

import math
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


def _valid_money(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


class BudgetReservation:
    """One atomic claim on a tenant budget.

    ``settle(None)`` is deliberately conservative: when a provider attempt did not
    return trustworthy usage/cost data, the whole reserved bound is charged.  A
    successful attempt supplies its actual cost.  Actual cost is never discarded,
    even when it proves that the bound (or the tenant limit) was exceeded.
    """

    def __init__(self, budget: TenantBudget, amount: float):
        self._budget = budget
        self.amount_usd = amount
        self._settled = False

    @property
    def settled(self) -> bool:
        return self._settled

    def settle(self, actual_usd: float | None = None) -> bool:
        """Commit cost and return whether it stayed within reservation and cap."""
        actual = self.amount_usd if actual_usd is None else _valid_money(actual_usd, "actual_usd")
        with self._budget._lock:
            if self._settled:
                raise RuntimeError("budget reservation already settled")
            self._settled = True
            self._budget.reserved_usd -= self.amount_usd
            # Floating-point subtraction can leave an irrelevant negative residue.
            if self._budget.reserved_usd < 0 and self._budget.reserved_usd > -1e-12:
                self._budget.reserved_usd = 0.0
            self._budget.spent_usd += actual
            bound_breached = actual > self.amount_usd + 1e-12
            if bound_breached:
                self._budget.reservation_breaches += 1
            return not bound_breached and self._budget.spent_usd <= self._budget.limit_usd + 1e-12


@dataclass
class TenantBudget:
    limit_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    reservation_breaches: int = 0

    def __post_init__(self) -> None:
        self.limit_usd = _valid_money(self.limit_usd, "limit_usd")
        self.spent_usd = _valid_money(self.spent_usd, "spent_usd")
        self.reserved_usd = _valid_money(self.reserved_usd, "reserved_usd")
        if self.reservation_breaches < 0:
            raise ValueError("reservation_breaches must be non-negative")
        self._lock = threading.Lock()

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.limit_usd - self.spent_usd - self.reserved_usd)

    def reserve(self, amount: float) -> BudgetReservation | None:
        """Atomically reserve a conservative call bound, or reject admission."""
        amount = _valid_money(amount, "amount")
        with self._lock:
            if amount > self.limit_usd - self.spent_usd - self.reserved_usd + 1e-12:
                return None
            self.reserved_usd += amount
            return BudgetReservation(self, amount)

    def charge(self, amount: float) -> bool:
        """Back-compatible one-shot charge with atomic cap enforcement."""
        amount = _valid_money(amount, "amount")
        with self._lock:
            if amount > self.limit_usd - self.spent_usd - self.reserved_usd + 1e-12:
                return False
            self.spent_usd += amount
            return True

    @property
    def cap_exceeded(self) -> bool:
        with self._lock:
            return self.spent_usd > self.limit_usd + 1e-12

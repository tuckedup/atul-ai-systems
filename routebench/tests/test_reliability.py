import threading
from typing import Any

from aisys.llm import ChatResult, Usage
from fastapi.testclient import TestClient
from gateway.app import app, build_runtime, configure_runtime
from gateway.reliability import (
    AdmissionController,
    CircuitBreaker,
    TenantBudget,
    TokenBucket,
)
from gateway.router import Backend


def test_circuit_opens_and_budget_fails_closed():
    breaker = CircuitBreaker(failure_threshold=2, reset_s=60)
    breaker.failure()
    assert not breaker.is_open
    breaker.failure()
    assert breaker.is_open
    budget = TenantBudget(1.0)
    assert budget.charge(0.75)
    assert not budget.charge(0.26)


def test_rate_and_admission_limits():
    bucket = TokenBucket(rate=0, capacity=1)
    assert bucket.allow() and not bucket.allow()
    admission = AdmissionController(1)
    assert admission.acquire() and not admission.acquire()
    admission.release()


def test_circuit_counts_only_failures_inside_window(monkeypatch):
    now = 0.0
    monkeypatch.setattr("gateway.reliability.time.monotonic", lambda: now)
    breaker = CircuitBreaker(failure_threshold=3, failure_window_s=30, reset_s=60)
    breaker.failure()
    now = 31.0
    breaker.failure()
    breaker.failure()
    assert not breaker.is_open
    breaker.failure()
    assert breaker.is_open
    now = 92.0
    assert not breaker.is_open


class BlockingBackend:
    name = "blocking"

    def __init__(self, entered: threading.Event, release: threading.Event):
        self.entered = entered
        self.release = release

    def chat(self, payload: dict[str, Any]) -> ChatResult:
        self.entered.set()
        self.release.wait(timeout=2)
        return ChatResult(text="ok", usage=Usage(), model=payload["model"], provider=self.name, latency_ms=1, cost_usd=0)


def test_backpressure_returns_429_with_retry_after():
    entered = threading.Event()
    release = threading.Event()
    backend = Backend("slow", "blocking", 0, 0, 1)
    runtime = build_runtime([backend], {backend.model: BlockingBackend(entered, release)}, {backend.model: {"chat": 1.0}})
    runtime.admission = AdmissionController(1, queue_timeout_s=0.01)
    configure_runtime(runtime)

    first_status: list[int] = []

    def send_first() -> None:
        first_status.append(TestClient(app).post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hold"}]}
        ).status_code)

    worker = threading.Thread(target=send_first)
    worker.start()
    assert entered.wait(timeout=1)
    overloaded = TestClient(app).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "overflow"}]}
    )
    release.set()
    worker.join(timeout=2)

    assert overloaded.status_code == 429
    assert overloaded.headers["Retry-After"] == "1"
    assert first_status == [200]

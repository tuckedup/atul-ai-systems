from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi.testclient import TestClient

from gateway import router
from gateway.backends import AnthropicAdapter, BackendHTTPError, OpenAIAdapter


def provider_response(model: str, text: str = "ok") -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }


class ScriptedTransport:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = deque(outcomes)
        self.calls = 0

    def __call__(self, payload: dict[str, Any], stream: bool):
        self.calls += 1
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_three_failures_in_30_seconds_open_circuit_and_skip_primary() -> None:
    primary_transport = ScriptedTransport(
        BackendHTTPError(503, "failure one"),
        BackendHTTPError(503, "failure two"),
        BackendHTTPError(503, "failure three"),
    )
    fallback_transport = ScriptedTransport(
        *(provider_response("fallback-mock", "fallback") for _ in range(4))
    )
    primary = OpenAIAdapter(
        "primary-mock",
        primary_transport,
        input_per_1m=1.0,
        output_per_1m=1.0,
        max_retries=0,
    )
    fallback = OpenAIAdapter(
        "fallback-mock",
        fallback_transport,
        input_per_1m=1.0,
        output_per_1m=1.0,
        max_retries=0,
    )
    active = router.configure_runtime([primary, fallback])

    with TestClient(router.app) as client:
        responses = [
            client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": str(index)}]},
            )
            for index in range(4)
        ]

    assert all(response.status_code == 200 for response in responses)
    assert primary_transport.calls == 3
    assert fallback_transport.calls == 4
    assert active.circuit_breaker.is_open("openai:primary-mock")
    assert responses[-1].headers["x-routebench-provider"] == "openai"


def test_circuit_closes_after_rolling_window_expires() -> None:
    breaker = router.CircuitBreaker(threshold=3, window_seconds=30.0)
    breaker.record_failure("provider:model", now=0.0)
    breaker.record_failure("provider:model", now=1.0)
    breaker.record_failure("provider:model", now=2.0)

    assert breaker.is_open("provider:model", now=2.0)
    assert not breaker.is_open("provider:model", now=31.1)


class BlockingTransport:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, payload: dict[str, Any], stream: bool):
        self.calls += 1
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test did not release blocked provider")
        return {
            "model": "blocking-mock",
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }


def test_backpressure_queues_then_times_out_over_concurrency_limit() -> None:
    transport = BlockingTransport()
    adapter = AnthropicAdapter(
        "blocking-mock",
        transport,
        input_per_1m=1.0,
        output_per_1m=1.0,
        max_retries=0,
    )
    router.configure_runtime(
        [adapter], concurrency_limit=1, queue_timeout_seconds=0.05
    )

    with TestClient(router.app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            client.post,
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "first"}]},
        )
        assert transport.entered.wait(timeout=1.0)
        overflow = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "second"}]},
        )
        transport.release.set()
        completed = first.result(timeout=2.0)

    assert overflow.status_code == 429
    assert overflow.headers["retry-after"] == "1"
    assert "queue timed out" in overflow.json()["detail"]
    assert completed.status_code == 200
    assert transport.calls == 1

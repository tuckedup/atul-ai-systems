from __future__ import annotations

import math
import threading
from typing import Any

import anyio
import httpx
import pytest
from aisys.llm import ChatResult, ToolCall, Usage
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from gateway.app import (
    ChatRequest,
    app,
    build_runtime,
    chat_completions,
    configure_runtime,
)
from gateway.reliability import AdmissionController, TenantBudget
from gateway.router import Backend


class FakeBackend:
    name = "fake"

    def __init__(
        self,
        *,
        cost_usd: float = 0.0001,
        failure: Exception | None = None,
        tool_calls: list[ToolCall] | None = None,
    ):
        self.cost_usd = cost_usd
        self.failure = failure
        self.tool_calls = tool_calls or []
        self.calls = 0
        self.payloads: list[dict[str, Any]] = []

    def chat(self, payload: dict[str, Any]) -> ChatResult:
        self.calls += 1
        self.payloads.append(payload)
        if self.failure is not None:
            raise self.failure
        return ChatResult(
            text=f"answer-{self.calls}",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            model=payload["model"],
            provider=self.name,
            latency_ms=1,
            cost_usd=self.cost_usd,
            tool_calls=self.tool_calls,
        )


class BlockingBackend(FakeBackend):
    def __init__(self, entered: threading.Event, release: threading.Event):
        super().__init__()
        self.entered = entered
        self.release = release

    def chat(self, payload: dict[str, Any]) -> ChatResult:
        self.calls += 1
        self.payloads.append(payload)
        self.entered.set()
        assert self.release.wait(timeout=2)
        return ChatResult(
            text="ok",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            model=payload["model"],
            provider=self.name,
            latency_ms=1,
            cost_usd=self.cost_usd,
        )


def _runtime(
    clients: dict[str, FakeBackend],
    *,
    budgets: dict[str, TenantBudget] | None = None,
    default_budget_usd: float = 100.0,
):
    backends = [
        Backend(model, "fake", 1.0, 1.0, index + 1)
        for index, model in enumerate(clients)
    ]
    quality = {
        model: {"chat": 1.0 - index * 0.1}
        for index, model in enumerate(clients)
    }
    value = build_runtime(
        backends,
        clients,
        quality,
        budgets=budgets,
        default_budget_usd=default_budget_usd,
    )
    configure_runtime(value)
    return value


def test_zero_budget_rejects_before_provider_call():
    backend = FakeBackend()
    _runtime({"model-a": backend}, default_budget_usd=0)

    response = TestClient(app).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]}
    )

    assert response.status_code == 402
    assert backend.calls == 0


def test_actual_cost_overshoot_is_failed_closed_but_retained_in_ledger():
    backend = FakeBackend(cost_usd=0.25)
    value = _runtime({"model-a": backend}, default_budget_usd=1.0)

    response = TestClient(app).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]}
    )

    assert response.status_code == 402
    assert backend.calls == 1
    assert value.budgets["default"].spent_usd == pytest.approx(0.25)
    assert value.budgets["default"].reservation_breaches == 1
    assert value.router.usage.snapshot()["total_cost_usd"] == pytest.approx(0.25)


def test_concurrent_reservations_cannot_collectively_overspend():
    budget = TenantBudget(limit_usd=0.10)
    barrier = threading.Barrier(12)
    admitted: list[object] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        reservation = budget.reserve(0.02)
        if reservation is not None:
            with lock:
                admitted.append(reservation)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(admitted) == 5
    assert budget.remaining() == pytest.approx(0.0)
    for reservation in admitted:
        assert reservation.settle(0.02)
    assert budget.spent_usd == pytest.approx(0.10)
    assert budget.reserved_usd == pytest.approx(0.0)


def test_inflight_reservation_rejects_racing_request_before_second_provider_post():
    entered = threading.Event()
    release = threading.Event()
    backend = BlockingBackend(entered, release)
    _runtime({"model-a": backend}, default_budget_usd=0.0035)
    first_status: list[int] = []

    def first_request() -> None:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hold"}]},
        )
        first_status.append(response.status_code)

    worker = threading.Thread(target=first_request)
    worker.start()
    assert entered.wait(timeout=1)
    racing = TestClient(app).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hold"}]},
    )
    release.set()
    worker.join(timeout=2)

    assert first_status == [200]
    assert racing.status_code == 402
    assert backend.calls == 1


@pytest.mark.parametrize("bad", [-1.0, math.inf, -math.inf, math.nan])
def test_budget_rejects_invalid_money(bad: float):
    with pytest.raises(ValueError):
        TenantBudget(bad)
    budget = TenantBudget(1.0)
    with pytest.raises(ValueError):
        budget.charge(bad)
    with pytest.raises(ValueError):
        budget.reserve(bad)


def test_unknown_tenant_is_refused_without_creating_a_wallet():
    backend = FakeBackend()
    value = _runtime({"model-a": backend})

    response = TestClient(app).post(
        "/v1/chat/completions",
        headers={"x-tenant-id": "unconfigured"},
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 403
    assert backend.calls == 0
    assert set(value.budgets) == {"default"}


def test_failed_backends_are_excluded_and_fallback_terminates():
    clients = {
        "model-a": FakeBackend(failure=RuntimeError("a failed")),
        "model-b": FakeBackend(failure=RuntimeError("b failed")),
        "model-c": FakeBackend(failure=RuntimeError("c failed")),
    }
    value = _runtime(clients)

    response = TestClient(app).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]}
    )

    assert response.status_code == 503
    assert [client.calls for client in clients.values()] == [1, 1, 1]
    assert all(not breaker.is_open for breaker in value.breakers.values())
    assert value.budgets["default"].spent_usd > 0


def test_provider_4xx_has_no_fallback_and_does_not_poison_circuit():
    request = httpx.Request("POST", "https://provider.invalid/chat/completions")
    provider_response = httpx.Response(400, request=request)
    error = httpx.HTTPStatusError("bad request", request=request, response=provider_response)
    clients = {
        "model-a": FakeBackend(failure=error),
        "model-b": FakeBackend(),
    }
    value = _runtime(clients)

    response = TestClient(app).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "bad"}]}
    )

    assert response.status_code == 400
    assert clients["model-a"].calls == 1
    assert clients["model-b"].calls == 0
    assert value.breakers["model-a"].failures == 0


def test_cache_is_isolated_by_tenant_and_router_policy_version():
    backend = FakeBackend()
    value = _runtime(
        {"model-a": backend},
        budgets={"tenant-a": TenantBudget(1), "tenant-b": TenantBudget(1)},
    )
    client = TestClient(app)
    body = {"messages": [{"role": "user", "content": "same"}]}

    first_a = client.post("/v1/chat/completions", headers={"x-tenant-id": "tenant-a"}, json=body)
    first_b = client.post("/v1/chat/completions", headers={"x-tenant-id": "tenant-b"}, json=body)
    hit_a = client.post("/v1/chat/completions", headers={"x-tenant-id": "tenant-a"}, json=body)
    value.router.apply_traffic_weights({"model-a": 0.5})
    after_rollout = client.post("/v1/chat/completions", headers={"x-tenant-id": "tenant-a"}, json=body)

    assert [first_a.status_code, first_b.status_code, hit_a.status_code, after_rollout.status_code] == [200] * 4
    assert backend.calls == 3
    assert first_a.headers["x-routebench-cache"] == "miss"
    assert first_b.headers["x-routebench-cache"] == "miss"
    assert hit_a.headers["x-routebench-cache"] == "hit"
    assert after_rollout.headers["x-routebench-cache"] == "miss"
    assert all(
        response.headers["x-routebench-backend"] == "model-a"
        for response in (first_a, first_b, hit_a, after_rollout)
    )


def test_extra_openai_fields_and_tool_history_reach_backend_unchanged():
    backend = FakeBackend()
    _runtime({"model-a": backend})
    tool_calls = [{
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{\"id\":1}"},
    }]
    body = {
        "messages": [
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
            {"role": "tool", "content": "result", "tool_call_id": "call-1"},
        ],
        "max_completion_tokens": 77,
        "response_format": {"type": "json_object"},
        "seed": 9,
    }

    response = TestClient(app).post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    payload = backend.payloads[0]
    assert payload["messages"] == body["messages"]
    assert payload["max_completion_tokens"] == 77
    assert "max_tokens" not in payload
    assert payload["response_format"] == body["response_format"]
    assert payload["seed"] == 9
    assert payload["stream"] is False


def test_non_text_message_content_is_rejected_before_provider_call():
    backend = FakeBackend()
    _runtime({"model-a": backend})

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    )

    assert response.status_code == 422
    assert backend.calls == 0


@pytest.mark.parametrize(
    "paid_feature",
    [
        {"n": 2},
        {"best_of": 2},
        {"modalities": ["text", "audio"]},
        {"audio": {"voice": "alloy", "format": "wav"}},
        {"web_search_options": {}},
    ],
)
def test_unbounded_paid_features_are_rejected_before_provider_call(
    paid_feature: dict[str, Any],
):
    backend = FakeBackend()
    _runtime({"model-a": backend})
    body = {"messages": [{"role": "user", "content": "hello"}], **paid_feature}

    response = TestClient(app).post("/v1/chat/completions", json=body)

    assert response.status_code == 422
    assert backend.calls == 0


def test_buffered_sse_forces_upstream_nonstream_and_holds_admission_until_close():
    backend = FakeBackend(
        tool_calls=[ToolCall(id="call-1", name="lookup", arguments="{}")]
    )
    value = _runtime({"model-a": backend})
    value.admission = AdmissionController(1, queue_timeout_s=0.001)
    response = chat_completions(
        ChatRequest(messages=[{"role": "user", "content": "hello"}], stream=True),
        "default",
    )

    assert isinstance(response, StreamingResponse)
    assert backend.payloads[0]["stream"] is False
    assert backend.payloads[0]["max_tokens"] == 2048
    assert not value.admission.acquire()

    async def consume() -> bytes:
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.encode() if isinstance(chunk, str) else chunk)
        return b"".join(chunks)

    content = anyio.run(consume)
    assert b'"finish_reason": "tool_calls"' in content
    assert content.endswith(b"data: [DONE]\n\n")
    assert value.admission.acquire()
    value.admission.release()
def test_missing_provider_usage_charges_bound_and_does_not_cache():
    class MissingUsage:
        name = "missing-usage"

        def chat(self, payload):
            return ChatResult(text="answer", usage=Usage(), model=payload["model"],
                              provider=self.name, cost_usd=0, latency_ms=1,
                              raw={"model": payload["model"], "choices": []})

    backend = Backend("missing", "missing-usage", 1, 1, 1)
    value = build_runtime([backend], {"missing": MissingUsage()}, {"missing": {"chat": 1}})
    configure_runtime(value)
    client = TestClient(app)
    body = {"messages": [{"role": "user", "content": "hello"}]}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200 and response.json()["usage"] is None
    assert value.budgets["default"].spent_usd > 0
    second = client.post("/v1/chat/completions", json=body)
    assert second.headers["x-routebench-cache"] == "miss"


@pytest.mark.parametrize("extra", [{"service_tier": "priority"}, {"prediction": {"content": "hello"}},
                                  {"max_tokens": True}, {"tools": [{"type": "web_search"}]}])
def test_unbounded_or_unknown_paid_options_never_call_backend(extra):
    backend = FakeBackend()
    _runtime({"model-a": backend})
    response = TestClient(app).post("/v1/chat/completions",
                                   json={"messages": [{"role": "user", "content": "hello"}], **extra})
    assert response.status_code == 422 and backend.calls == 0


def test_real_raw_usage_uses_admission_price_schedule_not_adapter_estimate():
    from gateway.app import _result_cost

    result = ChatResult(text="ok", usage=Usage(prompt_tokens=10, completion_tokens=5),
                        model="model", provider="mock", latency_ms=1, cost_usd=0,
                        raw={"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    assert _result_cost(result, Backend("model", "mock", 2, 4, 1)) == pytest.approx(0.00004)


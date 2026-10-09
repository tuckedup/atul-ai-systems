"""Native token streaming: incremental delivery, spend safety and failure handling. Fake streaming backends and httpx.MockTransport only; no provider call."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from gateway import streaming
from gateway.app import app, build_runtime, configure_runtime
from gateway.backends.base import OpenAICompatibleBackend
from gateway.reliability import TenantBudget
from gateway.router import Backend


def chunk(content: str | None = None, *, finish: str | None = None, role: bool = False) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = "assistant"
    if content is not None:
        delta["content"] = content
    return {"id": "up-1", "object": "chat.completion.chunk", "created": 1, "model": "model-a", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def usage_chunk(prompt=10, completion=5) -> dict[str, Any]:
    return {"id": "up-1", "object": "chat.completion.chunk", "created": 1, "model": "model-a", "choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}


class StreamBackend:
    name = "fake-stream"

    def __init__(self, chunks=None, *, fail_before=None, fail_after=None, hang=None):
        self.chunks = list(chunks if chunks is not None else [chunk(role=True), chunk("Hel"), chunk("lo"), chunk(finish="stop"), usage_chunk()])
        self.fail_before, self.fail_after, self.payloads, self.closed = fail_before, fail_after, [], False

    def chat(self, payload):  # pragma: no cover - the native path never calls the buffered method
        raise AssertionError("buffered chat must not be used for a native stream")

    def stream_chat(self, payload):
        self.payloads.append(payload)
        try:
            if self.fail_before:
                raise self.fail_before
            for i, c in enumerate(self.chunks):
                if self.fail_after is not None and i == self.fail_after:
                    raise RuntimeError("connection reset")
                yield c
        finally:
            self.closed = True


def setup(backends: dict[str, StreamBackend], *, budget=1.0, prices=(1.0, 2.0), native=True):
    metas = [Backend(m, "fake", prices[0], prices[1], i + 1) for i, m in enumerate(backends)]
    quality = {m: {"chat": 1.0 - i * 0.1} for i, m in enumerate(backends)}
    value = build_runtime(metas, backends, quality, budgets={"default": TenantBudget(budget)}, default_budget_usd=0.0)
    value.native_streaming = native
    configure_runtime(value)
    return value


def request(extra=None):
    return {"messages": [{"role": "user", "content": "hello"}], "stream": True, **(extra or {})}


def events(response) -> list[str]:
    return [e[len("data: "):] for e in response.text.split("\n\n") if e.startswith("data: ")]


def test_chunks_are_delivered_incrementally_and_priced_from_final_usage():
    b = StreamBackend()
    value = setup({"model-a": b})
    with TestClient(app).stream("POST", "/v1/chat/completions", json=request()) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(r.iter_text())
    ev = [e[len("data: "):] for e in raw.split("\n\n") if e.startswith("data: ")]
    parsed = [json.loads(e) for e in ev[:-1]]
    assert ev[-1] == "[DONE]"
    assert [c["choices"][0]["delta"].get("content") for c in parsed if c["choices"]] == [None, "Hel", "lo", None]  # separate deltas, not one buffered blob
    assert all(c["id"].startswith("chatcmpl-") for c in parsed) and len({c["id"] for c in parsed}) == 1
    assert all("usage" not in c for c in parsed)  # the client did not ask for usage, so the accounting chunk is not forwarded
    budget = value.budgets["default"]
    assert budget.spent_usd == pytest.approx((10 * 1.0 + 5 * 2.0) / 1e6) and budget.reserved_usd == pytest.approx(0)
    assert value.router.usage.snapshot()["total_cost_usd"] == pytest.approx(20 / 1e6) and b.closed


def test_upstream_is_asked_to_stream_with_usage_and_the_client_gets_usage_only_if_it_asked():
    b = StreamBackend()
    setup({"model-a": b})
    r = TestClient(app).post("/v1/chat/completions", json=request({"stream_options": {"include_usage": True}}))
    assert b.payloads[0]["stream"] is True and b.payloads[0]["stream_options"]["include_usage"] is True
    last = json.loads(events(r)[-2])
    assert last["choices"] == [] and last["usage"]["completion_tokens"] == 5
    b2 = StreamBackend()
    setup({"model-a": b2})
    TestClient(app).post("/v1/chat/completions", json=request())
    assert b2.payloads[0]["stream_options"] == {"include_usage": True}  # always requested upstream for accounting


def test_a_failure_before_the_first_chunk_reroutes_to_the_next_backend_and_charges_the_bound():
    bad, good = StreamBackend(fail_before=RuntimeError("boom")), StreamBackend()
    value = setup({"model-a": bad, "model-b": good})
    r = TestClient(app).post("/v1/chat/completions", json=request())
    assert r.status_code == 200 and r.headers["x-routebench-backend"] == "model-b" and events(r)[-1] == "[DONE]"
    assert len(good.payloads) == 1 and value.budgets["default"].spent_usd > (10 * 1.0 + 5 * 2.0) / 1e6  # the failed attempt consumed its conservative bound
    assert value.budgets["default"].reserved_usd == pytest.approx(0)


def test_a_provider_4xx_before_the_first_chunk_is_not_rerouted():
    err = httpx.HTTPStatusError("bad", request=httpx.Request("POST", "http://x"), response=httpx.Response(400))
    bad, good = StreamBackend(fail_before=err), StreamBackend()
    setup({"model-a": bad, "model-b": good})
    r = TestClient(app).post("/v1/chat/completions", json=request())
    assert r.status_code == 400 and not good.payloads


def test_a_midstream_failure_ends_with_an_error_event_charges_the_bound_and_does_not_reroute():
    broken, spare = StreamBackend(fail_after=2), StreamBackend()
    value = setup({"model-a": broken, "model-b": spare})
    r = TestClient(app).post("/v1/chat/completions", json=request())
    ev = events(r)
    assert r.status_code == 200 and json.loads(ev[-2])["error"]["type"] == "upstream_error" and ev[-1] == "[DONE]" and not spare.payloads
    b = value.budgets["default"]
    assert b.reserved_usd == pytest.approx(0) and b.spent_usd > (10 * 1.0 + 5 * 2.0) / 1e6  # bound, not zero
    assert "connection reset" not in r.text  # upstream error detail is not leaked to the client
    assert value.breakers["model-a"].failures == 1  # the mid-stream failure counted against the backend


def test_a_stream_without_final_usage_is_charged_at_the_bound_and_marked_unknown():
    b = StreamBackend([chunk(role=True), chunk("x"), chunk(finish="stop")])
    value = setup({"model-a": b})
    TestClient(app).post("/v1/chat/completions", json=request())
    assert value.budgets["default"].spent_usd > 20 / 1e6
    assert value.router.usage.snapshot()["total_cost_usd"] > 20 / 1e6


def test_abandoning_the_stream_settles_the_reservation_at_the_bound_and_releases_admission_exactly_once():
    from gateway.app import ChatRequest

    b = StreamBackend([chunk(role=True)] + [chunk("x")] * 50 + [usage_chunk()])
    value = setup({"model-a": b})
    budget = value.budgets["default"]
    opened = streaming.open_stream(ChatRequest(**request()), "default", budget, "chatcmpl-test")
    assert budget.reserved_usd > 0  # funds are held while the stream is open
    released = []
    events_iter = opened.events(lambda: released.append(1))
    assert next(events_iter).startswith("data: ")  # the client reads one event ...
    events_iter.close()  # ... then disconnects
    assert released == [1] and b.closed  # admission released once, upstream connection closed
    assert budget.reserved_usd == pytest.approx(0) and budget.spent_usd > (10 * 1.0 + 5 * 2.0) / 1e6  # the bound was charged: the real cost is unknown
    events_iter.close()
    assert released == [1]  # closing again releases nothing more
    assert value.router.usage.snapshot()["total_cost_usd"] == pytest.approx(budget.spent_usd)  # recorded as an unknown-bound attempt


def test_a_stream_that_is_fully_consumed_releases_admission_once_and_prices_the_usage():
    from gateway.app import ChatRequest

    b = StreamBackend()
    value = setup({"model-a": b})
    budget = value.budgets["default"]
    opened = streaming.open_stream(ChatRequest(**request()), "default", budget, "chatcmpl-test")
    released = []
    out = list(opened.events(lambda: released.append(1)))
    assert out[-1] == "data: [DONE]\n\n" and released == [1]
    assert budget.spent_usd == pytest.approx((10 * 1.0 + 5 * 2.0) / 1e6) and budget.reserved_usd == pytest.approx(0)


def test_an_overshoot_in_final_usage_is_recorded_not_discarded():
    b = StreamBackend([chunk(role=True), chunk("x"), chunk(finish="stop"), usage_chunk(10, 10_000_000)])
    value = setup({"model-a": b}, budget=100.0)
    TestClient(app).post("/v1/chat/completions", json=request({"max_tokens": 5}))
    budget = value.budgets["default"]
    assert budget.reservation_breaches == 1 and budget.spent_usd == pytest.approx((10 * 1.0 + 10_000_000 * 2.0) / 1e6)


def test_budget_exhaustion_refuses_before_any_upstream_call():
    b = StreamBackend()
    setup({"model-a": b}, budget=0.0)
    r = TestClient(app).post("/v1/chat/completions", json=request())
    assert r.status_code == 402 and not b.payloads


def test_streamed_responses_are_never_cached_and_ttft_is_exported():
    b = StreamBackend()
    value = setup({"model-a": b})
    for _ in range(2):
        r = TestClient(app).post("/v1/chat/completions", json=request())
        assert r.headers["x-routebench-cache"] == "miss"
    assert len(b.payloads) == 2 and value.budgets["default"].spent_usd == pytest.approx(2 * 20 / 1e6)
    metrics = TestClient(app).get("/metrics").text
    assert 'routebench_ttft_seconds_count{backend="model-a"}' in metrics


def test_without_the_flag_or_a_streaming_client_the_buffered_path_is_used():
    from test_gateway_spend_safety import FakeBackend

    backend = FakeBackend()
    metas = [Backend("model-a", "fake", 1.0, 1.0, 1)]
    value = build_runtime(metas, {"model-a": backend}, {"model-a": {"chat": 1.0}}, budgets={"default": TenantBudget(1.0)}, default_budget_usd=0.0)
    value.native_streaming = True  # flag on, but the client has no stream_chat: buffered delivery must remain
    configure_runtime(value)
    assert streaming.native_streaming_available() is False
    r = TestClient(app).post("/v1/chat/completions", json=request())
    assert r.status_code == 200 and backend.calls == 1 and events(r)[-1] == "[DONE]"
    b2 = StreamBackend()
    setup({"model-a": b2}, native=False)
    TestClient(app).post("/v1/chat/completions", json=request())
    assert not b2.payloads  # flag off: the streaming client is not used (buffered path would call chat, which this fake forbids)


# ------------------------------------------------------------------------------------------------ the real adapter against a mock transport
def sse(*items: str) -> bytes:
    return "".join(items).encode()


def adapter(handler):
    return OpenAICompatibleBackend("p", "http://upstream.invalid/v1", "sk-test", transport=httpx.MockTransport(handler))


def test_adapter_streams_parsed_chunks_ignores_keepalives_and_stops_at_done():
    body = sse(": keep-alive\n\n", f"data: {json.dumps(chunk('a'))}\n\n", "\n", f"data: {json.dumps(usage_chunk())}\n\n", "data: [DONE]\n\n", f"data: {json.dumps(chunk('after-done'))}\n\n")
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    out = list(adapter(handler).stream_chat({"model": "m", "messages": []}))
    assert [c["choices"][0]["delta"].get("content") if c["choices"] else "usage" for c in out] == ["a", "usage"]  # nothing after [DONE]
    assert seen["body"]["stream"] is True and seen["auth"] == "Bearer sk-test"


def test_adapter_raises_on_http_errors_before_yielding_and_on_malformed_data():
    gen = adapter(lambda req: httpx.Response(503, content=b"unavailable")).stream_chat({"model": "m", "messages": []})
    with pytest.raises(httpx.HTTPStatusError):
        next(gen)
    bad = adapter(lambda req: httpx.Response(200, content=sse("data: {not json\n\n")))
    with pytest.raises(ValueError):
        list(bad.stream_chat({"model": "m", "messages": []}))


def test_gateway_runs_end_to_end_over_the_real_adapter_and_mock_transport():
    body = sse(f"data: {json.dumps(chunk(role=True))}\n\n", f"data: {json.dumps(chunk('Hi'))}\n\n", f"data: {json.dumps(chunk(finish='stop'))}\n\n", f"data: {json.dumps(usage_chunk(12, 3))}\n\n", "data: [DONE]\n\n")
    real = adapter(lambda req: httpx.Response(200, content=body))
    value = setup({"model-a": StreamBackend()})
    value.clients["model-a"] = real
    r = TestClient(app).post("/v1/chat/completions", json=request())
    ev = events(r)
    assert r.status_code == 200 and ev[-1] == "[DONE]" and json.loads(ev[1])["choices"][0]["delta"]["content"] == "Hi"
    assert value.budgets["default"].spent_usd == pytest.approx((12 * 1.0 + 3 * 2.0) / 1e6)

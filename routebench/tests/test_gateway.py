from __future__ import annotations

from collections import deque
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway import router
from gateway.backends import AnthropicAdapter, BackendHTTPError, OpenAIAdapter


class ScriptedTransport:
    """Return or raise queued provider outcomes without performing I/O."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = deque(outcomes)
        self.calls: list[tuple[dict[str, Any], bool]] = []

    def __call__(self, payload: dict[str, Any], stream: bool):
        self.calls.append((payload, stream))
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def client() -> TestClient:
    router.configure_runtime([])
    with TestClient(router.app) as test_client:
        yield test_client


def openai_response(text: str = "hello") -> dict[str, Any]:
    return {
        "id": "upstream-id",
        "model": "openai-mock-1",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5},
    }


def anthropic_response(text: str = "fallback") -> dict[str, Any]:
    return {
        "id": "msg_mock",
        "model": "anthropic-mock-1",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 12, "output_tokens": 4},
    }


def test_gateway_returns_openai_compatible_shape(client: TestClient) -> None:
    transport = ScriptedTransport(openai_response())
    router.configure_runtime(
        [
            OpenAIAdapter(
                "openai-mock-1",
                transport,
                input_per_1m=1.0,
                output_per_1m=2.0,
            )
        ]
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert body["model"] == "openai-mock-1"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ]
    assert body["usage"] == {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
    }
    assert response.headers["x-routebench-provider"] == "openai"
    assert len(response.headers["x-routebench-trace-id"]) == 32
    assert transport.calls[0][0]["model"] == "openai-mock-1"


def test_gateway_falls_back_to_second_provider_on_5xx(client: TestClient) -> None:
    primary = ScriptedTransport(BackendHTTPError(503, "primary unavailable"))
    fallback = ScriptedTransport(anthropic_response())
    router.configure_runtime(
        [
            OpenAIAdapter(
                "openai-mock-1",
                primary,
                input_per_1m=1.0,
                output_per_1m=2.0,
                max_retries=0,
            ),
            AnthropicAdapter(
                "anthropic-mock-1",
                fallback,
                input_per_1m=3.0,
                output_per_1m=6.0,
                max_retries=0,
            ),
        ]
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "fallback"
    assert response.headers["x-routebench-provider"] == "anthropic"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1
    request_id = response.headers["x-routebench-request-id"]
    assert client.get(f"/internal/costs/{request_id}").json()["fallbacks"] == 1


def test_cost_is_tracked_per_request(client: TestClient) -> None:
    transport = ScriptedTransport(openai_response("first"), openai_response("second"))
    router.configure_runtime(
        [
            OpenAIAdapter(
                "openai-mock-1",
                transport,
                input_per_1m=2.0,
                output_per_1m=8.0,
            )
        ]
    )

    first = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "one"}]},
    )
    second = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "two"}]},
    )

    first_id = first.headers["x-routebench-request-id"]
    second_id = second.headers["x-routebench-request-id"]
    assert first_id != second_id
    expected_cost = (20 * 2.0 + 5 * 8.0) / 1_000_000
    first_cost = client.get(f"/internal/costs/{first_id}").json()
    second_cost = client.get(f"/internal/costs/{second_id}").json()
    assert first_cost["cost_usd"] == pytest.approx(expected_cost)
    assert second_cost["cost_usd"] == pytest.approx(expected_cost)
    assert first_cost["trace_id"] == first.headers["x-routebench-trace-id"]
    assert float(first.headers["x-routebench-cost-usd"]) == pytest.approx(expected_cost)


def test_gateway_streams_openai_compatible_sse(client: TestClient) -> None:
    stream = [
        {
            "type": "message_start",
            "message": {
                "model": "anthropic-mock-1",
                "usage": {"input_tokens": 7, "output_tokens": 0},
            },
        },
        {"type": "content_block_delta", "delta": {"text": "hel"}},
        {"type": "content_block_delta", "delta": {"text": "lo"}},
        {"type": "message_delta", "usage": {"output_tokens": 2}},
    ]
    router.configure_runtime(
        [
            AnthropicAdapter(
                "anthropic-mock-1",
                ScriptedTransport(stream),
                input_per_1m=3.0,
                output_per_1m=6.0,
            )
        ]
    )

    response = client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object":"chat.completion.chunk"' in response.text
    assert '"content":"hel"' in response.text
    assert '"content":"lo"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")

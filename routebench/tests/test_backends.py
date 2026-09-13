from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from gateway.backends import AnthropicAdapter, BackendHTTPError, OpenAIAdapter


class ScriptedTransport:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = deque(outcomes)
        self.calls = 0
        self.last_payload: dict[str, Any] | None = None
        self.last_stream = False

    def __call__(self, payload: dict[str, Any], stream: bool):
        self.calls += 1
        self.last_payload = payload
        self.last_stream = stream
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_openai_adapter_retries_transient_failure_and_estimates_cost() -> None:
    transport = ScriptedTransport(
        BackendHTTPError(500, "retry me"),
        {
            "model": "openai-mock",
            "choices": [{"message": {"content": "recovered"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 25},
        },
    )
    adapter = OpenAIAdapter(
        "openai-mock",
        transport,
        input_per_1m=2.0,
        output_per_1m=10.0,
        max_retries=1,
    )

    result = adapter.chat({"messages": [{"role": "user", "content": "Hi"}]})

    assert transport.calls == 2
    assert result.attempts == 2
    assert result.text == "recovered"
    assert result.cost_usd == pytest.approx((100 * 2.0 + 25 * 10.0) / 1_000_000)


def test_anthropic_adapter_handles_native_streaming_shape() -> None:
    events = [
        {
            "type": "message_start",
            "message": {
                "model": "claude-mock",
                "usage": {"input_tokens": 30, "output_tokens": 0},
            },
        },
        {"type": "content_block_delta", "delta": {"text": "route"}},
        {"type": "content_block_delta", "delta": {"text": "bench"}},
        {"type": "message_delta", "usage": {"output_tokens": 4}},
    ]
    transport = ScriptedTransport(events)
    adapter = AnthropicAdapter(
        "claude-mock",
        transport,
        input_per_1m=3.0,
        output_per_1m=15.0,
    )

    result = adapter.chat(
        {"messages": [{"role": "user", "content": "Hi"}]}, stream=True
    )

    assert transport.calls == 1
    assert transport.last_stream is True
    assert result.text == "routebench"
    assert result.chunks == ("route", "bench")
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 4
    assert result.cost_usd == pytest.approx((30 * 3.0 + 4 * 15.0) / 1_000_000)


def test_adapter_does_not_retry_non_transient_4xx() -> None:
    transport = ScriptedTransport(BackendHTTPError(400, "invalid request"))
    adapter = OpenAIAdapter(
        "openai-mock",
        transport,
        input_per_1m=1.0,
        output_per_1m=1.0,
        max_retries=3,
    )

    with pytest.raises(BackendHTTPError, match="invalid request"):
        adapter.chat({"messages": []})

    assert transport.calls == 1

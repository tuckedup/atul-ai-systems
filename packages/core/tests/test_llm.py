"""Tests for aisys.llm — uses respx to mock HTTP, one live test gated by env."""
import os

import httpx
import pytest
import respx

from aisys.llm import ChatResult, Usage, chat, estimate_cost, ProviderError


MOCK_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello, world!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}

MOCK_TOOL_RESPONSE = {
    "id": "chatcmpl-test456",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 15},
}


@respx.mock
def test_chat_basic():
    """Test basic chat returns correct ChatResult fields."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=MOCK_RESPONSE)
    )
    result = chat([{"role": "user", "content": "Hi"}])
    assert isinstance(result, ChatResult)
    assert result.text == "Hello, world!"
    assert result.model == "gpt-4o-mini"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.latency_ms >= 0
    assert result.cost_usd is not None
    assert result.cost_usd > 0


@respx.mock
def test_chat_tool_calls():
    """Test chat with tool calls parses correctly."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=MOCK_TOOL_RESPONSE)
    )
    result = chat(
        [{"role": "user", "content": "What's the weather?"}],
        tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
    )
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == '{"location": "NYC"}'


@respx.mock
def test_chat_retry_on_5xx():
    """Test retry logic on 5xx errors."""
    route = respx.post("https://api.openai.com/v1/chat/completions")
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(500),
        httpx.Response(200, json=MOCK_RESPONSE),
    ]
    result = chat([{"role": "user", "content": "Hi"}])
    assert result.text == "Hello, world!"
    assert route.call_count == 3


@respx.mock
def test_chat_retry_on_429():
    """Test retry logic on 429 rate limit."""
    route = respx.post("https://api.openai.com/v1/chat/completions")
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(200, json=MOCK_RESPONSE),
    ]
    result = chat([{"role": "user", "content": "Hi"}])
    assert result.text == "Hello, world!"
    assert route.call_count == 2


@respx.mock
def test_chat_fallback_models():
    """Test model fallback when first model fails."""
    route = respx.post("https://api.openai.com/v1/chat/completions")
    route.side_effect = [
        httpx.Response(500),  # First model fails
        httpx.Response(200, json=MOCK_RESPONSE),  # Fallback succeeds
    ]
    result = chat(
        [{"role": "user", "content": "Hi"}],
        model="gpt-4o",
        fallback=["gpt-4o-mini"],
    )
    assert result.text == "Hello, world!"


@respx.mock
def test_chat_exhausted_retries():
    """Test ProviderError when all retries exhausted."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(500)
    )
    with pytest.raises(ProviderError):
        chat([{"role": "user", "content": "Hi"}])


def test_cost_estimation():
    """Test cost estimation for known and unknown models."""
    usage = Usage(prompt_tokens=1000, completion_tokens=500)
    # gpt-4o-mini: input 0.15/1M, output 0.60/1M
    cost = estimate_cost("gpt-4o-mini", usage)
    assert cost is not None
    expected = (1000 * 0.15 + 500 * 0.60) / 1e6
    assert abs(cost - expected) < 1e-10

    # Unknown model returns None
    unknown_cost = estimate_cost("unknown-model", usage)
    assert unknown_cost is None


def test_unknown_model_no_crash():
    """Test that unknown model doesn't crash, just returns None cost."""
    usage = Usage(prompt_tokens=100, completion_tokens=50)
    cost = estimate_cost("nonexistent-model-xyz", usage)
    assert cost is None


# Live test — only runs if OPENAI_API_KEY is set
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live test",
)
def test_chat_live():
    """Live test against real OpenAI API. Gated by env var."""
    result = chat(
        [{"role": "user", "content": "Say exactly: live test passed"}],
        model="gpt-4o-mini",
        max_tokens=20,
    )
    assert "live test passed" in result.text.lower()
    assert result.usage.total > 0

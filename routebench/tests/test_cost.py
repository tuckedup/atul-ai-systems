from __future__ import annotations

from collections import deque
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway import router
from gateway.backends import OpenAIAdapter


class UsageTransport:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = deque(responses)

    def __call__(self, payload: dict[str, Any], stream: bool):
        return self.responses.popleft()


def response(
    model: str, prompt_tokens: int, completion_tokens: int
) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def test_usage_endpoint_aggregates_tokens_and_cost_by_model() -> None:
    adapter = OpenAIAdapter(
        "configured-mock",
        UsageTransport(
            response("model-a", 100, 20),
            response("model-a", 50, 10),
            response("model-b", 10, 5),
        ),
        input_per_1m=2.0,
        output_per_1m=10.0,
    )
    router.configure_runtime([adapter])

    with TestClient(router.app) as client:
        for content in ("one", "two", "three"):
            result = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": content}]},
            )
            assert result.status_code == 200
        usage = client.get("/v1/usage")

    assert usage.status_code == 200
    body = usage.json()
    by_model = {item["model"]: item for item in body["data"]}
    assert by_model["model-a"] == {
        "model": "model-a",
        "requests": 2,
        "prompt_tokens": 150,
        "completion_tokens": 30,
        "total_tokens": 180,
        "cost_usd": pytest.approx((150 * 2.0 + 30 * 10.0) / 1_000_000),
    }
    assert by_model["model-b"]["requests"] == 1
    assert by_model["model-b"]["total_tokens"] == 15
    assert body["total"]["requests"] == 3
    assert body["total"]["prompt_tokens"] == 160
    assert body["total"]["completion_tokens"] == 35
    assert body["total"]["total_tokens"] == 195
    assert body["total"]["cost_usd"] == pytest.approx(
        (160 * 2.0 + 35 * 10.0) / 1_000_000
    )

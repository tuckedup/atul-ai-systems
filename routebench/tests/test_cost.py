from typing import Any

from aisys.llm import ChatResult, Usage
from fastapi.testclient import TestClient
from gateway.app import app, build_runtime, configure_runtime
from gateway.router import Backend


class CostBackend:
    name = "cost"

    def chat(self, payload: dict[str, Any]) -> ChatResult:
        return ChatResult(
            text="metered",
            usage=Usage(prompt_tokens=100, completion_tokens=25),
            model=payload["model"],
            provider=self.name,
            latency_ms=1,
            cost_usd=0.0025,
        )


def test_usage_accumulates_tokens_cost_and_model_breakdown() -> None:
    backend = Backend("metered-model", "cost", 10, 20, 1)
    configure_runtime(build_runtime([backend], {backend.model: CostBackend()}, {backend.model: {"chat": 1.0}}))
    client = TestClient(app)

    for prompt in ("first request", "first request", "second request"):
        response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": prompt}]})
        assert response.status_code == 200

    usage = client.get("/v1/usage")
    assert usage.status_code == 200
    assert usage.json() == {
        "total_tokens": 375,
        "total_cost_usd": 0.005,
        "by_model": {
            "metered-model": {
                "input_tokens": 300,
                "output_tokens": 75,
                "total_tokens": 375,
                "cost_usd": 0.005,
            }
        },
    }

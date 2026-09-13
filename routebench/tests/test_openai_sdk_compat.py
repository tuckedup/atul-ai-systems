from typing import Any

from aisys.llm import ChatResult, Usage
from fastapi.testclient import TestClient
from gateway.app import app, build_runtime, configure_runtime
from gateway.router import Backend


class StubBackend:
    name = "stub"

    def chat(self, payload: dict[str, Any]) -> ChatResult:
        return ChatResult(
            text="hello", usage=Usage(prompt_tokens=3, completion_tokens=1), model=payload["model"],
            provider=self.name, latency_ms=1, cost_usd=0.001,
        )


def setup_function() -> None:
    metadata = Backend("model-a", "stub", 1, 1, 10)
    configure_runtime(build_runtime([metadata], {"model-a": StubBackend()}, {"model-a": {"chat": 0.9}}))


def test_chat_completions_matches_openai_wire_shape():
    response = TestClient(app).post("/v1/chat/completions", json={
        "model": "ignored", "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "hello"
    assert payload["usage"]["total_tokens"] == 4
    assert response.headers["x-routebench-backend"] == "model-a"


def test_streaming_uses_openai_sse_protocol():
    response = TestClient(app).post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}], "stream": True,
    })
    assert response.status_code == 200
    assert "chat.completion.chunk" in response.text and "data: [DONE]" in response.text


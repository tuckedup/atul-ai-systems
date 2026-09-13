from typing import Any

from aisys.llm import ChatResult, Usage
from fastapi.testclient import TestClient
from gateway.app import app, build_runtime, configure_runtime
from gateway.router import Backend


class DeadBackend:
    name = "dead"

    def chat(self, payload: dict[str, Any]) -> ChatResult:
        raise ConnectionError("backend terminated")


class HealthyBackend:
    name = "healthy"

    def chat(self, payload: dict[str, Any]) -> ChatResult:
        return ChatResult(text="ok", usage=Usage(), model=payload["model"], provider=self.name, latency_ms=1, cost_usd=0)


def test_backend_failure_reroutes_with_under_one_percent_failures():
    backends = [Backend("dead", "dead", 0, 0, 1), Backend("healthy", "healthy", 0, 0, 10)]
    quality = {"dead": {"chat": 1.0}, "healthy": {"chat": 0.8}}
    configure_runtime(build_runtime(backends, {"dead": DeadBackend(), "healthy": HealthyBackend()}, quality))
    client = TestClient(app)
    failures = sum(
        client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": str(i)}]}).status_code != 200
        for i in range(100)
    )
    assert failures < 1


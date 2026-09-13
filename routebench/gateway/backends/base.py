"""Backend protocol and OpenAI-wire implementation."""
from __future__ import annotations

import time
from typing import Any, Protocol

import httpx
from aisys.audit import default_audit_log
from aisys.llm import ChatResult, ToolCall, Usage, estimate_cost
from aisys.tracing import current_trace_id, traced


class BackendClient(Protocol):
    name: str

    def chat(self, payload: dict[str, Any]) -> ChatResult: ...


class OpenAICompatibleBackend:
    def __init__(self, name: str, base_url: str, api_key: str = "", transport: httpx.BaseTransport | None = None):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.transport = transport

    @traced(kind="llm")
    def chat(self, payload: dict[str, Any]) -> ChatResult:
        started = time.perf_counter()
        with httpx.Client(transport=self.transport, timeout=60) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        usage = Usage(**data.get("usage", {}))
        result = ChatResult(
            text=message.get("content") or "",
            tool_calls=[
                ToolCall(id=call["id"], name=call["function"]["name"], arguments=call["function"]["arguments"])
                for call in message.get("tool_calls", []) or []
            ],
            usage=usage,
            model=data.get("model", payload["model"]),
            provider=self.name,
            latency_ms=(time.perf_counter() - started) * 1_000,
            cost_usd=estimate_cost(payload["model"], usage),
            raw=data,
        )
        default_audit_log().append({
            "type": "model_call", "trace_id": current_trace_id.get(), "provider": self.name,
            "model": result.model, "tokens": usage.total, "cost_usd": result.cost_usd,
        })
        return result


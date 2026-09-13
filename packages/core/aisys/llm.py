"""Provider-agnostic chat client speaking the OpenAI wire format.

Point OPENAI_BASE_URL at RouteBench (http://localhost:8080/v1) and every project routes through the
gateway with zero code change. Every call returns tokens, cost, latency, and is traced.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Generator

import httpx
import yaml
from pydantic import BaseModel

from .settings import settings
from .tracing import traced

_PRICING: dict[str, dict[str, float]] = yaml.safe_load(
    (Path(__file__).parent / "pricing.yaml").read_text()
)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # raw JSON string, caller parses/validates via aisys.tools


class ChatResult(BaseModel):
    text: str
    tool_calls: list[ToolCall] = []
    usage: Usage
    model: str
    provider: str
    latency_ms: float
    cost_usd: float | None
    raw: dict[str, Any] = {}


def estimate_cost(model: str, usage: Usage) -> float | None:
    p = _PRICING.get(model)
    if not p:
        return None
    return (usage.prompt_tokens * p["input_per_1m"] + usage.completion_tokens * p["output_per_1m"]) / 1e6


class ProviderError(RuntimeError):
    pass


class StreamChunk(BaseModel):
    """One incremental chunk from chat_stream. text is the delta for this chunk."""
    text: str = ""
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    cost_usd: float | None = None


@traced(kind="llm")
def chat(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    fallback: list[str] | None = None,
    **extra: Any,
) -> ChatResult:
    """Synchronous chat with retries (429/5xx), timeout, and ordered model fallback."""
    candidates = [model or settings.default_model] + list(fallback or settings.fallback_models)
    last_err: Exception | None = None
    for m in candidates:
        for attempt in range(settings.max_retries):
            try:
                return _call(m, messages, tools, temperature, max_tokens, extra)
            except ProviderError as e:
                last_err = e
                time.sleep(min(8.0, (2**attempt) * 0.5 + random.random() * 0.25))
    raise ProviderError(f"all candidates failed: {candidates}") from last_err


def _call(model: str, messages, tools, temperature, max_tokens, extra) -> ChatResult:
    body: dict[str, Any] = {
        "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **extra
    }
    if tools:
        body["tools"] = tools
    t0 = time.perf_counter()
    with httpx.Client(timeout=settings.request_timeout_s) as client:
        r = client.post(
            f"{settings.openai_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json=body,
        )
    if r.status_code == 429 or r.status_code >= 500:
        raise ProviderError(f"{model}: HTTP {r.status_code}")
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    usage = Usage(**data.get("usage", {}))
    return ChatResult(
        text=msg.get("content") or "",
        tool_calls=[
            ToolCall(id=t["id"], name=t["function"]["name"], arguments=t["function"]["arguments"])
            for t in msg.get("tool_calls", []) or []
        ],
        usage=usage,
        model=data.get("model", model),
        provider=r.headers.get("x-routebench-backend", "direct"),
        latency_ms=(time.perf_counter() - t0) * 1000,
        cost_usd=estimate_cost(model, usage),
        raw=data,
    )


@traced(kind="llm")
def chat_stream(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    fallback: list[str] | None = None,
    **extra: Any,
) -> Generator[StreamChunk, None, None]:
    """Streaming variant of chat. Yields StreamChunk per SSE fragment.

    Uses the same retry (429/5xx, jittered backoff), timeout, and ordered model
    fallback as `chat()`. The final yielded chunk carries full usage + cost.
    """
    candidates = [model or settings.default_model] + list(fallback or settings.fallback_models)
    last_err: Exception | None = None
    for m in candidates:
        for attempt in range(settings.max_retries):
            try:
                yield from _stream_one(m, messages, tools, temperature, max_tokens, extra)
                return
            except ProviderError as e:
                last_err = e
                time.sleep(min(8.0, (2**attempt) * 0.5 + random.random() * 0.25))
    raise ProviderError(f"all candidates failed: {candidates}") from last_err


def _stream_one(model: str, messages, tools, temperature, max_tokens, extra):
    body: dict[str, Any] = dict(model=model, messages=messages, temperature=temperature,
                                 max_tokens=max_tokens, stream=True, **extra)
    if tools:
        body["tools"] = tools
    t0 = time.perf_counter()
    prov = "direct"
    model_label = model
    prompt_tokens = 0
    completion_tokens = 0
    with httpx.Client(timeout=settings.request_timeout_s) as client:
        with client.stream(
            "POST",
            f"{settings.openai_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json=body,
        ) as r:
            prov = r.headers.get("x-routebench-backend", "direct")
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.strip():
                    continue
                txt = line if isinstance(line, str) else line.decode("utf-8", "replace")
                if not txt.startswith("data:"):
                    continue
                payload = txt[5:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                model_label = data.get("model", model_label)
                delta = data.get("choices", [{}])[0].get("delta", {})
                delta_text = delta.get("content") or ""
                usage = data.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens", 0))
                    completion_tokens = int(usage.get("completion_tokens", 0))
                yield StreamChunk(
                    text=delta_text,
                    model=model_label,
                    provider=prov,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    cost_usd=estimate_cost(model_label, Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)) if usage else None,
                )
            # emit a final aggregate chunk if the server never sent a usage frame
            if prompt_tokens == 0 and completion_tokens == 0:
                yield StreamChunk(text="", model=model_label, provider=prov,
                                  latency_ms=(time.perf_counter() - t0) * 1000, cost_usd=None)

"""Provider-agnostic chat client speaking the OpenAI wire format.

Point OPENAI_BASE_URL at RouteBench (http://localhost:8080/v1) and every project routes through the
gateway with zero code change. Every call returns tokens, cost, latency, and is traced.
"""
from __future__ import annotations

import random
import time
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

import httpx
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from .audit import default_audit_log
from .settings import settings
from .tracing import current_trace_id, traced

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
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage
    model: str
    provider: str
    latency_ms: float
    cost_usd: float | None
    raw: dict[str, Any] = Field(default_factory=dict)


def estimate_cost(model: str, usage: Usage) -> float | None:
    p = _PRICING.get(model)
    if not p:
        warnings.warn(f"no pricing configured for model '{model}'; cost is unknown", RuntimeWarning, stacklevel=2)
        return None
    return (usage.prompt_tokens * p["input_per_1m"] + usage.completion_tokens * p["output_per_1m"]) / 1e6


class ProviderError(RuntimeError):
    pass


class QuotaExceededError(ProviderError):
    """Billing/quota block requiring operator action, not a transient rate limit."""


_QUOTA_CODES = frozenset({
    "insufficient_quota", "credit_balance_exhausted", "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
})


# Pass to `chat(fallback=...)` to pin a call to exactly one model, disabling configured fallback -
# `[]` works too, this is just a documented, greppable spelling of the same thing.
NO_FALLBACK: Final[tuple[str, ...]] = ()


def _candidates(model: str | None, fallback: Sequence[str] | None) -> list[str]:
    """Resolve the ordered list of models `chat()` will try.

    `fallback=None` (the default) is "unspecified" and falls back to `settings.fallback_models`, for
    back-compat with every existing caller. Any explicit sequence - including `[]`/`NO_FALLBACK` - is
    used exactly as given: `fallback or settings.fallback_models` couldn't tell "caller wants no
    fallback" apart from "caller didn't say", since `[]` is falsy, so `fallback=[]` silently fell back
    to the configured models anyway. That silently mixes a second judge's outputs into a calibration
    run that must be pinned to one model.
    """
    primary = model or settings.default_model
    configured = fallback if fallback is not None else settings.fallback_models
    return [primary] + list(configured)


@traced(kind="llm")
def chat(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    fallback: Sequence[str] | None = None,
    **extra: Any,
) -> ChatResult:
    """Synchronous chat with retries (429/5xx), timeout, and ordered model fallback.

    `fallback=None` uses `settings.fallback_models`; pass `[]` or `NO_FALLBACK` to disable fallback
    entirely and pin the call to `model` (or `settings.default_model`).
    """
    candidates = _candidates(model, fallback)
    last_err: Exception | None = None
    for m in candidates:
        for attempt in range(settings.max_retries):
            try:
                result = _call(m, messages, tools, temperature, max_tokens, extra)
                default_audit_log().append({
                    "type": "model_call", "model": result.model, "provider": result.provider,
                    "tokens": result.usage.total, "cost_usd": result.cost_usd,
                    "latency_ms": result.latency_ms, "trace_id": current_trace_id.get(),
                })
                return result
            except QuotaExceededError:
                # Retries and model fallback use the same account and cannot restore its quota.
                raise
            except ProviderError as e:
                last_err = e
                time.sleep(min(8.0, (2**attempt) * 0.5 + random.random() * 0.25))
    raise ProviderError(f"all candidates failed: {candidates}") from last_err


@traced(kind="llm")
def chat_stream(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    **extra: Any,
) -> Iterator[str]:
    """Yield text deltas from an OpenAI-compatible server-sent event stream."""
    body: dict[str, Any] = {
        "model": model or settings.default_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        **extra,
    }
    if tools:
        body["tools"] = tools
    with httpx.Client(timeout=settings.request_timeout_s) as client, client.stream(
        "POST",
        f"{settings.openai_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json=body,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = __import__("json").loads(data)
            delta = event.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                yield str(delta)


def _call(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float,
    max_tokens: int,
    extra: dict[str, Any],
) -> ChatResult:
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
    if r.status_code == 429:
        try:
            data = r.json()
        except ValueError:
            data = {}
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            code, kind = error.get("code"), error.get("type")
            if (isinstance(code, str) and code in _QUOTA_CODES) or kind == "insufficient_quota":
                # Only known codes reach logs; provider messages may contain credentials/PII.
                safe_code = code if isinstance(code, str) and code in _QUOTA_CODES else "insufficient_quota"
                raise QuotaExceededError(f"HTTP 429: {safe_code}; resolve provider quota before retrying")
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
        provider=r.headers.get("x-routebench-backend", "direct"),  # RouteBench stamps the chosen backend
        latency_ms=(time.perf_counter() - t0) * 1000,
        cost_usd=estimate_cost(model, usage),
        raw=data,
    )

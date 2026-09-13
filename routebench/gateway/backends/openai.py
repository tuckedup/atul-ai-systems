"""Adapter for deterministic OpenAI response and streaming chunk shapes."""
from __future__ import annotations

from typing import Any, Iterable

from .base import BackendHTTPError, BackendResult, RetryingMockAdapter


class OpenAIAdapter(RetryingMockAdapter):
    name = "openai"

    def chat(self, payload: dict[str, Any], *, stream: bool = False) -> BackendResult:
        raw, attempts = self._request(payload, stream)
        if stream:
            return self._stream_result(raw, attempts)
        if not isinstance(raw, dict):
            raise BackendHTTPError(502, "OpenAI mock returned a non-object response")
        choices = raw.get("choices") or []
        if not choices:
            raise BackendHTTPError(502, "OpenAI mock response has no choices")
        message = choices[0].get("message", {})
        usage = raw.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        model = str(raw.get("model") or self.model)
        return BackendResult(
            text=str(message.get("content") or ""),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self.estimate_cost(prompt_tokens, completion_tokens),
            attempts=attempts,
        )

    def _stream_result(self, raw: object, attempts: int) -> BackendResult:
        events: Iterable[dict[str, Any] | str]
        if isinstance(raw, dict):
            events = raw.get("events", [])
        else:
            events = raw  # type: ignore[assignment]
        chunks: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        model = self.model
        for event in events:
            if event == "[DONE]":
                continue
            if not isinstance(event, dict):
                raise BackendHTTPError(502, "invalid OpenAI streaming event")
            model = str(event.get("model") or model)
            choices = event.get("choices") or []
            if choices:
                text = choices[0].get("delta", {}).get("content")
                if text:
                    chunks.append(str(text))
            usage = event.get("usage") or {}
            prompt_tokens = max(prompt_tokens, int(usage.get("prompt_tokens", 0)))
            completion_tokens = max(
                completion_tokens, int(usage.get("completion_tokens", 0))
            )
        return BackendResult(
            text="".join(chunks),
            chunks=tuple(chunks),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self.estimate_cost(prompt_tokens, completion_tokens),
            attempts=attempts,
        )

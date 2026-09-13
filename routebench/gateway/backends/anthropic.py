"""Adapter for deterministic Anthropic message and event-stream shapes."""
from __future__ import annotations

from typing import Any, Iterable

from .base import BackendHTTPError, BackendResult, RetryingMockAdapter


class AnthropicAdapter(RetryingMockAdapter):
    name = "anthropic"

    def chat(self, payload: dict[str, Any], *, stream: bool = False) -> BackendResult:
        raw, attempts = self._request(payload, stream)
        if stream:
            return self._stream_result(raw, attempts)
        if not isinstance(raw, dict):
            raise BackendHTTPError(502, "Anthropic mock returned a non-object response")
        blocks = raw.get("content") or []
        text = "".join(
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = raw.get("usage", {})
        prompt_tokens = int(usage.get("input_tokens", 0))
        completion_tokens = int(usage.get("output_tokens", 0))
        model = str(raw.get("model") or self.model)
        return BackendResult(
            text=text,
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
            if not isinstance(event, dict):
                raise BackendHTTPError(502, "invalid Anthropic streaming event")
            model = str(event.get("message", {}).get("model") or event.get("model") or model)
            if event.get("type") == "content_block_delta":
                text = event.get("delta", {}).get("text")
                if text:
                    chunks.append(str(text))
            usage = event.get("message", {}).get("usage") or event.get("usage") or {}
            prompt_tokens = max(prompt_tokens, int(usage.get("input_tokens", 0)))
            completion_tokens = max(
                completion_tokens, int(usage.get("output_tokens", 0))
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

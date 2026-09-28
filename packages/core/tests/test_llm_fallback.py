from __future__ import annotations

from typing import Any

import pytest
from aisys import llm
from aisys.llm import NO_FALLBACK, ProviderError, _candidates
from aisys.settings import settings


def test_candidates_unspecified_uses_configured_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "fallback_models", ["backup-a", "backup-b"])
    assert _candidates("primary", None) == ["primary", "backup-a", "backup-b"]


def test_candidates_empty_list_disables_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "fallback_models", ["backup-a", "backup-b"])
    assert _candidates("primary", []) == ["primary"]


def test_candidates_no_fallback_sentinel_disables_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "fallback_models", ["backup-a", "backup-b"])
    assert _candidates("primary", NO_FALLBACK) == ["primary"]


def test_candidates_falls_back_to_default_model_when_model_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "default_model", "the-default")
    monkeypatch.setattr(settings, "fallback_models", [])
    assert _candidates(None, None) == ["the-default"]


def test_chat_with_fallback_none_tries_configured_fallback_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end (network-free) check that `fallback=None` still reaches the configured fallbacks."""
    monkeypatch.setattr(settings, "fallback_models", ["backup-a", "backup-b"])
    monkeypatch.setattr(settings, "max_retries", 1)
    monkeypatch.setattr(llm.time, "sleep", lambda *_a, **_k: None)

    attempted: list[str] = []

    def fake_call(model: str, *_a: Any, **_k: Any) -> llm.ChatResult:
        attempted.append(model)
        raise ProviderError(f"{model}: boom")

    monkeypatch.setattr(llm, "_call", fake_call)
    with pytest.raises(ProviderError):
        llm.chat([{"role": "user", "content": "hi"}], model="primary")
    assert attempted == ["primary", "backup-a", "backup-b"]


def test_chat_with_fallback_disabled_pins_to_one_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A judge-calibration-style call: `fallback=[]` must never silently reach a configured fallback model."""
    monkeypatch.setattr(settings, "fallback_models", ["backup-a", "backup-b"])
    monkeypatch.setattr(settings, "max_retries", 1)
    monkeypatch.setattr(llm.time, "sleep", lambda *_a, **_k: None)

    attempted: list[str] = []

    def fake_call(model: str, *_a: Any, **_k: Any) -> llm.ChatResult:
        attempted.append(model)
        raise ProviderError(f"{model}: boom")

    monkeypatch.setattr(llm, "_call", fake_call)
    with pytest.raises(ProviderError):
        llm.chat([{"role": "user", "content": "hi"}], model="primary", fallback=[])
    assert attempted == ["primary"]


def test_chat_with_no_fallback_sentinel_pins_to_one_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "fallback_models", ["backup-a", "backup-b"])
    monkeypatch.setattr(settings, "max_retries", 1)
    monkeypatch.setattr(llm.time, "sleep", lambda *_a, **_k: None)

    attempted: list[str] = []

    def fake_call(model: str, *_a: Any, **_k: Any) -> llm.ChatResult:
        attempted.append(model)
        raise ProviderError(f"{model}: boom")

    monkeypatch.setattr(llm, "_call", fake_call)
    with pytest.raises(ProviderError):
        llm.chat([{"role": "user", "content": "hi"}], model="primary", fallback=NO_FALLBACK)
    assert attempted == ["primary"]

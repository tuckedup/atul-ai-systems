"""A billing failure must never be retried as a transient 429."""
import pytest
import respx
from aisys import llm
from aisys.settings import settings


@pytest.mark.parametrize("code", sorted(llm._QUOTA_CODES))
@respx.mock
def test_quota_stops_retry_and_fallback_without_logging_response_message(monkeypatch, code):
    monkeypatch.setattr(settings, "openai_base_url", "https://provider.test/v1")
    monkeypatch.setattr(settings, "max_retries", 3)
    monkeypatch.setattr(llm.time, "sleep", lambda _: pytest.fail("quota must not sleep/retry"))
    route = respx.post("https://provider.test/v1/chat/completions").respond(
        429, json={"error": {"code": code, "type": "insufficient_quota",
                             "message": "private-provider-response"}})
    with pytest.raises(llm.QuotaExceededError, match=code) as exc:
        llm.chat([], model="gpt-4.1", fallback=["gpt-4o-mini"])
    assert route.call_count == 1
    assert "private-provider-response" not in str(exc.value)


@respx.mock
def test_legacy_quota_type_without_known_code_is_still_terminal(monkeypatch):
    monkeypatch.setattr(settings, "openai_base_url", "https://provider.test/v1")
    route = respx.post("https://provider.test/v1/chat/completions").respond(
        429, json={"error": {"code": "private-code", "type": "insufficient_quota"}})
    with pytest.raises(llm.QuotaExceededError, match="insufficient_quota") as exc:
        llm.chat([], model="gpt-4.1", fallback=[])
    assert route.call_count == 1
    assert "private-code" not in str(exc.value)


@pytest.mark.parametrize("body", [
    {"error": {"code": "rate_limit_exceeded", "type": "rate_limit_error"}},
    {"error": {"code": ["unexpected"]}},
    {"error": "unstructured"}, ["unexpected"],
])
@respx.mock
def test_nonquota_429_retains_bounded_retry(monkeypatch, body):
    monkeypatch.setattr(settings, "openai_base_url", "https://provider.test/v1")
    monkeypatch.setattr(settings, "max_retries", 2)
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    route = respx.post("https://provider.test/v1/chat/completions").respond(429, json=body)
    with pytest.raises(llm.ProviderError) as exc:
        llm.chat([], model="gpt-4.1", fallback=[])
    assert not isinstance(exc.value, llm.QuotaExceededError)
    assert route.call_count == 2


@respx.mock
def test_nonjson_429_does_not_crash_parser(monkeypatch):
    monkeypatch.setattr(settings, "openai_base_url", "https://provider.test/v1")
    monkeypatch.setattr(settings, "max_retries", 1)
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    respx.post("https://provider.test/v1/chat/completions").respond(429, text="slow down")
    with pytest.raises(llm.ProviderError):
        llm.chat([], model="gpt-4.1", fallback=[])

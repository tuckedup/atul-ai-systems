import json

import httpx
import pytest
import respx
from aisys import llm
from aisys.settings import Settings, settings


def test_settings_accept_standard_openai_environment_names(monkeypatch):
    monkeypatch.delenv("AISYS_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AISYS_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://standard-openai.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "standard-key")
    configured = Settings(_env_file=None)
    assert configured.openai_base_url == "https://standard-openai.test/v1"
    assert configured.openai_api_key == "standard-key"


@respx.mock
def test_chat_returns_usage_cost_and_tool_calls(monkeypatch):
    monkeypatch.setattr(settings, "openai_base_url", "https://provider.test/v1")
    route = respx.post("https://provider.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"x-routebench-backend": "mock"},
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": "ok", "tool_calls": [{
                    "id": "call-1", "function": {"name": "lookup", "arguments": "{\"id\":1}"}
                }]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )
    )
    result = llm.chat([{"role": "user", "content": "hi"}], model="gpt-4o-mini")
    assert route.called
    assert result.text == "ok" and result.usage.total == 12
    assert result.cost_usd is not None and result.provider == "mock"
    assert result.tool_calls[0].name == "lookup"


@respx.mock
def test_unknown_model_warns_instead_of_crashing(monkeypatch):
    monkeypatch.setattr(settings, "openai_base_url", "https://provider.test/v1")
    respx.post("https://provider.test/v1/chat/completions").mock(return_value=httpx.Response(
        200, json={"model": "private-model", "choices": [{"message": {"content": "ok"}}], "usage": {}}
    ))
    with pytest.warns(RuntimeWarning, match="no pricing"):
        result = llm.chat([], model="private-model")
    assert result.cost_usd is None


@respx.mock
def test_chat_stream_yields_sse_content(monkeypatch):
    monkeypatch.setattr(settings, "openai_base_url", "https://provider.test/v1")
    body = "".join([
        "data: " + json.dumps({"choices": [{"delta": {"content": "hel"}}]}) + "\n\n",
        "data: " + json.dumps({"choices": [{"delta": {"content": "lo"}}]}) + "\n\n",
        "data: [DONE]\n\n",
    ])
    respx.post("https://provider.test/v1/chat/completions").mock(return_value=httpx.Response(200, text=body))
    assert "".join(llm.chat_stream([], model="gpt-4o-mini")) == "hello"

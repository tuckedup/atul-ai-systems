"""Reasoning-model support must not disturb any previously measured judge hash."""
from aisys import llm
from evalops.judge import JudgeConfig


def test_reasoning_effort_is_hash_neutral_when_unset():
    a = JudgeConfig(variant_id="a", model="gpt-4.1")
    assert "reasoning_effort" not in a.as_dict() or a.reasoning_effort == ""
    b = JudgeConfig(variant_id="b", model="gpt-4.1", reasoning_effort="high")
    assert a.config_hash != b.config_hash


def test_reasoning_models_are_detected():
    for m in ("o3", "o4-mini", "gpt-5.4", "gpt-5"):
        assert llm.is_reasoning_model(m)
    for m in ("gpt-4.1", "gpt-4o-mini"):
        assert not llm.is_reasoning_model(m)


def test_reasoning_request_shape(monkeypatch):
    sent = {}

    class R:
        status_code = 200
        headers = {}  # noqa: RUF012
        def json(self):
            return {"choices": [{"message": {"content": "x"}}], "model": "o3-2025-04-16",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        def raise_for_status(self):
            pass

    class C:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def post(self, url, headers, json):
            sent.update(json)
            return R()

    monkeypatch.setattr(llm.httpx, "Client", lambda **k: C())
    llm._call("o3", [{"role": "user", "content": "hi"}], None, 0.0, 500, {"reasoning_effort": "low"})
    assert sent["max_completion_tokens"] == 500
    assert "max_tokens" not in sent and "temperature" not in sent
    assert sent["reasoning_effort"] == "low"

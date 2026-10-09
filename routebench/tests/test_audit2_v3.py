"""v3 tests run through the REAL client path (`aisys.llm._call`) with a mocked HTTP transport: no network, no spend."""
import json

import pytest
from aisys import llm
from evalops import audit2_v3 as a

SRC = "Mayor Lee said on Monday that the council will vote on the budget next week. She hoped repairs would start in spring."
OK_TEXT = json.dumps({"verdict": "supported"})


def item(i, source=SRC):
    return {"id": f"c{i}", "source": source, "candidate": "The council will vote next week.", "label": 1, "v18": 1}


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body, self.headers = status, body, {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def reply(usage="default", text=OK_TEXT, model="o4-mini-2025-04-16"):
    body = {"choices": [{"message": {"content": text}}], "model": model}
    if usage == "default":
        body["usage"] = {"prompt_tokens": 700, "completion_tokens": 1500}
    elif usage is not None:
        body["usage"] = usage
    return FakeResponse(200, body)


@pytest.fixture
def transport(monkeypatch):
    """Queue of responses for httpx.Client.post inside aisys.llm; counts real HTTP requests and forbids llm.chat."""
    state = {"queue": [], "posts": 0}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers=None, json=None):
            state["posts"] += 1
            state["last_body"] = json
            return state["queue"].pop(0)

    monkeypatch.setattr(llm.httpx, "Client", Client)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("llm.chat retries internally; v3 must not use it")))
    return state


def frozen(tmp_path, items):
    protocol = {"protocol": "audit2-v3", "hashes": a.code_and_data_hashes(), "cap_usd_total": a.TOTAL_CAP_USD,
                "stages": {"train": len(items), "dev": len(items)}, "population_digest": {s: a.population_digest(items) for s in ("train", "dev")}}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf8")
    return tmp_path


def run(tmp_path, items, **kw):
    return a.execute_stage(frozen(tmp_path, items), "train", items, a.http_attempt, sleep=lambda s: None, **kw)


def journal_ops(tmp_path):
    return [json.loads(x) for x in (tmp_path / "spend_journal.jsonl").read_text().splitlines()]


def test_one_reservation_is_exactly_one_http_request(tmp_path, transport):
    transport["queue"] = [reply()]
    out = run(tmp_path, [item(0)])
    assert out["status"] == "COMPLETE" and transport["posts"] == 1
    assert [e["op"] for e in journal_ops(tmp_path)] == ["reserve", "settle"]


def test_transient_failures_are_retried_by_the_journaled_outer_loop(tmp_path, transport):
    transport["queue"] = [FakeResponse(503), reply()]
    out = run(tmp_path, [item(0)])
    assert out["status"] == "COMPLETE" and transport["posts"] == 2  # the client did NOT retry on its own
    assert [e["op"] for e in journal_ops(tmp_path)] == ["reserve", "settle", "reserve", "settle"]
    assert json.loads((tmp_path / "rows_train.jsonl").read_text())["attempts"] == 2


def test_failed_request_is_charged_at_its_reservation(tmp_path, transport):
    transport["queue"] = [FakeResponse(503), FakeResponse(503)]
    out = run(tmp_path, [item(0)])
    assert out["status"] == "INCOMPLETE: failed audits" and transport["posts"] == 2
    reserve = journal_ops(tmp_path)[0]["usd"]
    assert out["spent_usd_total"] == pytest.approx(2 * reserve, rel=1e-3)


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 0, "completion_tokens": 0}, {"prompt_tokens": "9", "completion_tokens": 5},
                                   {"prompt_tokens": -1, "completion_tokens": 5}, {"prompt_tokens": 100}, {"prompt_tokens": True, "completion_tokens": 3}])
def test_missing_or_malformed_usage_charges_the_reservation_not_zero(tmp_path, transport, usage):
    transport["queue"] = [reply(usage=usage)]
    out = run(tmp_path, [item(0)])
    reserve = journal_ops(tmp_path)[0]["usd"]
    assert out["spent_usd_total"] == pytest.approx(reserve, rel=1e-3) and out["usage_unverified"] == 1


def test_valid_usage_is_charged_at_the_real_price(tmp_path, transport):
    transport["queue"] = [reply(usage={"prompt_tokens": 700, "completion_tokens": 1500})]
    out = run(tmp_path, [item(0)])
    assert out["spent_usd_total"] == pytest.approx(a.usd(700, 1500), rel=1e-4)


def test_usage_above_the_reserved_token_bounds_aborts(tmp_path, transport):
    transport["queue"] = [reply(usage={"prompt_tokens": 700, "completion_tokens": a.MAX_TOKENS + 1})]
    out = run(tmp_path, [item(0), item(1)])
    assert out["status"].startswith("ABORTED") and out["report"]["status"] == "WITHHELD_INCOMPLETE" and transport["posts"] == 1


def test_input_longer_than_the_old_character_estimate_is_still_covered(tmp_path, transport):
    dense = "数据" * 4000 + " " + SRC  # non-ASCII text: far more tokens per character than the v2 estimate (len//2 + 200) assumed
    prompt = a.build_prompt(dense, "The council will vote next week.")
    assert a.prompt_tokens_bound(prompt) > len(prompt) // 2 + 200
    assert a.prompt_tokens_bound(prompt) >= len(prompt.encode("utf8"))  # tokens <= bytes for byte-level BPE
    transport["queue"] = [reply()]
    run(tmp_path, [item(0, source=dense)])
    assert journal_ops(tmp_path)[0]["usd"] == pytest.approx(a.worst_case_usd(prompt))
    assert transport["last_body"]["max_completion_tokens"] == a.MAX_TOKENS  # the output side is capped by the API itself


def test_request_uses_reasoning_model_parameters_and_one_model(tmp_path, transport):
    transport["queue"] = [reply()]
    run(tmp_path, [item(0)])
    body = transport["last_body"]
    assert body["model"] == "o4-mini" and body["reasoning_effort"] == "medium" and "temperature" not in body


def test_substituted_model_aborts(tmp_path, transport):
    transport["queue"] = [reply(model="gpt-4.1-mini")]
    out = run(tmp_path, [item(0)])
    assert out["status"].startswith("ABORTED") and "gpt-4.1-mini" in out["status"]


def test_quota_error_stops_the_run_and_is_charged(tmp_path, transport):
    transport["queue"] = [FakeResponse(429, {"error": {"code": "insufficient_quota"}})]
    out = run(tmp_path, [item(0)])
    assert out["status"].startswith("ABORTED") and transport["posts"] == 1 and out["spent_usd_total"] > 0


def test_cap_stops_before_a_request_that_could_overspend(tmp_path, transport):
    transport["queue"] = [reply()]
    prompt = a.build_prompt(SRC, "The council will vote next week.")
    out = run(tmp_path, [item(0)], cap_usd=a.worst_case_usd(prompt) * 0.5)
    assert out["status"].startswith("ABORTED") and transport["posts"] == 0


def test_pricing_constants_match_the_client_pricing():
    assert a._pricing_matches()


def test_v3_freeze_binds_all_three_code_files_and_rejects_changes(tmp_path):
    items = [item(0)]
    work = frozen(tmp_path, items)
    assert set(a.code_and_data_hashes()) >= {"audit2.py", "audit2_v2.py", "audit2_v3.py", "settings"}
    proto = json.loads((work / "protocol.json").read_text())
    proto["hashes"]["audit2_v3.py"] = "0" * 64
    (work / "protocol.json").write_text(json.dumps(proto))
    with pytest.raises(a.FreezeError, match="changed since the freeze"):
        a.execute_stage(work, "train", items, a.http_attempt)


def test_non_mapping_usage_crashes_the_client_parse_and_is_still_charged(tmp_path, transport):
    transport["queue"] = [reply(usage=[1, 2])]  # llm._call fails on it AFTER the request was sent (billed): charged at the bound, then retried
    out = run(tmp_path, [item(0)])
    assert out["status"] != "COMPLETE" and out["spent_usd_total"] >= journal_ops(tmp_path)[0]["usd"]

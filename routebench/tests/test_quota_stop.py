"""Offline quota propagation through judge repair, sampling and concurrent execution."""
import threading

import pytest
from aisys import llm
from evalops import experiments
from evalops.dataset import CaseRecord, Judgment
from evalops.experiments import JudgmentCache, SpendMeter, run_variant
from evalops.judge import JudgeConfig, judge_case


def cases(n=1):
    return [CaseRecord(case_id=f"c{i}", group_id=f"g{i}", task_class="summarize",
                        task_input=f"Summarize {i}", context="The fee was undisclosed.",
                        candidate_output="The fee was undisclosed.") for i in range(n)]


def test_quota_does_not_repeat_for_repair_or_samples(monkeypatch):
    calls = []

    def exhausted(*a, **kw):
        calls.append(kw)
        raise llm.QuotaExceededError("credit_balance_exhausted")

    monkeypatch.setattr(llm, "chat", exhausted)
    config = JudgeConfig(variant_id="quota", model="gpt-4.1", max_attempts=3, samples=3)
    judgment = judge_case(cases()[0], config)
    assert len(calls) == 1
    assert judgment.status == "quota_exhausted"
    assert judgment.raw_score is None
    assert not judgment.usable


@pytest.mark.parametrize("concurrency", [1, 4])
def test_run_stops_new_work_and_preserves_cost(monkeypatch, tmp_path, concurrency):
    calls = []
    lock = threading.Lock()

    def exhausted(case, config):
        with lock:
            calls.append(case.case_id)
        return Judgment(case_id=case.case_id, variant_id=config.variant_id,
                        model_requested=config.model, config_hash=config.config_hash,
                        status="quota_exhausted", cost_usd=0.001)

    monkeypatch.setattr(experiments, "judge_case", exhausted)
    meter = SpendMeter(cap_usd=1)
    cache = JudgmentCache(tmp_path / "cache.jsonl")
    out, summary = run_variant(cases(40), JudgeConfig(variant_id="quota", model="gpt-4.1"),
                               cache=cache, meter=meter, concurrency=concurrency, progress=False)
    assert 1 <= len(calls) <= concurrency
    assert summary.errors["quota_stopped"] == 40 - len(out)
    assert summary.errors["quota_exhausted"] == len(out)
    assert "budget_stopped" not in summary.errors
    assert meter.spent_usd == pytest.approx(len(out) * 0.001)
    assert meter.reserved_usd == pytest.approx(0)
    assert meter.in_flight == 0

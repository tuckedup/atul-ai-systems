"""The spend cap, and the three ways it was not hard.

An independent review reproduced `$0.03 cap -> $0.04 recorded spending` and pointed at three
mechanisms. All three were confirmed before being fixed, and each has a test here that fails
against the old behaviour:

1.  `reserve_usd` was a flat $0.01 estimate, not a bound. `_commit` added the real cost with no
    comparison, so one call costing more than its reservation walked through the cap. The gap was
    not marginal: a 2,000-token gpt-4.1 judgment has a worst-case cost of $0.08, eight times the
    estimate that was supposed to cover it.
2.  `reserve()` scaled by `max_attempts` but not by `samples`. `judge_case` draws k samples and
    retries each one, so the declared v10/v11 configs (samples=3, max_attempts=3) reserved three
    calls' worth of budget for a nine-call worst case.
3.  A `provider_error` released the whole reservation and recorded nothing, discarding the cost of
    the attempts that had already run. That makes the cap a limit on *successful* spending, which
    is the wrong quantity and the more forgiving one.
"""
from __future__ import annotations

import threading

import pytest
from evalops.experiments import (
    BudgetExceeded,
    SpendMeter,
    UnpricedModel,
    worst_case_call_usd,
)
from evalops.judge import JudgeConfig

# ---------------------------------------------------------------- 1. the reservation is a bound


def test_the_reviewers_reproduction_is_recorded_as_a_breach():
    # $0.03 cap, one call that really cost $0.04. The money is already gone, so it is committed --
    # but it can no longer pass silently.
    m = SpendMeter(cap_usd=0.03, reserve_usd=0.01)
    m.reserve(max_attempts=1).record(0.04, model="gpt-4o-mini")
    d = m.as_dict()
    assert d["spent_usd"] == pytest.approx(0.04)
    assert d["cap_exceeded"] is True
    assert d["reservation_breaches"] == 1
    assert d["worst_breach_usd"] == pytest.approx(0.03)


def test_a_price_derived_bound_absorbs_the_same_call():
    cfg = JudgeConfig(variant_id="v", model="gpt-4.1", max_tokens=2000)
    m = SpendMeter(cap_usd=1.0)
    m.reserve(max_attempts=1, worst_case_usd=worst_case_call_usd("gpt-4.1", cfg)).record(
        0.04, model="gpt-4.1"
    )
    d = m.as_dict()
    assert d["cap_exceeded"] is False
    assert d["reservation_breaches"] == 0


def test_the_flat_estimate_was_far_below_the_real_bound():
    # Documents why the estimate failed rather than asserting that it did: the shortfall scales
    # with the model's price, so a cheap-model run looked fine and a frontier-model run did not.
    cfg = JudgeConfig(variant_id="v", model="gpt-4.1", max_tokens=2000)
    flat = SpendMeter(cap_usd=1.0).reserve_usd
    assert worst_case_call_usd("gpt-4.1", cfg) == pytest.approx(8 * flat), (
        "the bound for a 2000-token gpt-4.1 judgment is exactly 8x the flat estimate that "
        "was supposed to cover it"
    )
    # The cheap model is the reason the gap went unnoticed: there the estimate was generous.
    cheap = JudgeConfig(variant_id="v", model="gpt-4o-mini", max_tokens=1200)
    assert worst_case_call_usd("gpt-4o-mini", cheap) < flat


def test_worst_case_bound_scales_with_max_tokens():
    cheap = JudgeConfig(variant_id="v", model="gpt-4.1-mini", max_tokens=500)
    dear = JudgeConfig(variant_id="v", model="gpt-4.1-mini", max_tokens=4000)
    assert worst_case_call_usd("gpt-4.1-mini", dear) > worst_case_call_usd("gpt-4.1-mini", cheap)


def test_worst_case_bound_uses_counted_prompt_tokens_when_given():
    cfg = JudgeConfig(variant_id="v", model="gpt-4o-mini", max_tokens=100)
    counted = worst_case_call_usd("gpt-4o-mini", cfg, prompt_tokens=50)
    ceiling = worst_case_call_usd("gpt-4o-mini", cfg)
    assert 0 < counted < ceiling


def test_worst_case_bound_refuses_an_unpriced_model():
    # Returning 0.0 would make the cap unbounded, which is the failure `require_priced` exists for.
    cfg = JudgeConfig(variant_id="v", model="no-such-model")
    with pytest.raises(UnpricedModel):
        worst_case_call_usd("no-such-model", cfg)


# ---------------------------------------------------------------- 2. samples multiply


def test_samples_multiply_the_reservation():
    cfg = JudgeConfig(variant_id="v11", model="gpt-4.1-mini", samples=3, max_attempts=3,
                      max_tokens=2000)
    bound = worst_case_call_usd(cfg.model, cfg)
    m = SpendMeter(cap_usd=10.0)
    m.reserve(max_attempts=cfg.max_attempts, samples=cfg.samples, worst_case_usd=bound)
    assert m.reserved_usd == pytest.approx(bound * cfg.max_attempts * cfg.samples)


def test_a_nine_call_judgment_cannot_be_admitted_by_a_three_call_budget():
    # The concrete v11 shape. Under the old sizing this was admitted and then overspent.
    cfg = JudgeConfig(variant_id="v11", model="gpt-4.1-mini", samples=3, max_attempts=3,
                      max_tokens=2000)
    bound = worst_case_call_usd(cfg.model, cfg)
    m = SpendMeter(cap_usd=bound * 3)
    with pytest.raises(BudgetExceeded):
        m.reserve(max_attempts=cfg.max_attempts, samples=cfg.samples, worst_case_usd=bound)


def test_samples_of_one_is_unchanged():
    m = SpendMeter(cap_usd=1.0)
    m.reserve(max_attempts=2, samples=1, worst_case_usd=0.01)
    assert m.reserved_usd == pytest.approx(0.02)


@pytest.mark.parametrize("samples,attempts", [(0, 1), (1, 0), (0, 0)])
def test_degenerate_counts_still_reserve_one_call(samples, attempts):
    m = SpendMeter(cap_usd=1.0)
    m.reserve(max_attempts=attempts, samples=samples, worst_case_usd=0.01)
    assert m.reserved_usd == pytest.approx(0.01)


# ---------------------------------------------------------------- 3. failed attempts still cost


def test_a_provider_error_commits_the_cost_it_burned():
    m = SpendMeter(cap_usd=1.0)
    m.reserve(max_attempts=3, worst_case_usd=0.02).release(
        accrued_usd=0.0042, model="gpt-4o-mini"
    )
    assert m.spent_usd == pytest.approx(0.0042)
    assert m.calls == 1
    assert m.reserved_usd == pytest.approx(0.0)


def test_a_costless_failure_releases_without_charging():
    m = SpendMeter(cap_usd=1.0)
    m.reserve(max_attempts=3, worst_case_usd=0.02).release(accrued_usd=0.0)
    assert m.spent_usd == 0.0
    assert m.calls == 0
    assert m.reserved_usd == pytest.approx(0.0)


def test_release_with_no_accrued_cost_argument_is_still_free():
    # Back-compat: existing callers that call release() with no arguments keep the old meaning.
    m = SpendMeter(cap_usd=1.0)
    m.reserve(max_attempts=1, worst_case_usd=0.01).release()
    assert m.spent_usd == 0.0
    assert m.reserved_usd == pytest.approx(0.0)


def test_accrued_cost_from_failures_eventually_closes_the_cap():
    # The property that matters: repeated failures cannot be retried forever for free.
    m = SpendMeter(cap_usd=0.05, reserve_usd=0.01)
    for _ in range(100):
        try:
            slot = m.reserve(max_attempts=1, worst_case_usd=0.01)
        except BudgetExceeded:
            break
        slot.release(accrued_usd=0.01, model="gpt-4o-mini")
    else:
        pytest.fail("failures never exhausted the cap")
    assert m.spent_usd <= m.cap_usd + 1e-12


def test_exhausted_reservation_cannot_be_settled_twice():
    m = SpendMeter(cap_usd=1.0)
    slot = m.reserve(max_attempts=1, worst_case_usd=0.01)
    slot.record(0.001, model="gpt-4o-mini")
    with pytest.raises(RuntimeError, match="already recorded"):
        slot.record(0.001, model="gpt-4o-mini")


def test_release_after_record_is_a_no_op():
    m = SpendMeter(cap_usd=1.0)
    slot = m.reserve(max_attempts=1, worst_case_usd=0.01)
    slot.record(0.001, model="gpt-4o-mini")
    slot.release(accrued_usd=0.05)  # must not double-charge
    assert m.spent_usd == pytest.approx(0.001)


def test_an_unpriced_cost_releases_and_raises_rather_than_recording_zero():
    m = SpendMeter(cap_usd=1.0)
    slot = m.reserve(max_attempts=1, worst_case_usd=0.01)
    with pytest.raises(UnpricedModel):
        slot.record(None, model="mystery-model")
    assert m.spent_usd == 0.0
    assert m.reserved_usd == pytest.approx(0.0)


# ---------------------------------------------------------------- the cap under concurrency


def test_concurrent_workers_cannot_collectively_pass_the_cap():
    # The original reason reservations exist. Every worker reserves a true bound before spending,
    # so committed spend stays inside the cap even though none of them knows the others' costs.
    bound = 0.01
    m = SpendMeter(cap_usd=0.10)
    admitted: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(20):
            try:
                slot = m.reserve(max_attempts=1, worst_case_usd=bound)
            except BudgetExceeded:
                return
            with lock:
                admitted.append(1)
            slot.record(bound, model="gpt-4o-mini")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert m.spent_usd <= m.cap_usd + 1e-12, m.as_dict()
    assert len(admitted) == 10
    assert m.as_dict()["cap_exceeded"] is False
    assert m.as_dict()["reservation_breaches"] == 0


def test_reservations_are_released_when_the_protected_call_raises():
    m = SpendMeter(cap_usd=1.0)
    with pytest.raises(ValueError), m.reserve(max_attempts=1, worst_case_usd=0.01):
        raise ValueError("provider blew up before record()")
    assert m.reserved_usd == pytest.approx(0.0)
    assert m.in_flight == 0

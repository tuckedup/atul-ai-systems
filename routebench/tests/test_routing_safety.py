"""Local-only routing safety regressions: no providers, services, network, or GPU."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest
from gateway.router import Backend, NoEligibleBackend, Request, Router


class Audit:
    def append(self, event: dict[str, Any]) -> str:
        return str(event.get("type", "event"))


def backend(
    model: str,
    *,
    weight: float = 1.0,
    latency: float = 10.0,
    context: int = 10_000,
    input_price: float = 1.0,
    output_price: float = 1.0,
    circuit_open: bool = False,
) -> Backend:
    return Backend(
        model,
        "local",
        input_price,
        output_price,
        latency,
        circuit_open=circuit_open,
        traffic_weight=weight,
        max_context=context,
    )


def request(**changes: Any) -> Request:
    values: dict[str, Any] = {
        "task_class": "chat",
        "difficulty": "easy",
        "est_prompt_tokens": 100,
        "latency_sla_ms": 100.0,
        "tenant_budget_remaining_usd": 1.0,
        "quality_floor": 0.5,
        "max_output_tokens": 50,
        "request_id": "request-1",
    }
    values.update(changes)
    return Request(**values)


def test_all_safety_constraints_are_hard_eligibility_filters() -> None:
    backends = [
        backend("safe"),
        backend("disabled", weight=0),
        backend("open", circuit_open=True),
        backend("excluded"),
        backend("context", context=149),
        backend("budget", input_price=1_000_000),
        backend("missing"),
        backend("nonfinite"),
        backend("floor"),
        backend("slow", latency=101),
    ]
    quality = {item.model: {"chat": 1.0} for item in backends}
    quality.pop("missing")
    quality["nonfinite"]["chat"] = math.nan
    quality["floor"]["chat"] = 0.49
    router = Router(quality, backends, Audit())  # type: ignore[arg-type]

    decision = router.route(request(excluded_models=frozenset({"excluded"})))

    assert decision.model == "safe"
    assert all(model != "disabled" for model, _ in decision.alternatives)


def test_no_available_backend_raises_with_exclusion_metadata() -> None:
    router = Router(
        {"off": {"chat": 1.0}, "unknown": {}},
        [backend("off", weight=0), backend("unknown")],
        Audit(),  # type: ignore[arg-type]
    )

    with pytest.raises(NoEligibleBackend) as raised:
        router.route(request())

    assert raised.value.exclusions == {
        "off": ("traffic disabled",),
        "unknown": ("missing or invalid quality",),
    }


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("input_per_1m", math.nan),
        ("output_per_1m", math.inf),
        ("p95_latency_ms", math.nan),
        ("traffic_weight", math.inf),
        ("traffic_weight", -1.0),
    ],
)
def test_nonfinite_or_negative_backend_fields_are_never_eligible(
    field_name: str, invalid: float
) -> None:
    unsafe = replace(backend("unsafe"), **{field_name: invalid})
    router = Router(
        {"unsafe": {"chat": 1.0}, "safe": {"chat": 0.6}},
        [unsafe, backend("safe")],
        Audit(),  # type: ignore[arg-type]
    )

    assert router.route(request()).model == "safe"


def test_zero_weight_cannot_win_when_all_eligible_scores_are_negative() -> None:
    router = Router(
        {"zero": {"chat": 1.0}, "active": {"chat": 0.1}},
        [backend("zero", weight=0), backend("active")],
        Audit(),  # type: ignore[arg-type]
        w_q=0,
        w_c=1,
        w_l=1,
    )

    decision = router.route(request(quality_floor=0))

    assert decision.model == "active"
    assert decision.score < 0


def test_request_exclusions_are_respected_deterministically_during_retry() -> None:
    router = Router(
        {"a": {"chat": 0.9}, "b": {"chat": 0.8}},
        [backend("a", weight=0.25), backend("b", weight=1.0)],
        Audit(),  # type: ignore[arg-type]
    )

    first = router.route(request(request_id="stable-retry"))
    second = router.route(
        request(request_id="stable-retry", excluded_models=frozenset({first.model}))
    )

    assert second.model != first.model
    assert router.route(request(request_id="stable-retry")).model == first.model


def test_fractional_5_25_100_weights_form_stable_relative_eligible_cohorts() -> None:
    models = [backend("five", weight=0.05), backend("twenty-five", weight=0.25), backend("hundred")]
    quality = {item.model: {"chat": 0.8} for item in models}
    router = Router(quality, models, Audit())  # type: ignore[arg-type]

    ids = [f"cohort-{index}" for index in range(10_000)]
    first = [router.route(request(request_id=value)).model for value in ids]
    counts = Counter(first)

    assert [router.route(request(request_id=value)).model for value in ids[:100]] == first[:100]
    assert counts["five"] / len(ids) == pytest.approx(0.05 / 1.30, abs=0.01)
    assert counts["twenty-five"] / len(ids) == pytest.approx(0.25 / 1.30, abs=0.015)
    assert counts["hundred"] / len(ids) == pytest.approx(1.0 / 1.30, abs=0.015)


def test_all_weight_one_uses_quality_cost_latency_ranking() -> None:
    router = Router(
        {"better": {"chat": 0.9}, "worse": {"chat": 0.2}},
        [backend("worse"), backend("better")],
        Audit(),  # type: ignore[arg-type]
        w_c=0,
        w_l=0,
    )

    assert router.route(request(quality_floor=0, request_id="irrelevant")).model == "better"


def test_context_limit_includes_requested_output_tokens() -> None:
    router = Router(
        {"small": {"chat": 1.0}, "large": {"chat": 0.8}},
        [backend("small", context=120), backend("large", context=200)],
        Audit(),  # type: ignore[arg-type]
    )

    assert router.route(request(est_prompt_tokens=100, max_output_tokens=21)).model == "large"


def test_cost_estimate_keeps_legacy_500_default_and_scales_with_output_limit() -> None:
    priced = backend("priced", input_price=2, output_price=10)
    router = Router({"priced": {"chat": 1.0}}, [priced], Audit())  # type: ignore[arg-type]
    legacy = Request("chat", "easy", 100, None, 1.0)
    larger = Request("chat", "easy", 100, None, 1.0, max_output_tokens=1_000)

    assert router._est_cost(priced, legacy) == pytest.approx((100 * 2 + 500 * 10) / 1e6)
    assert router._est_cost(priced, larger) == pytest.approx((100 * 2 + 1_000 * 10) / 1e6)


def test_weight_updates_are_atomic_validated_and_change_policy_fingerprint() -> None:
    one, two = backend("one"), backend("two")
    router = Router(
        {"one": {"chat": 0.7}, "two": {"chat": 0.8}},
        [one, two],
        Audit(),  # type: ignore[arg-type]
    )
    original = router.policy_fingerprint()

    with pytest.raises(ValueError, match="unknown backend"):
        router.apply_traffic_weights({"one": 0.25, "missing": 1.0})
    assert router.traffic_weights() == {"one": 1.0, "two": 1.0}
    for invalid in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="invalid traffic weight"):
            router.apply_traffic_weights({"one": invalid})
    router.apply_traffic_weights({"one": 0.25})

    assert router.traffic_weights() == {"one": 0.25, "two": 1.0}
    assert router.policy_fingerprint() != original
    weighted = router.policy_fingerprint()
    one.circuit_open = True
    assert router.policy_fingerprint() == weighted


@pytest.mark.parametrize(
    "change",
    [
        {"tenant_budget_remaining_usd": math.nan},
        {"latency_sla_ms": math.inf},
        {"quality_floor": math.nan},
        {"max_output_tokens": -1},
    ],
)
def test_invalid_request_fields_fail_closed(change: dict[str, Any]) -> None:
    router = Router(
        {"model": {"chat": 1.0}},
        [backend("model")],
        Audit(),  # type: ignore[arg-type]
    )

    with pytest.raises(NoEligibleBackend, match="invalid request"):
        router.route(request(**change))

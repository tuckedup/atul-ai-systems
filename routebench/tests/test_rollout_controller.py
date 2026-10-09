"""Local state-machine tests; observations and calibration are explicit test simulations."""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from evalops.calibrate import identity_hash
from evalops.rollout_controller import OfflineEvidence, RolloutController
from evalops.rubrics import bundle_hash
from gateway.state import ModelState


class FakeClock:
    def __init__(self, value: float = 1_000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeRouter:
    def __init__(self, weights: dict[str, float]):
        self.weights = dict(weights)
        self.calls: list[dict[str, float]] = []

    def apply_traffic_weights(self, weights: dict[str, float]) -> None:
        self.weights = dict(weights)
        self.calls.append(dict(weights))


def _calibration(tmp_path: Path, *, dataset_hash: str = "dataset-1") -> tuple[Path, dict[str, object]]:
    identity = {
        "variant_id": "simulation-only",
        "threshold": 0.5,
        "judge_config": {"model": "unit-test-judge"},
        "rubric_bundle_hash": bundle_hash(),
        "dataset_hash": dataset_hash,
        "split_seed": 7,
        "annotations_hash": "unit-test-annotations",
        "split_membership_hash": "unit-test-split",
    }
    payload: dict[str, object] = {
        "bundle_id": identity_hash(**identity),
        "created_at": "2026-10-06T00:00:00+00:00",
        **identity,
        "min_kappa": 0.74,
        "dev_kappa": 0.8,
        "dev_n": 100,
        "selection_rule": "unit-test simulation",
        "threshold_grid": [],
        "label_provenance_counts": {"unit_test_simulation": 100},
        "test_result": {
            "completion": {
                "paired": 100,
                "unjudged": 0,
                "completion_rate": 1.0,
                "judge_errors": {},
            },
            "tracks": {
                "human_judgment_of_response": {
                    "track": "human_judgment_of_response",
                    "insufficient": False,
                    "n": 100,
                    "kappa": 0.78,
                    "defined": True,
                    "observed_agreement": 0.9,
                    "passes_min_kappa": True,
                    "interval": {"low": 0.70, "high": 0.84, "usable": True},
                    "reason": "",
                    "n_labelled": 100,
                    "track_completion": 1.0,
                }
            },
        },
    }
    path = tmp_path / "unit-test-calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _state(tmp_path: Path, bundle_id: str, dataset_hash: str = "dataset-1") -> ModelState:
    state = ModelState(f"sqlite:///{tmp_path / 'state.db'}")
    provenance = {
        "calibration_bundle_id": bundle_id,
        "rubric_bundle_hash": bundle_hash(),
        "dataset_hash": dataset_hash,
        "n_samples": {"reason": 100},
        "measured_at": "2026-10-06T00:00:00+00:00",
    }
    state.upsert("candidate", "local", {"reason": 0.91}, 0.2, **provenance)
    state.upsert("incumbent", "local", {"reason": 0.90}, 0.7, **provenance)
    state.upsert("other", "local", {"reason": 0.88}, 0.1, **provenance)
    return state


def _controller(tmp_path: Path, *, deadline: float = 90.0):
    calibration, payload = _calibration(tmp_path)
    state = _state(tmp_path, str(payload["bundle_id"]))
    baseline = state.weight_map()
    clock = FakeClock()
    router = FakeRouter(baseline)
    controller = RolloutController(
        state,
        candidate="candidate",
        incumbent="incumbent",
        task_class="reason",
        calibration=calibration,
        expected_dataset_hash="dataset-1",
        on_weights_changed=router.apply_traffic_weights,
        clock=clock,
        deadline_seconds=deadline,
        rollout_id="rollout-test",
    )
    evidence = OfflineEvidence(
        candidate="candidate",
        task_class="reason",
        calibration_bundle_id=str(payload["bundle_id"]),
        rubric_bundle_hash=bundle_hash(),
        dataset_hash="dataset-1",
        n_samples=100,
        min_samples=50,
    )
    return controller, state, clock, router, evidence, baseline, payload


def _feed(controller: RolloutController, payload: dict[str, object], prefix: str, value: float) -> None:
    for index in range(20):
        controller.observe(
            observation_id=f"{prefix}-{index}",
            candidate="candidate",
            task_class="reason",
            calibration_bundle_id=payload["bundle_id"],
            dataset_hash="dataset-1",
            value=value,
        )


def test_full_rollout_requires_twenty_matching_observations_per_window(tmp_path):
    controller, state, _clock, router, evidence, _baseline, payload = _controller(tmp_path)
    assert controller.start(evidence)["weights"] == {
        "candidate": 0.0, "incumbent": 1.0, "other": 0.0,
    }
    assert router.weights == state.weight_map()

    for expected, prefix in (("canary-5", "shadow"), ("canary-25", "five"), ("full", "twenty-five")):
        _feed(controller, payload, prefix, 0.91)
        assert controller.status()["observation_count"] == 20
        controller.tick()
        assert controller.status()["stage"] == expected
        assert router.weights == state.weight_map()

    assert state.weight_map() == {"candidate": 1.0, "incumbent": 0.0, "other": 0.0}
    assert controller.status()["status"] == "completed"
    assert [event["to_stage"] for event in state.rollout_events("rollout-test")] == [
        "shadow", "canary-5", "canary-25", "full",
    ]


def test_regression_restores_exact_sql_snapshot_and_router_mapping(tmp_path):
    controller, state, clock, router, evidence, baseline, payload = _controller(tmp_path)
    controller.start(evidence)
    _feed(controller, payload, "shadow", 0.91)
    controller.tick()
    clock.advance(3)
    _feed(controller, payload, "regression", 0.80)
    result = controller.tick()

    assert result["stage"] == "rollback"
    assert state.weight_map() == baseline
    assert router.weights == baseline
    assert router.calls[-1] == baseline
    event = state.rollout_events("rollout-test")[-1]
    assert event["to_stage"] == "rollback"
    assert event["weights"] == baseline
    assert event["elapsed_seconds"] == pytest.approx(3.0)


def test_fake_clock_deadline_rolls_back_without_claiming_measurement(tmp_path):
    controller, state, clock, router, evidence, baseline, payload = _controller(tmp_path)
    controller.start(evidence)
    _feed(controller, payload, "shadow", 0.91)
    controller.tick()
    clock.advance(90)

    assert controller.check_deadline() is True
    assert controller.status()["status"] == "rolled-back"
    assert state.weight_map() == baseline
    assert router.weights == baseline
    assert "deadline expired" in controller.status()["reason"]


def test_callback_exception_restores_database_and_callback_view(tmp_path):
    controller, state, _clock, router, evidence, baseline, payload = _controller(tmp_path)
    controller.start(evidence)
    _feed(controller, payload, "shadow", 0.91)
    original = router.apply_traffic_weights
    failed = False

    def fail_once(weights: dict[str, float]) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("simulated router apply failure")
        original(weights)

    controller.on_weights_changed = fail_once
    with pytest.raises(RuntimeError, match="simulated router"):
        controller.tick()
    assert state.weight_map() == baseline
    assert router.weights == baseline
    assert controller.status()["stage"] == "rollback"


def test_observations_reject_mismatch_duplicate_and_nonfinite_values(tmp_path):
    controller, _state, _clock, _router, evidence, _baseline, payload = _controller(tmp_path)
    controller.start(evidence)
    common = {
        "candidate": "candidate",
        "task_class": "reason",
        "calibration_bundle_id": payload["bundle_id"],
        "dataset_hash": "dataset-1",
    }
    controller.observe(observation_id="one", value=0.9, **common)
    with pytest.raises(ValueError, match="duplicate"):
        controller.observe(observation_id="one", value=0.9, **common)
    with pytest.raises(ValueError, match="identity mismatch"):
        controller.observe(observation_id="wrong-data", value=0.9, **{**common, "dataset_hash": "other"})
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            controller.observe(observation_id=f"bad-{bad}", value=bad, **common)
    assert controller.status()["observation_count"] == 1


def test_state_dataset_filter_and_strict_provenance(tmp_path):
    state = ModelState(f"sqlite:///{tmp_path / 'filter.db'}")
    state.upsert(
        "good", "local", {"reason": 0.9}, calibration_bundle_id="bundle",
        rubric_bundle_hash="rubric", dataset_hash="right", n_samples={"reason": 20},
        measured_at="2026-10-06T00:00:00Z",
    )
    state.upsert(
        "wrong", "local", {"reason": 0.9}, calibration_bundle_id="bundle",
        rubric_bundle_hash="rubric", dataset_hash="wrong", n_samples={"reason": 20},
        measured_at="2026-10-06T00:00:00Z",
    )
    state.upsert("unmeasured", "local", {"reason": 0.9})
    assert state.quality_matrix(dataset_hash="right", strict_provenance=True) == {
        "good": {"reason": 0.9}
    }
    with pytest.raises(ValueError, match="incomplete measured provenance"):
        state.quality_matrix(strict_provenance=True)


def test_no_calibration_bypass_or_too_small_window(tmp_path):
    state = ModelState(f"sqlite:///{tmp_path / 'guards.db'}")
    with pytest.raises(ValueError, match="calibration is required"):
        RolloutController(
            state,
            candidate="candidate",
            incumbent="incumbent",
            task_class="reason",
            calibration=None,  # type: ignore[arg-type]
            expected_dataset_hash="dataset-1",
        )
    calibration, _payload = _calibration(tmp_path)
    with pytest.raises(ValueError, match="at least 20"):
        RolloutController(
            state,
            candidate="candidate",
            incumbent="incumbent",
            task_class="reason",
            calibration=calibration,
            expected_dataset_hash="dataset-1",
            min_observations=19,
        )


def test_offline_declared_minimum_cannot_lower_policy_floor(tmp_path):
    controller, state, _clock, _router, evidence, baseline, _payload = _controller(tmp_path)
    with pytest.raises(ValueError, match="minimum sample"):
        controller.start(replace(evidence, n_samples=1, min_samples=1))
    assert state.weight_map() == baseline


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_out_of_range_online_quality_cannot_promote(tmp_path, value):
    controller, _state, _clock, _router, evidence, _baseline, payload = _controller(tmp_path)
    controller.start(evidence)
    with pytest.raises(ValueError, match="finite"):
        controller.observe(observation_id="invalid", candidate="candidate", task_class="reason",
                           calibration_bundle_id=payload["bundle_id"], dataset_hash="dataset-1", value=value)
    assert controller.status()["observation_count"] == 0


def test_second_controller_cannot_overwrite_active_global_rollout(tmp_path):
    controller, state, clock, _router, evidence, _baseline, _payload = _controller(tmp_path)
    controller.start(evidence)
    second = RolloutController(state, candidate="candidate", incumbent="incumbent", task_class="reason",
                               calibration=controller.calibration, expected_dataset_hash="dataset-1",
                               clock=clock, rollout_id="second")
    before = state.weight_map()
    with pytest.raises(RuntimeError, match="another rollout"):
        second.start(evidence)
    assert state.weight_map() == before
    assert not state.rollout_events("second")


def test_concurrent_ticks_consume_window_only_once(tmp_path):
    controller, state, _clock, _router, evidence, _baseline, payload = _controller(tmp_path)
    controller.start(evidence)
    _feed(controller, payload, "shadow", 0.91)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _i: controller.tick(), range(4)))
    assert controller.status()["stage"] == "canary-5"
    assert len(state.rollout_events("rollout-test")) == 2

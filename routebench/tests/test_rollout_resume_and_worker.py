# ruff: noqa: RUF059
"""Restart-resume and the observation worker for the rollout controller. Local SQLite and fake clocks only: no provider call, no live traffic."""
from __future__ import annotations

import time

import pytest
from evalops.rollout_controller import OfflineEvidence, RolloutController
from evalops.rollout_worker import RolloutWorker
from evalops.rubrics import bundle_hash
from gateway.state import ModelState
from test_rollout_controller import FakeClock, FakeRouter, _calibration, _state

EPOCH = 1_800_000_000.0


def started(tmp_path, *, clock_value=EPOCH, deadline=90.0):
    artifact, payload = _calibration(tmp_path)
    state = _state(tmp_path, str(payload["bundle_id"]))
    baseline = state.weight_map()
    clock = FakeClock(clock_value)
    router = FakeRouter(baseline)
    controller = RolloutController(state, candidate="candidate", incumbent="incumbent", task_class="reason", calibration=artifact, expected_dataset_hash="dataset-1",
                                   on_weights_changed=router.apply_traffic_weights, clock=clock, deadline_seconds=deadline, rollout_id="r-1")
    controller.start(OfflineEvidence("candidate", "reason", str(payload["bundle_id"]), bundle_hash(), "dataset-1", 100, 50))
    return controller, state, clock, router, baseline, artifact, payload


def obs(payload, ident, value=0.91, **kw):
    return {"observation_id": ident, "candidate": "candidate", "task_class": "reason", "calibration_bundle_id": payload["bundle_id"], "dataset_hash": "dataset-1", "value": value, **kw}


def feed(controller, payload, prefix, n=20, value=0.91):
    for i in range(n):
        controller.observe(obs(payload, f"{prefix}-{i}", value))


def restart(tmp_path):
    return ModelState(f"sqlite:///{tmp_path / 'state.db'}")  # a new connection to the same database file = a new process


# ------------------------------------------------------------------------------------------------------------ restart-resume
def test_a_canary_resumes_after_restart_with_its_window_ids_and_weights(tmp_path):
    controller, state, clock, router, baseline, artifact, payload = started(tmp_path)
    feed(controller, payload, "shadow")
    controller.tick()
    assert controller.status()["stage"] == "canary-5"
    feed(controller, payload, "c5", n=7)
    clock.advance(10)
    state2, router2 = restart(tmp_path), FakeRouter({})
    resumed = RolloutController.resume(state2, calibration=artifact, expected_dataset_hash="dataset-1", clock=clock, on_weights_changed=router2.apply_traffic_weights)
    st = resumed.status()
    assert (st["rollout_id"], st["stage"], st["status"], st["observation_count"]) == ("r-1", "canary-5", "active", 7)
    assert router2.weights == state2.weight_map() and router2.weights["candidate"] == 0.05  # the fresh router was re-synchronised
    with pytest.raises(ValueError, match="duplicate observation_id"):
        resumed.observe(obs(payload, "c5-0"))  # ids recorded before the restart are still unique
    feed(resumed, payload, "after", n=13)
    assert resumed.tick()["stage"] == "canary-25"  # 7 recovered + 13 new observations met the minimum of 20
    assert state2.weight_map()["candidate"] == 0.25


def test_downtime_cannot_extend_a_canary_deadline(tmp_path):
    controller, state, clock, router, baseline, artifact, payload = started(tmp_path)
    feed(controller, payload, "shadow")
    controller.tick()
    clock.advance(500)  # the process was down for far longer than the 90-second deadline
    state2, router2 = restart(tmp_path), FakeRouter({})
    resumed = RolloutController.resume(state2, calibration=artifact, expected_dataset_hash="dataset-1", clock=clock, on_weights_changed=router2.apply_traffic_weights)
    assert resumed.status()["status"] == "rolled-back" and "deadline expired" in resumed.status()["reason"]
    assert state2.weight_map() == baseline and router2.weights == baseline and state2.active_rollout() is None


def test_shadow_resumes_with_no_deadline(tmp_path):
    controller, state, clock, router, baseline, artifact, payload = started(tmp_path)
    clock.advance(10_000)
    resumed = RolloutController.resume(restart(tmp_path), calibration=artifact, expected_dataset_hash="dataset-1", clock=clock)
    assert resumed.status()["stage"] == "shadow" and resumed.status()["status"] == "active"


def test_weights_that_diverged_from_the_stage_restore_the_baseline_on_resume(tmp_path):
    controller, state, clock, router, baseline, artifact, payload = started(tmp_path)
    state.apply_weight_map({"candidate": 1.0, "incumbent": 0.0, "other": 0.0})  # something outside the controller changed them
    state2, router2 = restart(tmp_path), FakeRouter({})
    resumed = RolloutController.resume(state2, calibration=artifact, expected_dataset_hash="dataset-1", clock=clock, on_weights_changed=router2.apply_traffic_weights)
    assert resumed.status()["status"] == "rolled-back" and "diverged" in resumed.status()["reason"]
    assert state2.weight_map() == baseline == router2.weights


def test_a_non_wall_clock_rollout_is_not_resumed_and_can_be_rolled_back_explicitly(tmp_path):
    controller, state, clock, router, baseline, artifact, payload = started(tmp_path, clock_value=1_000.0)  # recorded with a small (monotonic-style) clock
    state2 = restart(tmp_path)
    with pytest.raises(ValueError, match="wall-clock"):
        RolloutController.resume(state2, calibration=artifact, expected_dataset_hash="dataset-1", clock=lambda: EPOCH)
    seen = []
    record = RolloutController.rollback_active(state2, clock=lambda: EPOCH, on_weights_changed=seen.append)
    assert record["rollout_id"] == "r-1" and state2.weight_map() == baseline and seen == [baseline] and state2.active_rollout() is None
    assert RolloutController.rollback_active(state2) is None  # idempotent: nothing left to roll back


def test_resume_refuses_a_clock_earlier_than_the_record_and_wrong_identities(tmp_path):
    controller, state, clock, router, baseline, artifact, payload = started(tmp_path)
    state2 = restart(tmp_path)
    with pytest.raises(ValueError, match="earlier than the recorded"):
        RolloutController.resume(state2, calibration=artifact, expected_dataset_hash="dataset-1", clock=lambda: EPOCH - 3600)
    with pytest.raises(ValueError, match="different dataset"):
        RolloutController.resume(state2, calibration=artifact, expected_dataset_hash="other-dataset", clock=clock)
    missing = tmp_path / "gone.json"
    with pytest.raises(ValueError, match="calibration"):
        RolloutController.resume(state2, calibration=missing, expected_dataset_hash="dataset-1", clock=clock)
    assert state2.active_rollout() is not None  # a refused resume changes nothing


def test_resume_without_an_active_rollout_is_an_error(tmp_path):
    artifact, payload = _calibration(tmp_path)
    state = _state(tmp_path, str(payload["bundle_id"]))
    with pytest.raises(RuntimeError, match="no active rollout"):
        RolloutController.resume(state, calibration=artifact, expected_dataset_hash="dataset-1", clock=lambda: EPOCH)


# ------------------------------------------------------------------------------------------------------------ worker
class Feed:
    """A scripted telemetry source: yields the next batch on each call (a batch may be a list or an exception)."""

    def __init__(self, batches):
        self.batches, self.calls = list(batches), 0

    def __call__(self):
        self.calls += 1
        if not self.batches:
            return []
        b = self.batches.pop(0)
        if isinstance(b, Exception):
            raise b
        return b


def batch(payload, prefix, n=20, value=0.91):
    return [obs(payload, f"{prefix}-{i}", value) for i in range(n)]


def test_the_worker_drives_a_rollout_to_completion_and_skips_duplicates(tmp_path):
    controller, state, clock, router, baseline, artifact, payload = started(tmp_path)
    slept = []
    src = Feed([batch(payload, "s"), batch(payload, "s") + batch(payload, "a"), batch(payload, "b")])
    worker = RolloutWorker(controller, src, interval_s=2.0, sleep=lambda x: (slept.append(x), clock.advance(x)))
    final = worker.run(max_steps=10)
    assert final["status"] == "completed" and final["stage"] == "full" and state.weight_map()["candidate"] == 1.0
    assert worker.duplicates == 20 and worker.ingested == 60 and slept and set(slept) == {2.0} and not worker.rejected


def test_invalid_observations_are_quarantined_not_fatal(tmp_path):
    controller, state, clock, router, baseline, artifact, payload = started(tmp_path)
    bad = [obs(payload, "x1", 2.0), obs(payload, "x2", 0.9, candidate="someone-else"), {"observation_id": "x3"}, obs(payload, "ok-1", 0.9)]
    worker = RolloutWorker(controller, Feed([bad]), sleep=lambda x: None)
    worker.step()
    assert worker.ingested == 1 and len(worker.rejected) == 3 and controller.status()["observation_count"] == 1


def test_a_regression_seen_by_the_worker_rolls_back_and_stops_it(tmp_path):
    controller, state, clock, router, baseline, artifact, payload = started(tmp_path)
    worker = RolloutWorker(controller, Feed([batch(payload, "s"), batch(payload, "bad", value=0.1)]), sleep=lambda x: clock.advance(x))
    final = worker.run(max_steps=10)
    assert final["status"] == "rolled-back" and "online regression" in final["reason"] and state.weight_map() == baseline == router.weights


def test_dead_telemetry_cannot_keep_a_canary_alive(tmp_path):
    controller, state, clock, router, baseline, artifact, payload = started(tmp_path)
    feed(controller, payload, "shadow")
    controller.tick()  # canary-5 begins; its deadline is 90 s away
    broken = Feed([RuntimeError("metrics endpoint down")] * 50)
    worker = RolloutWorker(controller, broken, interval_s=10.0, sleep=lambda x: clock.advance(x))
    final = worker.run(max_steps=40)
    assert final["status"] == "rolled-back" and "deadline expired" in final["reason"] and state.weight_map() == baseline
    assert worker.source_errors >= 9 and "metrics endpoint down" in worker.last_source_error


def test_the_worker_thread_starts_once_polls_and_stops_on_request(tmp_path):
    controller, state, clock, router, baseline, artifact, payload = started(tmp_path)
    worker = RolloutWorker(controller, Feed([batch(payload, "s")] + [[] for _ in range(1000)]), interval_s=0.01)
    worker.start()
    with pytest.raises(RuntimeError, match="already running"):
        worker.start()
    time.sleep(0.2)
    worker.stop()
    assert worker._thread is not None and not worker._thread.is_alive()
    assert controller.status()["status"] == "active" and worker.ingested == 20


def test_stepping_after_a_terminal_state_is_harmless(tmp_path):
    controller, state, clock, router, baseline, artifact, payload = started(tmp_path)
    worker = RolloutWorker(controller, Feed([batch(payload, "late")]), sleep=lambda x: clock.advance(x))
    controller._rollback("test: force terminal")
    assert worker.step()["status"] == "rolled-back"  # late observations are refused quietly, nothing changes
    assert worker.step()["status"] == "rolled-back" and state.weight_map() == baseline and worker.ingested == 0


def test_worker_rejects_a_non_positive_interval(tmp_path):
    controller = started(tmp_path)[0]
    with pytest.raises(ValueError):
        RolloutWorker(controller, Feed([]), interval_s=0)

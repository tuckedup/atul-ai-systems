"""State-to-serving integration only; calibration and observations are test simulations."""
import json
import math

import pytest
from aisys.llm import ChatResult, Usage
from evalops.rollout_controller import OfflineEvidence, RolloutController
from evalops.rubrics import bundle_hash
from fastapi.testclient import TestClient
from gateway.app import app, configure_runtime
from gateway.bootstrap import (
    GatewayStartupError,
    runtime_from_config,
    runtime_from_state,
)
from gateway.router import Backend
from test_rollout_controller import FakeClock, _calibration, _state


class LocalDouble:
    name = "local-test"

    def chat(self, payload):
        return ChatResult(text=payload["model"], usage=Usage(prompt_tokens=2, completion_tokens=1),
                          model=payload["model"], provider=self.name, cost_usd=0, latency_ms=1)


def configured(tmp_path):
    artifact, payload = _calibration(tmp_path)
    state = _state(tmp_path, str(payload["bundle_id"]))
    state.apply_weight_map({"candidate": 0, "incumbent": 1, "other": 0})
    backends = [Backend(m, "local-test", 0, 0, 1) for m in state.weight_map()]
    clients = {b.model: LocalDouble() for b in backends}
    runtime = runtime_from_state(state, backends, clients, calibration=artifact,
                                 expected_dataset_hash="dataset-1", tenant_limits={"default": 1})
    return artifact, payload, state, runtime, backends, clients


def test_validated_state_drives_gateway_and_rollback_restores_serving(tmp_path):
    artifact, payload, state, runtime, _backends, _clients = configured(tmp_path)
    configure_runtime(runtime)
    assert TestClient(app).get("/readyz").status_code == 200
    clock = FakeClock()
    controller = RolloutController(state, candidate="candidate", incumbent="incumbent",
                                   task_class="reason", calibration=artifact,
                                   expected_dataset_hash="dataset-1", clock=clock,
                                   on_weights_changed=runtime.router.apply_traffic_weights)
    evidence = OfflineEvidence("candidate", "reason", str(payload["bundle_id"]),
                               bundle_hash(), "dataset-1", 100, 20)
    controller.start(evidence)
    for index in range(20):
        controller.observe(observation_id=f"shadow-{index}", candidate="candidate", task_class="reason",
                           calibration_bundle_id=payload["bundle_id"], dataset_hash="dataset-1", value=0.91)
    controller.tick()
    assert state.weight_map() == runtime.router.traffic_weights()
    assert state.weight_map()["candidate"] == 0.05
    for index in range(20):
        controller.observe(observation_id=f"bad-{index}", candidate="candidate", task_class="reason",
                           calibration_bundle_id=payload["bundle_id"], dataset_hash="dataset-1", value=0.1)
    result = controller.tick()
    assert result["stage"] == "rollback"
    assert state.weight_map() == runtime.router.traffic_weights() == {"candidate": 0, "incumbent": 1, "other": 0}
    response = TestClient(app).post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "calculate 2+2"}]})
    assert response.status_code == 200 and response.headers["x-routebench-backend"] == "incumbent"


def test_wrong_dataset_blocks_bootstrap(tmp_path):
    artifact, _payload, state, _runtime, backends, clients = configured(tmp_path)
    with pytest.raises(GatewayStartupError, match="calibration"):
        runtime_from_state(state, backends, clients, calibration=artifact,
                           expected_dataset_hash="different", tenant_limits={"default": 1})


def test_explicit_tenant_limits_are_required(tmp_path):
    artifact, _payload, state, _runtime, backends, clients = configured(tmp_path)
    with pytest.raises(GatewayStartupError, match="tenant"):
        runtime_from_state(state, backends, clients, calibration=artifact,
                           expected_dataset_hash="dataset-1")


def test_provenance_changes_invalidate_response_namespace(tmp_path):
    _artifact, _payload, _state, runtime, _backends, _clients = configured(tmp_path)
    before = runtime.router.policy_fingerprint()
    changed = runtime.router.quality_bindings | {"dataset_hash": "new-identity"}
    runtime.router.bind_quality(runtime.router.quality, changed)
    assert runtime.router.policy_fingerprint() != before


def test_config_without_calibration_refuses_before_opening_database(tmp_path, monkeypatch):
    from gateway import bootstrap

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"calibration_artifact": "absent.json", "expected_dataset_hash": "dataset-1"}), encoding="utf8")
    monkeypatch.setattr(bootstrap, "ModelState", lambda *_a: pytest.fail("database was opened"))
    with pytest.raises(GatewayStartupError, match="calibration"):
        runtime_from_config(path)


@pytest.mark.parametrize("field,value", [("input_per_1m", -1), ("output_per_1m", math.inf),
                                        ("p95_latency_ms", math.nan), ("max_context", 0),
                                        ("provider", ""), ("model", " ")])
def test_invalid_serving_metadata_refuses_validated_startup(tmp_path, field, value):
    artifact, _payload, state, _runtime, backends, clients = configured(tmp_path)
    setattr(backends[0], field, value)
    with pytest.raises(GatewayStartupError, match="serving metadata"):
        runtime_from_state(state, backends, clients, calibration=artifact,
                           expected_dataset_hash="dataset-1", tenant_limits={"default": 1})


def test_readiness_tracks_static_serving_eligibility_not_stale_mode_flag(tmp_path):
    _artifact, _payload, _state, runtime, _backends, _clients = configured(tmp_path)
    configure_runtime(runtime)
    client = TestClient(app)
    assert client.get("/readyz").status_code == 200
    runtime.router.apply_traffic_weights({b.model: 0 for b in runtime.router.backends})
    assert client.get("/readyz").status_code == 503
    runtime.router.apply_traffic_weights({b.model: 1 for b in runtime.router.backends})
    for backend in runtime.router.backends:
        backend.p95_latency_ms = math.nan
    assert client.get("/readyz").status_code == 503

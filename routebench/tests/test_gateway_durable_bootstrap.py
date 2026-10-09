"""The durable budget ledger wired through validated startup and shutdown (local fixtures only; no provider call)."""
import pytest
from fastapi.testclient import TestClient
from gateway.app import app, configure_runtime
from gateway.bootstrap import GatewayStartupError, runtime_from_state
from gateway.durable_budget import DurableTenantBudget
from gateway.router import Backend
from test_gateway_bootstrap_safety import LocalDouble
from test_rollout_controller import _calibration, _state


def durable_runtime(tmp_path, ledger, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact, payload = _calibration(tmp_path)
    state = _state(tmp_path, str(payload["bundle_id"]))
    state.apply_weight_map({"candidate": 0, "incumbent": 1, "other": 0})
    backends = [Backend(m, "local-test", 0, 0, 1) for m in state.weight_map()]
    clients = {b.model: LocalDouble() for b in backends}
    return runtime_from_state(state, backends, clients, calibration=artifact, expected_dataset_hash="dataset-1",
                              tenant_limits={"default": 1.0}, budget_ledger=ledger, **kw)


def test_validated_startup_builds_durable_budgets_and_recovers_spend_after_restart(tmp_path):
    ledger = tmp_path / "ledger" / "budget.jsonl"
    runtime = durable_runtime(tmp_path / "a", ledger)
    budget = runtime.budgets["default"]
    assert isinstance(budget, DurableTenantBudget) and runtime.budget_ledger is not None
    reservation = budget.reserve(0.4)
    assert reservation is not None and reservation.settle(0.25) is True
    runtime.budget_ledger.close()
    again = durable_runtime(tmp_path / "b", ledger)
    assert again.budgets["default"].spent_usd == pytest.approx(0.25) and again.budgets["default"].remaining() == pytest.approx(0.75)
    again.budget_ledger.close()


def test_a_second_worker_on_the_same_ledger_blocks_startup_unless_takeover_is_explicit(tmp_path):
    ledger = tmp_path / "budget.jsonl"
    first = durable_runtime(tmp_path / "a", ledger)
    with pytest.raises(GatewayStartupError, match="budget ledger blocks startup"):
        durable_runtime(tmp_path / "b", ledger)
    first.budget_ledger.close()
    durable_runtime(tmp_path / "c", ledger).budget_ledger.close()


def test_application_shutdown_releases_the_ledger_lock(tmp_path):
    ledger = tmp_path / "budget.jsonl"
    runtime = durable_runtime(tmp_path / "a", ledger)
    configure_runtime(runtime)
    lock = tmp_path / "budget.jsonl.lock"
    assert lock.exists()
    with TestClient(app):
        assert lock.exists()
    assert not lock.exists()


def test_without_a_file_ledger_validated_startup_uses_the_durable_database_by_default(tmp_path):
    from gateway.budget_store import DbTenantBudget

    runtime = durable_runtime(tmp_path / "a", None)
    assert runtime.budget_ledger is None and isinstance(runtime.budgets["default"], DbTenantBudget) and runtime.budget_mode == "durable_database"


def test_the_native_streaming_flag_flows_through_validated_startup(tmp_path):
    artifact, payload = _calibration(tmp_path)
    state = _state(tmp_path, str(payload["bundle_id"]))
    state.apply_weight_map({"candidate": 0, "incumbent": 1, "other": 0})
    backends = [Backend(m, "local-test", 0, 0, 1) for m in state.weight_map()]
    clients = {b.model: LocalDouble() for b in backends}
    common = {"calibration": artifact, "expected_dataset_hash": "dataset-1", "tenant_limits": {"default": 1.0}}
    assert runtime_from_state(state, backends, clients, native_streaming=True, **common).native_streaming is True
    assert runtime_from_state(state, backends, clients, **common).native_streaming is False  # opt-in only

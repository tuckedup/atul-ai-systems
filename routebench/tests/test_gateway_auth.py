"""Bearer-key tenant authentication (the tenant comes from the key; x-tenant-id can no longer select another tenant). Fake providers only."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from gateway.app import app, build_runtime, configure_runtime
from gateway.auth import AuthConfigError, TenantAuth, hash_key
from gateway.bootstrap import GatewayStartupError, runtime_from_state
from gateway.reliability import TenantBudget
from gateway.router import Backend
from test_gateway_bootstrap_safety import LocalDouble
from test_gateway_spend_safety import FakeBackend
from test_rollout_controller import _calibration, _state

KEY_A = "acme-" + "a" * 40
KEY_B = "beta-" + "b" * 40
BODY = {"messages": [{"role": "user", "content": "hello"}]}


def runtime_with_auth(auth=True):
    backend = FakeBackend(cost_usd=0.0002)
    value = build_runtime([Backend("model-a", "fake", 1.0, 1.0, 1)], {"model-a": backend}, {"model-a": {"chat": 1.0}},
                          budgets={"acme": TenantBudget(1.0), "beta": TenantBudget(1.0)}, default_budget_usd=0.0)
    if auth:
        value.auth = TenantAuth({"acme": hash_key(KEY_A), "beta": hash_key(KEY_B)})
    configure_runtime(value)
    return value, backend


def post(headers=None):
    return TestClient(app).post("/v1/chat/completions", json=BODY, headers=headers or {})


def test_a_valid_key_selects_its_tenant_and_spends_that_tenants_budget():
    value, backend = runtime_with_auth()
    r = post({"Authorization": f"Bearer {KEY_A}"})
    assert r.status_code == 200 and backend.calls == 1
    assert value.budgets["acme"].spent_usd == pytest.approx(0.0002) and value.budgets["beta"].spent_usd == 0


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-" + "x" * 40}, {"Authorization": KEY_A}, {"Authorization": "Basic " + KEY_A}, {"Authorization": "Bearer "}])
def test_missing_or_invalid_credentials_get_401_before_any_provider_call(headers):
    _value, backend = runtime_with_auth()
    r = post(headers)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer" and backend.calls == 0


def test_x_tenant_id_cannot_select_another_tenant_once_auth_is_on():
    value, backend = runtime_with_auth()
    spoof = post({"Authorization": f"Bearer {KEY_A}", "x-tenant-id": "beta"})
    assert spoof.status_code == 403 and backend.calls == 0 and value.budgets["beta"].spent_usd == 0
    assert post({"Authorization": f"Bearer {KEY_A}", "x-tenant-id": "acme"}).status_code == 200
    assert post({"Authorization": f"Bearer {KEY_A}", "x-tenant-id": "default"}).status_code == 200
    assert post({"x-tenant-id": "acme"}).status_code == 401  # a header alone is no longer an identity


def test_models_and_usage_require_a_key_when_auth_is_on_but_health_probes_stay_open():
    runtime_with_auth()
    c = TestClient(app)
    assert c.get("/v1/models").status_code == 401 and c.get("/v1/usage").status_code == 401
    assert c.get("/v1/models", headers={"Authorization": f"Bearer {KEY_B}"}).status_code == 200
    assert c.get("/v1/usage", headers={"Authorization": f"Bearer {KEY_B}"}).status_code == 200
    assert c.get("/healthz").status_code == 200 and c.get("/readyz").status_code in (200, 503)


def test_without_auth_the_legacy_trusted_edge_behaviour_is_unchanged():
    value, _backend = runtime_with_auth(auth=False)
    assert post({"x-tenant-id": "acme"}).status_code == 200 and value.budgets["acme"].spent_usd > 0
    assert post({"x-tenant-id": "ghost"}).status_code == 403
    assert TestClient(app).get("/v1/models").status_code == 200


def test_the_cache_is_not_shared_across_authenticated_tenants():
    _value, _backend = runtime_with_auth()
    assert post({"Authorization": f"Bearer {KEY_A}"}).headers["x-routebench-cache"] == "miss"
    assert post({"Authorization": f"Bearer {KEY_A}"}).headers["x-routebench-cache"] == "hit"
    assert post({"Authorization": f"Bearer {KEY_B}"}).headers["x-routebench-cache"] == "miss"  # another tenant's identical prompt is not served from acme's cache


def test_digest_configuration_is_validated_and_never_holds_a_usable_key():
    good = hash_key(KEY_A)
    assert good.startswith("sha256:") and KEY_A not in good
    for bad in ({}, {"a": "plaintext-key"}, {"a": "sha256:xyz"}, {"": good}, {"a": good, "b": good}):
        with pytest.raises(AuthConfigError):
            TenantAuth(bad)
    with pytest.raises(AuthConfigError):
        hash_key("short")
    auth = TenantAuth({"acme": good})
    assert auth.resolve(f"bearer {KEY_A}") == "acme" and auth.resolve(None) is None and auth.resolve("Bearer") is None


def test_every_digest_is_compared_in_constant_time_style(monkeypatch):
    import hmac

    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr("gateway.auth.hmac.compare_digest", lambda a, b: (calls.append(1), real(a, b))[1])
    auth = TenantAuth({"acme": hash_key(KEY_A), "beta": hash_key(KEY_B)})
    assert auth.resolve(f"Bearer {KEY_A}") == "acme"
    assert len(calls) == 2  # no early exit on the first match


def test_validated_startup_requires_a_limit_for_every_authenticated_tenant(tmp_path):
    artifact, payload = _calibration(tmp_path)
    state = _state(tmp_path, str(payload["bundle_id"]))
    state.apply_weight_map({"candidate": 0, "incumbent": 1, "other": 0})
    backends = [Backend(m, "local-test", 0, 0, 1) for m in state.weight_map()]
    clients = {b.model: LocalDouble() for b in backends}
    common = {"calibration": artifact, "expected_dataset_hash": "dataset-1"}
    with pytest.raises(GatewayStartupError, match="without a spending limit"):
        runtime_from_state(state, backends, clients, tenant_limits={"default": 1}, tenant_key_digests={"ghost": hash_key(KEY_A)}, **common)
    with pytest.raises(GatewayStartupError, match="authentication configuration"):
        runtime_from_state(state, backends, clients, tenant_limits={"default": 1}, tenant_key_digests={"default": "not-a-digest"}, **common)
    ok = runtime_from_state(state, backends, clients, tenant_limits={"default": 1}, tenant_key_digests={"default": hash_key(KEY_A)}, **common)
    assert ok.auth is not None and ok.auth.tenants == {"default"}
    assert runtime_from_state(state, backends, clients, tenant_limits={"default": 1}, **common).auth is None

"""Explicit fail-closed startup from provenance-bound persisted measurements.

Construction performs no model calls. The default application remains unavailable
until an operator supplies real calibration and registry evidence.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from aisys.settings import settings
from evalops.calibrate import CalibrationBundle, validate_bundle
from evalops.taxonomy import canonical

from .auth import AuthConfigError, TenantAuth
from .backends.anthropic import AnthropicBackend
from .backends.openai import OpenAIBackend
from .backends.vllm import VLLMBackend
from .budget_store import BudgetStateError, BudgetStore, DbTenantBudget
from .durable_budget import BudgetLedgerError, DurableBudgetLedger
from .reliability import TenantBudget
from .router import Backend, Router
from .state import ModelState


class GatewayStartupError(ValueError):
    pass


def runtime_from_state(
    state: ModelState, backends: list[Backend], clients: dict[str, Any], *,
    calibration: str | Path, expected_dataset_hash: str,
    tenant_limits: dict[str, float] | None = None,
    budget_ledger: str | Path | None = None, ledger_takeover: bool = False,
    tenant_key_digests: dict[str, str] | None = None, native_streaming: bool = False,
    ephemeral_budgets: bool = False,
) -> Any:
    from .app import build_runtime

    if not expected_dataset_hash:
        raise GatewayStartupError("expected dataset identity is required")
    ok, reasons = validate_bundle(calibration, expected_dataset_hash=expected_dataset_hash)
    if not ok:
        raise GatewayStartupError("calibration blocks gateway startup: " + "; ".join(reasons))
    bundle = CalibrationBundle.load(calibration)
    identities = {
        "calibration_bundle_id": bundle.bundle_id,
        "rubric_bundle_hash": bundle.rubric_bundle_hash,
        "dataset_hash": expected_dataset_hash,
    }
    for backend in backends:
        errors = Router.metadata_errors(backend)
        if errors:
            raise GatewayStartupError(f"invalid backend serving metadata for {backend.model!r}: {errors}")
    names = [b.model for b in backends]
    if not names or len(names) != len(set(names)) or set(names) != set(clients):
        raise GatewayStartupError("backend definitions and clients must have identical unique model IDs")
    if set(names) != set(state.weight_map()):
        raise GatewayStartupError("configured backends differ from the persisted registry")
    quality = state.quality_matrix(**identities, strict_provenance=True)
    records = {row["model"]: row for row in state.rows()}
    normalized: dict[str, dict[str, float]] = {}
    for model, tasks in quality.items():
        for task, score in tasks.items():
            count = records[model]["n_samples"].get(task, 0)
            if isinstance(count, bool) or not isinstance(count, int) or count < 20:
                continue
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise GatewayStartupError("registry contains invalid quality values")
            name = canonical(task)
            if name in normalized.setdefault(model, {}):
                raise GatewayStartupError("duplicate task aliases in measured registry")
            normalized[model][name] = score
    normalized = {m: q for m, q in normalized.items() if q}
    if not normalized:
        raise GatewayStartupError("registry has no adequately sampled, bound quality cells")
    limits = tenant_limits or {}
    if not limits:
        raise GatewayStartupError("operator must configure tenant spending limits explicitly")
    auth = None
    if tenant_key_digests is not None:
        try:
            auth = TenantAuth(tenant_key_digests)
        except AuthConfigError as exc:
            raise GatewayStartupError(f"authentication configuration blocks startup: {exc}") from exc
        unknown = sorted(auth.tenants - set(limits))
        if unknown:
            raise GatewayStartupError(f"authenticated tenants without a spending limit: {unknown}")
    if budget_ledger and ephemeral_budgets:
        raise GatewayStartupError("budget_ledger and ephemeral_budgets are mutually exclusive; choose one explicit budget mode")
    ledger = None
    store = None
    unresolved: list[dict[str, Any]] = []
    if budget_ledger:  # explicit single-host file ledger (kept for deployments without a shared database)
        try:
            ledger = DurableBudgetLedger(budget_ledger, takeover=ledger_takeover)
        except BudgetLedgerError as exc:
            raise GatewayStartupError(f"budget ledger blocks startup: {exc}") from exc
        budgets = {tenant: ledger.budget(tenant, limit) for tenant, limit in limits.items()}
        budget_mode = "durable_file_ledger"
    elif ephemeral_budgets:  # explicit, disclosed opt-out: spend resets on restart
        budgets = {tenant: TenantBudget(limit_usd=limit) for tenant, limit in limits.items()}
        budget_mode = "ephemeral_disclosed"
    else:  # default: budgets live in the single configured database and are verified before serving
        try:
            store = BudgetStore(state)
            for tenant, limit in limits.items():
                store.ensure_tenant(tenant, limit)
            unresolved = store.verify(limits)
        except (BudgetStateError, ValueError) as exc:
            raise GatewayStartupError(f"durable budget state blocks startup: {exc}") from exc
        budgets = {tenant: DbTenantBudget(store, tenant) for tenant in limits}
        budget_mode = "durable_database"
    value = build_runtime(backends, clients, normalized, budgets=budgets, default_budget_usd=0.0)
    value.budget_ledger = ledger
    value.budget_store = store
    value.budget_mode = budget_mode
    value.unresolved_reservations = len(unresolved)
    value.auth = auth
    value.native_streaming = bool(native_streaming)
    value.router.bind_quality(normalized, identities)
    value.router.apply_traffic_weights(state.weight_map())
    if not value.router.has_serving_configuration():
        raise GatewayStartupError("no statically eligible measured backend has positive serving weight")
    value.model_state = state
    value.calibration_identity = identities
    value.evidence_mode = "validated_state"
    return value


def runtime_from_config(path: str | Path) -> Any:
    """Owner-supplied JSON config; API keys are environment references, never serialized."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf8"))
    calibration = Path(data["calibration_artifact"])
    if not calibration.is_absolute():
        calibration = path.parent / calibration
    dataset = data.get("expected_dataset_hash", "")
    # Refuse missing evidence before opening a DSN or constructing provider clients.
    ok, reasons = validate_bundle(calibration, expected_dataset_hash=dataset)
    if not ok:
        raise GatewayStartupError("calibration blocks configuration: " + "; ".join(reasons))
    adapters = {"openai": OpenAIBackend, "anthropic": AnthropicBackend, "vllm": VLLMBackend}
    backends, clients = [], {}
    for record in data["backends"]:
        provider = record["provider"]
        if provider not in adapters:
            raise GatewayStartupError(f"unsupported backend provider {provider!r}")
        metadata = Backend(**{k: record[k] for k in (
            "model", "provider", "input_per_1m", "output_per_1m", "p95_latency_ms")})
        if "max_context" in record:
            metadata.max_context = record["max_context"]
        key_env = record.get("api_key_env")
        key = os.environ.get(key_env, "") if key_env else ""
        clients[metadata.model] = adapters[provider](provider, record["base_url"], key)
        backends.append(metadata)
    state = ModelState(settings.database_url)
    ledger_path = data.get("budget_ledger")
    if ledger_path and not Path(ledger_path).is_absolute():
        ledger_path = path.parent / ledger_path
    return runtime_from_state(state, backends, clients, calibration=calibration,
                              expected_dataset_hash=dataset, tenant_limits=data.get("tenant_limits"),
                              budget_ledger=ledger_path, ledger_takeover=bool(data.get("budget_ledger_takeover", False)),
                              tenant_key_digests=data.get("tenant_key_digests"),
                              native_streaming=bool(data.get("native_streaming", False)),
                              ephemeral_budgets=bool(data.get("ephemeral_budgets", False)))

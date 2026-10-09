"""RouteBench routing policy.

Quality is measured by evalops and is never inferred from a model name or filled in by the
router. Safety constraints are eligibility gates; the quality/cost/latency score only ranks
backends that are actually able and allowed to serve the request.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from aisys import audit, tracing
from prometheus_client import Counter

TOKENS = Counter("routebench_tokens_total", "Accounted tokens; cache hits are separate", ["backend", "direction", "cache_hit"])
COST = Counter("routebench_cost_usd_total", "Conservative cost accounting, not an invoice", ["backend", "basis"])

TaskClass = Literal["classify", "extract", "summarize", "code", "sql", "reason", "tool_use", "chat"]
Difficulty = Literal["easy", "medium", "hard"]

_TASK_CLASSES = frozenset({"classify", "extract", "summarize", "code", "sql", "reason", "tool_use", "chat"})
_DIFFICULTIES = frozenset({"easy", "medium", "hard"})


class NoEligibleBackend(RuntimeError):
    """Raised when every configured backend is excluded from a routing decision."""

    def __init__(self, exclusions: Mapping[str, list[str] | tuple[str, ...]]):
        self.exclusions = {model: tuple(reasons) for model, reasons in exclusions.items()}
        detail = "; ".join(
            f"{model}: {', '.join(reasons)}" for model, reasons in self.exclusions.items()
        )
        super().__init__(f"no eligible backend{': ' + detail if detail else ''}")


@dataclass
class Backend:
    model: str
    provider: str                     # "openai" | "anthropic" | "vllm"
    input_per_1m: float
    output_per_1m: float
    p95_latency_ms: float             # rolling, from telemetry
    circuit_open: bool = False
    traffic_weight: float = 1.0       # relative share among eligible backends
    max_context: int = 128_000


@dataclass
class Request:
    task_class: TaskClass
    difficulty: Difficulty
    est_prompt_tokens: int
    latency_sla_ms: float | None
    tenant_budget_remaining_usd: float
    quality_floor: float = 0.0        # caller can demand a minimum measured quality
    max_output_tokens: int = 500
    request_id: str = ""
    excluded_models: frozenset[str] = field(default_factory=frozenset)


@dataclass
class Decision:
    model: str
    provider: str
    score: float
    reasons: list[str]
    alternatives: list[tuple[str, float]]


class UsageTracker:
    """Thread-safe cumulative token and cost accounting for completed requests."""

    def __init__(self, audit_log: audit.AuditLog):
        self.audit = audit_log
        self._lock = threading.Lock()
        self._total_tokens = 0
        self._total_cost_usd = 0.0
        self._by_model: dict[str, dict[str, int | float]] = {}

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        *,
        cache_hit: bool = False,
        cost_basis: str = "usage_priced",
    ) -> None:
        total_tokens = input_tokens + output_tokens
        event = {
            "type": "gateway_usage",
            "trace_id": tracing.current_trace_id.get(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "cache_hit": cache_hit,
            "cost_basis": cost_basis,
        }
        with self._lock:
            self._total_tokens += total_tokens
            self._total_cost_usd += cost_usd
            model_usage = self._by_model.setdefault(
                model,
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
            )
            model_usage["input_tokens"] += input_tokens
            model_usage["output_tokens"] += output_tokens
            model_usage["total_tokens"] += total_tokens
            model_usage["cost_usd"] += cost_usd
        self.audit.append(event)
        TOKENS.labels(model, "prompt", str(cache_hit).lower()).inc(input_tokens)
        TOKENS.labels(model, "completion", str(cache_hit).lower()).inc(output_tokens)
        COST.labels(model, cost_basis).inc(cost_usd)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_tokens": self._total_tokens,
                "total_cost_usd": self._total_cost_usd,
                "by_model": {model: dict(values) for model, values in self._by_model.items()},
            }


def _finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _fingerprint_value(value: Any) -> Any:
    """Return a deterministic JSON value, including for invalid fail-closed configuration."""
    if isinstance(value, Mapping):
        return {
            str(key): _fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_float": repr(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"invalid_value": repr(value)}


class Router:
    def __init__(
        self,
        quality: dict[str, dict[str, float]],
        backends: list[Backend],
        audit_log: audit.AuditLog,
        w_q: float = 1.0,
        w_c: float = 0.35,
        w_l: float = 0.25,
    ):
        self.quality = quality            # {model: {task_class: measured_quality 0..1}}
        self.backends = backends
        self.audit = audit_log
        self.usage = UsageTracker(audit_log)
        self.w = (w_q, w_c, w_l)
        self._lock = threading.RLock()
        self.quality_bindings: dict[str, str] = {}

    def bind_quality(self, quality: dict[str, dict[str, float]], bindings: Mapping[str, str]) -> None:
        """Install a caller-validated matrix and its identity as one policy snapshot."""
        required = {"calibration_bundle_id", "rubric_bundle_hash", "dataset_hash"}
        if not required <= set(bindings) or not all(isinstance(bindings[k], str) and bindings[k] for k in required):
            raise ValueError("quality publication requires bundle, rubric and dataset bindings")
        with self._lock:
            unknown = set(quality) - {b.model for b in self.backends}
            if unknown:
                raise ValueError(f"quality contains unknown models: {sorted(unknown)}")
            for tasks in quality.values():
                for task, value in tasks.items():
                    if task not in _TASK_CLASSES or not _finite_number(value) or not 0 <= value <= 1:
                        raise ValueError("quality matrix has an invalid task or score")
            self.quality = {model: dict(tasks) for model, tasks in quality.items()}
            self.quality_bindings = dict(bindings)

    def _est_cost(self, b: Backend, r: Request) -> float:
        return (
            r.est_prompt_tokens * b.input_per_1m
            + r.max_output_tokens * b.output_per_1m
        ) / 1e6

    def traffic_weights(self) -> dict[str, float]:
        """Return a consistent snapshot of the active traffic policy."""
        with self._lock:
            return {backend.model: backend.traffic_weight for backend in self.backends}

    def apply_traffic_weights(self, mapping: Mapping[str, float]) -> None:
        """Atomically apply a partial model-to-weight update after validating all entries."""
        with self._lock:
            known = {backend.model for backend in self.backends}
            unknown = set(mapping) - known
            if unknown:
                raise ValueError(f"unknown backend(s): {', '.join(sorted(unknown))}")
            validated: dict[str, float] = {}
            for model, weight in mapping.items():
                if not _finite_number(weight) or weight < 0:
                    raise ValueError(f"invalid traffic weight for {model!r}: {weight!r}")
                validated[model] = float(weight)
            for backend in self.backends:
                if backend.model in validated:
                    backend.traffic_weight = validated[backend.model]

    def policy_fingerprint(self) -> str:
        """Hash policy inputs that can affect routing, excluding live circuit-breaker state."""
        with self._lock:
            policy = {
                "quality": self.quality,
                "quality_bindings": self.quality_bindings,
                "score_weights": self.w,
                "backends": sorted(
                    (
                        {
                            "model": backend.model,
                            "provider": backend.provider,
                            "input_per_1m": backend.input_per_1m,
                            "output_per_1m": backend.output_per_1m,
                            "p95_latency_ms": backend.p95_latency_ms,
                            "traffic_weight": backend.traffic_weight,
                            "max_context": backend.max_context,
                        }
                        for backend in self.backends
                    ),
                    key=lambda item: (item["model"], item["provider"]),
                ),
            }
        encoded = json.dumps(
            _fingerprint_value(policy), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _request_errors(r: Request) -> list[str]:
        errors: list[str] = []
        if not isinstance(r.task_class, str) or r.task_class not in _TASK_CLASSES:
            errors.append("invalid task class")
        if not isinstance(r.difficulty, str) or r.difficulty not in _DIFFICULTIES:
            errors.append("invalid difficulty")
        if (
            not isinstance(r.est_prompt_tokens, int)
            or isinstance(r.est_prompt_tokens, bool)
            or r.est_prompt_tokens < 0
            or not _finite_number(r.est_prompt_tokens)
        ):
            errors.append("invalid prompt token estimate")
        if (
            not isinstance(r.max_output_tokens, int)
            or isinstance(r.max_output_tokens, bool)
            or r.max_output_tokens < 0
            or not _finite_number(r.max_output_tokens)
        ):
            errors.append("invalid output token limit")
        if (
            not _finite_number(r.tenant_budget_remaining_usd)
            or r.tenant_budget_remaining_usd < 0
        ):
            errors.append("invalid tenant budget")
        if r.latency_sla_ms is not None and (
            not _finite_number(r.latency_sla_ms) or r.latency_sla_ms < 0
        ):
            errors.append("invalid latency SLA")
        if not _finite_number(r.quality_floor) or not 0 <= r.quality_floor <= 1:
            errors.append("invalid quality floor")
        if not isinstance(r.request_id, str):
            errors.append("invalid request id")
        if not isinstance(r.excluded_models, frozenset) or not all(
            isinstance(model, str) for model in r.excluded_models
        ):
            errors.append("invalid excluded models")
        return errors

    @staticmethod
    def metadata_errors(b: Backend) -> list[str]:
        """Shared static serving invariants for startup, readiness and routing."""
        reasons: list[str] = []
        if not isinstance(b.model, str) or not b.model.strip():
            reasons.append("invalid model")
        if not isinstance(b.provider, str) or not b.provider.strip():
            reasons.append("invalid provider")
        if not _finite_number(b.traffic_weight) or b.traffic_weight < 0:
            reasons.append("invalid traffic weight")
        if not isinstance(b.circuit_open, bool):
            reasons.append("invalid circuit state")
        if (
            not isinstance(b.max_context, int)
            or isinstance(b.max_context, bool)
            or b.max_context <= 0
        ):
            reasons.append("invalid max context")
        if not _finite_number(b.input_per_1m) or b.input_per_1m < 0:
            reasons.append("invalid input price")
        if not _finite_number(b.output_per_1m) or b.output_per_1m < 0:
            reasons.append("invalid output price")
        if not _finite_number(b.p95_latency_ms) or b.p95_latency_ms < 0:
            reasons.append("invalid latency")
        return reasons

    def has_serving_configuration(self) -> bool:
        """Static readiness, not a provider health probe or guarantee for every task."""
        with self._lock:
            if not all(_finite_number(w) and w >= 0 for w in self.w):
                return False
            for backend in self.backends:
                if self.metadata_errors(backend) or backend.traffic_weight <= 0:
                    continue
                tasks = self.quality.get(backend.model)
                if isinstance(tasks, Mapping) and any(
                    task in _TASK_CLASSES and _finite_number(q) and 0 <= q <= 1
                    for task, q in tasks.items()
                ):
                    return True
        return False

    def _exclusion_reasons(self, b: Backend, r: Request, quality: object) -> list[str]:
        reasons = self.metadata_errors(b)
        if "invalid traffic weight" not in reasons and b.traffic_weight <= 0:
            reasons.append("traffic disabled")
        if isinstance(b.circuit_open, bool) and b.circuit_open:
            reasons.append("circuit open")
        if b.model in r.excluded_models:
            reasons.append("request excluded")
        if "invalid max context" not in reasons and r.est_prompt_tokens + r.max_output_tokens > b.max_context:
            reasons.append("context too long")
        if "invalid latency" not in reasons and r.latency_sla_ms is not None and b.p95_latency_ms > r.latency_sla_ms:
            reasons.append("misses SLA")
        if not _finite_number(quality) or not 0 <= quality <= 1:
            reasons.append("missing or invalid quality")
        elif quality < r.quality_floor:
            reasons.append("below quality floor")
        if not any(reason in reasons for reason in ("invalid input price", "invalid output price")):
            estimated_cost = self._est_cost(b, r)
            if not math.isfinite(estimated_cost) or estimated_cost < 0:
                reasons.append("invalid estimated cost")
            elif estimated_cost > r.tenant_budget_remaining_usd:
                reasons.append("over budget")
        return reasons

    @staticmethod
    def _cohort_backend(scored: list[tuple[Backend, float]], request_id: str) -> Backend:
        ordered = sorted(scored, key=lambda item: (item[0].model, item[0].provider))
        total_weight = sum(item[0].traffic_weight for item in ordered)
        digest = hashlib.sha256(request_id.encode("utf-8")).digest()
        point = (int.from_bytes(digest, "big") / (1 << 256)) * total_weight
        cumulative = 0.0
        for backend, _ in ordered:
            cumulative += backend.traffic_weight
            if point < cumulative:
                return backend
        return ordered[-1][0]  # floating-point boundary protection

    @tracing.traced(kind="router")
    def route(self, r: Request) -> Decision:
        with self._lock:
            backends = [replace(backend) for backend in self.backends]
            quality = {
                model: dict(scores) if isinstance(scores, Mapping) else scores
                for model, scores in self.quality.items()
            }
            score_weights = self.w

        request_errors = self._request_errors(r)
        if request_errors:
            exclusions = {
                backend.model: [f"invalid request: {reason}" for reason in request_errors]
                for backend in backends
            }
            raise NoEligibleBackend(exclusions)

        exclusions: dict[str, list[str]] = {}
        eligible: list[tuple[Backend, float]] = []
        for backend in backends:
            model_quality = quality.get(backend.model)
            measured_quality = (
                model_quality.get(r.task_class) if isinstance(model_quality, Mapping) else None
            )
            reasons = self._exclusion_reasons(backend, r, measured_quality)
            if reasons:
                exclusions[backend.model] = reasons
                continue
            eligible.append((backend, float(measured_quality)))

        if not eligible:
            raise NoEligibleBackend(exclusions)

        w_q, w_c, w_l = score_weights
        if not all(_finite_number(weight) and weight >= 0 for weight in score_weights):
            raise NoEligibleBackend(
                {backend.model: ["invalid score weights"] for backend, _ in eligible}
            )
        costs = {backend.model: self._est_cost(backend, r) for backend, _ in eligible}
        max_cost = max(costs.values()) or 1e-9
        max_latency = max(backend.p95_latency_ms for backend, _ in eligible) or 1e-9
        scored: list[tuple[Backend, float]] = []
        for backend, measured_quality in eligible:
            ranking_quality = measured_quality ** 2 if r.difficulty == "hard" else measured_quality
            score = (
                w_q * ranking_quality
                - w_c * (costs[backend.model] / max_cost)
                - w_l * (backend.p95_latency_ms / max_latency)
            )
            if not math.isfinite(score):
                exclusions[backend.model] = ["non-finite score"]
                continue
            scored.append((backend, score))

        if not scored:
            raise NoEligibleBackend(exclusions)
        scored.sort(key=lambda item: (-item[1], item[0].model, item[0].provider))

        if all(backend.traffic_weight == 1.0 for backend, _ in scored):
            best = scored[0][0]
            decision_reasons = ["best quality/cost/latency tradeoff"]
        else:
            if not math.isfinite(sum(backend.traffic_weight for backend, _ in scored)):
                raise NoEligibleBackend(
                    {backend.model: ["invalid total traffic weight"] for backend, _ in scored}
                )
            best = self._cohort_backend(scored, r.request_id)
            decision_reasons = [
                "deterministic traffic cohort among eligible backends",
                "quality/cost/latency score retained for ranking",
            ]
        score_by_model = {backend.model: score for backend, score in scored}
        alternatives = [
            (backend.model, round(score, 3))
            for backend, score in scored
            if backend.model != best.model
        ][:3]
        decision = Decision(
            best.model,
            best.provider,
            score_by_model[best.model],
            decision_reasons,
            alternatives,
        )
        self.audit.append(
            {
                "type": "router_decision",
                "trace_id": tracing.current_trace_id.get(),
                "task_class": r.task_class,
                "difficulty": r.difficulty,
                "chosen": decision.model,
                "score": round(decision.score, 3),
                "alternatives": decision.alternatives,
                "reasons": decision.reasons,
            }
        )
        return decision

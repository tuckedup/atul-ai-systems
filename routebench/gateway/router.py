"""RouteBench routing policy. Data-driven: quality comes from evalops/offline.py + online.py, not from
hardcoded opinions about model names. Prompt 03 wires this into app.py and state.py.

score(model) = w_q * quality[model][task_class]
             - w_c * normalized_cost
             - w_l * normalized_p95_latency
             - big penalty if circuit open / over budget / SLA impossible
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal

from aisys import audit, tracing

TaskClass = Literal["classify", "extract", "summarize", "code", "sql", "reason", "tool_use", "chat"]
Difficulty = Literal["easy", "medium", "hard"]


@dataclass
class Backend:
    model: str
    provider: str                     # "openai" | "anthropic" | "vllm"
    input_per_1m: float
    output_per_1m: float
    p95_latency_ms: float             # rolling, from telemetry
    circuit_open: bool = False
    traffic_weight: float = 1.0       # canary control: 0.05 / 0.25 / 1.0
    max_context: int = 128_000


@dataclass
class Request:
    task_class: TaskClass
    difficulty: Difficulty
    est_prompt_tokens: int
    latency_sla_ms: float | None
    tenant_budget_remaining_usd: float
    quality_floor: float = 0.0        # caller can demand a minimum measured quality


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

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_tokens": self._total_tokens,
                "total_cost_usd": self._total_cost_usd,
                "by_model": {model: dict(values) for model, values in self._by_model.items()},
            }


class Router:
    def __init__(self, quality: dict[str, dict[str, float]], backends: list[Backend], audit_log: audit.AuditLog,
                 w_q: float = 1.0, w_c: float = 0.35, w_l: float = 0.25):
        self.quality = quality            # {model: {task_class: measured_quality 0..1}}
        self.backends = backends
        self.audit = audit_log
        self.usage = UsageTracker(audit_log)
        self.w = (w_q, w_c, w_l)

    def _est_cost(self, b: Backend, r: Request) -> float:
        return (r.est_prompt_tokens * b.input_per_1m + 500 * b.output_per_1m) / 1e6

    @tracing.traced(kind="router")
    def route(self, r: Request) -> Decision:
        w_q, w_c, w_l = self.w
        max_cost = max(self._est_cost(b, r) for b in self.backends) or 1e-9
        max_lat = max(b.p95_latency_ms for b in self.backends) or 1e-9
        scored: list[tuple[Backend, float, list[str]]] = []
        for b in self.backends:
            reasons = []
            q = self.quality.get(b.model, {}).get(r.task_class, 0.5)
            if r.difficulty == "hard":
                q = q ** 2   # penalize mid-quality models harder when the task is hard
            s = w_q * q - w_c * (self._est_cost(b, r) / max_cost) - w_l * (b.p95_latency_ms / max_lat)
            if b.circuit_open:
                s -= 10; reasons.append("circuit open")
            if r.est_prompt_tokens > b.max_context:
                s -= 10; reasons.append("context too long")
            if r.latency_sla_ms and b.p95_latency_ms > r.latency_sla_ms:
                s -= 5; reasons.append("misses SLA")
            if self._est_cost(b, r) > r.tenant_budget_remaining_usd:
                s -= 10; reasons.append("over budget")
            if q < r.quality_floor:
                s -= 5; reasons.append("below quality floor")
            s *= min(1.0, b.traffic_weight)   # canary: shrink share, don't exclude
            scored.append((b, s, reasons))
        scored.sort(key=lambda t: t[1], reverse=True)
        best, score, reasons = scored[0]
        d = Decision(best.model, best.provider, score, reasons or ["best quality/cost/latency tradeoff"],
                     [(b.model, round(s, 3)) for b, s, _ in scored[1:4]])
        self.audit.append({"type": "router_decision", "trace_id": tracing.current_trace_id.get(),
                           "task_class": r.task_class, "difficulty": r.difficulty, "chosen": d.model,
                           "score": round(score, 3), "alternatives": d.alternatives, "reasons": d.reasons})
        return d

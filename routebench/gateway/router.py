"""RouteBench routing policy. Data-driven: quality comes from evalops/offline.py + online.py, not from
hardcoded opinions about model names. Prompt 03 wires this into app.py and state.py.

score(model) = w_q * quality[model][task_class]
             - w_c * normalized_cost
             - w_l * normalized_p95_latency
             - big penalty if circuit open / over budget / SLA impossible
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aisys import audit, tracing
from aisys.settings import settings

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


class Router:
    def __init__(self, quality: dict[str, dict[str, float]], backends: list[Backend], audit_log: audit.AuditLog,
                 w_q: float = 1.0, w_c: float = 0.35, w_l: float = 0.25):
        self.quality = quality            # {model: {task_class: measured_quality 0..1}}
        self.backends = backends
        self.audit = audit_log
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
            s *= b.traffic_weight if b.traffic_weight < 1.0 else 1.0   # canary: shrink share, don't exclude
            scored.append((b, s, reasons))
        scored.sort(key=lambda t: t[1], reverse=True)
        best, score, reasons = scored[0]
        d = Decision(best.model, best.provider, score, reasons or ["best quality/cost/latency tradeoff"],
                     [(b.model, round(s, 3)) for b, s, _ in scored[1:4]])
        self.audit.append({"type": "router_decision", "trace_id": tracing.current_trace_id.get(),
                           "task_class": r.task_class, "difficulty": r.difficulty, "chosen": d.model,
                           "score": round(score, 3), "alternatives": d.alternatives, "reasons": d.reasons})
        return d

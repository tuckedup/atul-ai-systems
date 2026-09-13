"""RouteBench routing policy. Data-driven: quality comes from evalops/offline.py + online.py, not from
hardcoded opinions about model names. Prompt 03 wires this into app.py and state.py.

score(model) = w_q * quality[model][task_class]
             - w_c * normalized_cost
             - w_l * normalized_p95_latency
             - big penalty if circuit open / over budget / SLA impossible
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from aisys import audit, tracing
from aisys.settings import settings
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .backends import BackendAdapter, BackendHTTPError, BackendResult

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


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP gateway
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


@dataclass(frozen=True)
class RequestCost:
    """The successful route's billable usage, keyed by gateway request ID."""

    request_id: str
    trace_id: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    attempts: int
    fallbacks: int


class CostTracker:
    """Small in-memory ledger for per-request accounting.

    Production storage can implement the same ``record``/``get`` boundary. A
    lock keeps the mock gateway deterministic under concurrent TestClient or
    development-server requests.
    """

    def __init__(self) -> None:
        self._records: dict[str, RequestCost] = {}
        self._lock = threading.Lock()

    def record(self, cost: RequestCost) -> None:
        with self._lock:
            self._records[cost.request_id] = cost

    def get(self, request_id: str) -> RequestCost | None:
        with self._lock:
            return self._records.get(request_id)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def summary(self) -> dict[str, Any]:
        """Aggregate completed request usage by the model that served it."""

        with self._lock:
            records = tuple(self._records.values())
        by_model: dict[str, dict[str, int | float | str]] = defaultdict(
            lambda: {
                "model": "",
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        )
        for record in records:
            item = by_model[record.model]
            item["model"] = record.model
            item["requests"] += 1
            item["prompt_tokens"] += record.prompt_tokens
            item["completion_tokens"] += record.completion_tokens
            item["total_tokens"] += record.prompt_tokens + record.completion_tokens
            item["cost_usd"] += record.cost_usd
        data = sorted(by_model.values(), key=lambda item: str(item["model"]))
        return {
            "object": "usage.summary",
            "data": data,
            "total": {
                "requests": len(records),
                "prompt_tokens": sum(record.prompt_tokens for record in records),
                "completion_tokens": sum(
                    record.completion_tokens for record in records
                ),
                "total_tokens": sum(
                    record.prompt_tokens + record.completion_tokens
                    for record in records
                ),
                "cost_usd": sum(record.cost_usd for record in records),
            },
        }


class CircuitBreaker:
    """Rolling failure-window circuit breaker, isolated per configured backend."""

    def __init__(self, threshold: int = 3, window_seconds: float = 30.0) -> None:
        if threshold < 1 or window_seconds <= 0:
            raise ValueError("circuit breaker threshold and window must be positive")
        self.threshold = threshold
        self.window_seconds = window_seconds
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, failures: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while failures and failures[0] < cutoff:
            failures.popleft()

    def record_failure(self, backend_key: str, now: float | None = None) -> None:
        observed_at = time.monotonic() if now is None else now
        with self._lock:
            failures = self._failures[backend_key]
            self._prune(failures, observed_at)
            failures.append(observed_at)

    def record_success(self, backend_key: str) -> None:
        with self._lock:
            self._failures.pop(backend_key, None)

    def is_open(self, backend_key: str, now: float | None = None) -> bool:
        observed_at = time.monotonic() if now is None else now
        with self._lock:
            failures = self._failures[backend_key]
            self._prune(failures, observed_at)
            return len(failures) >= self.threshold


@dataclass
class GatewayRuntime:
    """Ordered backends plus request-scoped accounting state."""

    backends: list[BackendAdapter]
    costs: CostTracker = field(default_factory=CostTracker)
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    concurrency_limit: int = 8
    queue_timeout_seconds: float = 5.0
    _capacity: threading.BoundedSemaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.concurrency_limit < 1 or self.queue_timeout_seconds < 0:
            raise ValueError("concurrency limit must be positive and timeout non-negative")
        self._capacity = threading.BoundedSemaphore(self.concurrency_limit)

    def acquire_capacity(self) -> bool:
        return self._capacity.acquire(timeout=self.queue_timeout_seconds)

    def release_capacity(self) -> None:
        self._capacity.release()


runtime = GatewayRuntime(backends=[])
app = FastAPI(title="RouteBench Gateway", version="0.1.0")


def configure_runtime(
    backends: list[BackendAdapter],
    *,
    concurrency_limit: int = 8,
    queue_timeout_seconds: float = 5.0,
) -> GatewayRuntime:
    """Replace the ordered backend chain; primarily used by tests and startup."""

    global runtime
    runtime = GatewayRuntime(
        backends=list(backends),
        concurrency_limit=concurrency_limit,
        queue_timeout_seconds=queue_timeout_seconds,
    )
    return runtime


@tracing.traced(kind="router", name="routebench.chat.completions")
def _route_to_backend(
    payload: dict[str, Any], *, stream: bool, gateway: GatewayRuntime | None = None
) -> tuple[BackendResult, BackendAdapter, int]:
    active = gateway or runtime
    if not active.backends:
        raise HTTPException(status_code=503, detail="no RouteBench backends configured")

    last_error: BackendHTTPError | None = None
    for fallback_count, backend in enumerate(active.backends):
        backend_key = f"{backend.name}:{backend.model}"
        if active.circuit_breaker.is_open(backend_key):
            last_error = BackendHTTPError(503, f"circuit open for {backend_key}")
            continue
        provider_payload = dict(payload)
        # The externally requested model may be "auto". Each adapter receives
        # its concrete configured model, while the response reports the model
        # that actually served the request.
        provider_payload["model"] = backend.model
        try:
            result = backend.chat(provider_payload, stream=stream)
            active.circuit_breaker.record_success(backend_key)
            return result, backend, fallback_count
        except BackendHTTPError as error:
            last_error = error
            # Fallback is deliberately limited to upstream 5xx failures. A
            # 4xx means the caller's request is invalid and trying another
            # provider would hide the real error.
            if error.status_code < 500:
                raise HTTPException(
                    status_code=error.status_code, detail=str(error)
                ) from error
            active.circuit_breaker.record_failure(backend_key)
        except (ConnectionError, TimeoutError) as error:
            last_error = BackendHTTPError(503, str(error))
            active.circuit_breaker.record_failure(backend_key)

    detail = str(last_error) if last_error else "all RouteBench backends failed"
    raise HTTPException(status_code=502, detail=detail)


def _completion_body(request_id: str, result: BackendResult) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
    }


def _stream_events(request_id: str, result: BackendResult):
    created = int(time.time())
    for index, text in enumerate(result.chunks):
        delta: dict[str, str] = {"content": text}
        if index == 0:
            delta["role"] = "assistant"
        event = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": result.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
    final_event = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": result.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
    }
    yield f"data: {json.dumps(final_event, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionRequest):
    active = runtime
    if not active.acquire_capacity():
        raise HTTPException(
            status_code=429,
            detail="RouteBench request queue timed out; retry later",
            headers={"Retry-After": "1"},
        )
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    trace_id = tracing.new_trace_id()
    try:
        payload = body.model_dump()
        result, backend, fallback_count = _route_to_backend(
            payload, stream=body.stream, gateway=active
        )
        cost = RequestCost(
            request_id=request_id,
            trace_id=trace_id,
            provider=backend.name,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            attempts=result.attempts,
            fallbacks=fallback_count,
        )
        active.costs.record(cost)
    finally:
        active.release_capacity()
    headers = {
        "x-routebench-request-id": request_id,
        "x-routebench-trace-id": trace_id,
        "x-routebench-provider": backend.name,
        "x-routebench-cost-usd": f"{result.cost_usd:.10f}",
    }
    if body.stream:
        return StreamingResponse(
            _stream_events(request_id, result),
            media_type="text/event-stream",
            headers=headers,
        )
    return JSONResponse(_completion_body(request_id, result), headers=headers)


@app.get("/internal/costs/{request_id}")
def request_cost(request_id: str) -> dict[str, Any]:
    cost = runtime.costs.get(request_id)
    if cost is None:
        raise HTTPException(status_code=404, detail="request cost not found")
    return asdict(cost)


@app.get("/v1/usage")
def usage() -> dict[str, Any]:
    return runtime.costs.summary()


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}

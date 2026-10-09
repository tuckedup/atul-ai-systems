"""FastAPI application implementing the OpenAI chat-completions surface."""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from aisys.audit import default_audit_log
from aisys.cache import create_cache
from aisys.llm import ChatResult
from aisys.settings import settings
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .backends import BackendClient
from .budget_store import BudgetStateError
from .cache import ResponseCache
from .classifier import Classifier
from .observability import RequestTraceMiddleware, record_completion
from .reliability import AdmissionController, CircuitBreaker, TenantBudget
from .router import Backend, NoEligibleBackend, Request, Router

REQUESTS = Counter("routebench_requests_total", "Gateway requests", ["backend", "status"])
LATENCY = Histogram("routebench_request_latency_seconds", "Gateway latency", ["backend"])
CACHE_REQUESTS = Counter("routebench_cache_requests_total", "Cache lookups")
CACHE_HITS = Counter("routebench_cache_hits_total", "Cache hits")


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | None = None

    @model_validator(mode="after")
    def supported_message_fields(self) -> Message:
        unknown = set(self.model_extra or {}) - {"name", "tool_calls", "tool_call_id", "function_call", "refusal"}
        if unknown:
            raise ValueError("unsupported message fields: " + ", ".join(sorted(unknown)))
        return self


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[Message]
    stream: bool = False
    max_tokens: int | None = Field(default=None, gt=0, strict=True)
    max_completion_tokens: int | None = Field(default=None, gt=0, strict=True)
    temperature: float = 0.0
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def one_output_limit(self) -> ChatRequest:
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("specify only one of max_tokens and max_completion_tokens")
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        try:
            json.dumps(self.model_dump(), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("request fields must be finite JSON values") from error
        extra = self.model_extra or {}
        supported = {"n", "best_of", "audio", "modalities", "web_search_options", "service_tier",
                     "stop", "top_p", "logit_bias", "presence_penalty", "frequency_penalty", "seed",
                     "response_format", "parallel_tool_calls", "tool_choice", "stream_options",
                     "logprobs", "top_logprobs", "user", "store", "reasoning_effort",
                     "functions", "function_call"}
        unknown = set(extra) - supported
        if unknown:
            raise ValueError("unsupported request fields: " + ", ".join(sorted(unknown)))
        if extra.get("service_tier", "default") != "default":
            raise ValueError("only default service_tier has configured bounded pricing")
        if self.tools and any(tool.get("type") != "function" for tool in self.tools):
            raise ValueError("only local function tool declarations are supported")
        for name in ("n", "best_of"):
            value = extra.get(name, 1)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            if value > 1:
                raise ValueError(f"{name} > 1 is unsupported because its spend is not bounded")
        unsupported_paid = [
            name
            for name in ("audio", "modalities", "web_search_options")
            if extra.get(name) is not None
        ]
        if unsupported_paid:
            raise ValueError(
                "unsupported paid feature(s): " + ", ".join(unsupported_paid)
            )
        return self


@dataclass
class Runtime:
    router: Router
    clients: dict[str, BackendClient]
    classifier: Classifier
    response_cache: ResponseCache
    admission: AdmissionController
    breakers: dict[str, CircuitBreaker]
    budgets: dict[str, TenantBudget]
    model_state: Any = None
    calibration_identity: dict[str, str] = field(default_factory=dict)
    evidence_mode: str = "unverified_configuration"
    budget_ledger: Any = None  # DurableBudgetLedger when budgets are restart-safe; closed on shutdown
    auth: Any = None  # TenantAuth when bearer-key authentication is configured; None = trusted-edge x-tenant-id (legacy)
    native_streaming: bool = False  # opt-in: true token streaming when every backend client implements stream_chat
    budget_mode: str = "in_memory"  # durable_database | durable_file_ledger | ephemeral_disclosed | in_memory (library/test construction)
    budget_store: Any = None
    unresolved_reservations: int = 0  # reservations held against caps awaiting an explicit operator decision


def build_runtime(
    backends: list[Backend],
    clients: dict[str, BackendClient],
    quality: dict[str, dict[str, float]],
    *,
    budgets: dict[str, TenantBudget] | None = None,
    default_budget_usd: float = 100.0,
) -> Runtime:
    cache = create_cache(settings.cache_backend)
    configured_budgets = {"default": TenantBudget(limit_usd=default_budget_usd)}
    if budgets:
        configured_budgets.update(budgets)
    return Runtime(
        router=Router(quality, backends, default_audit_log()),
        clients=clients,
        classifier=Classifier(cache),
        response_cache=ResponseCache(cache),
        admission=AdmissionController(50, queue_timeout_s=0.1),
        breakers={backend.model: CircuitBreaker() for backend in backends},
        budgets=configured_budgets,
    )


runtime = build_runtime([], {}, {})


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    config = os.environ.get("ROUTEBENCH_RUNTIME_CONFIG")
    if config:
        from .bootstrap import runtime_from_config

        configure_runtime(runtime_from_config(config))
    try:
        yield
    finally:
        ledger = getattr(runtime, "budget_ledger", None)
        if ledger is not None:
            ledger.close()


app = FastAPI(title="RouteBench", version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestTraceMiddleware)


@app.exception_handler(BudgetStateError)
async def _budget_state_unavailable(_request: Any, error: BudgetStateError) -> Response:
    """The durable budget state could not be read or written: refuse (fail closed). No provider call is made or repeated because of this."""
    return Response(json.dumps({"error": {"message": "budget accounting unavailable; request refused", "type": "budget_state_error"}}),
                    status_code=503, media_type="application/json")


def configure_runtime(value: Runtime) -> None:
    global runtime
    runtime = value


def _wire(result: ChatResult, request_id: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.text}
    if result.tool_calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in result.tool_calls
        ]
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if result.tool_calls else "stop"}],
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total,
        },
    }


def _output_cap(body: ChatRequest) -> int:
    return body.max_completion_tokens or body.max_tokens or 2048


def _forward_payload(body: ChatRequest) -> dict[str, Any]:
    # Keep OpenAI fields this gateway does not interpret (including message tool
    # history) instead of reconstructing a lossy subset.  The current backend
    # contract consumes a completed JSON response, so streaming is buffered at
    # this gateway and must be disabled on the upstream request.
    payload = body.model_dump(exclude={"metadata"}, exclude_unset=True)
    payload["stream"] = False
    payload.setdefault("temperature", body.temperature)
    for name in ("max_tokens", "max_completion_tokens"):
        if payload.get(name) is None:
            payload.pop(name, None)
    if "max_tokens" not in payload and "max_completion_tokens" not in payload:
        payload["max_tokens"] = _output_cap(body)
    return payload


def _prompt_token_bound(payload: dict[str, Any]) -> int:
    # One token per UTF-8 byte is intentionally conservative for text, JSON
    # framing, tools, and tool-call history.  It avoids the old len(text)//4
    # underestimate without depending on a provider-specific tokenizer.
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    # The fixed allowance covers provider-side chat framing that is not present
    # in the JSON body (role/name delimiters and request envelope fields).
    return max(1, len(encoded) + 1024)


def _metadata_number(body: ChatRequest, name: str, default: float | None) -> float | None:
    raw = body.metadata.get(name, default)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise HTTPException(422, f"metadata.{name} must be numeric") from error
    if not math.isfinite(value) or value < 0:
        raise HTTPException(422, f"metadata.{name} must be finite and non-negative")
    return value


def _call_bound(metadata: Backend, prompt_tokens: int, output_tokens: int) -> float:
    values = (metadata.input_per_1m, metadata.output_per_1m)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"invalid pricing for backend {metadata.model}")
    return (prompt_tokens * metadata.input_per_1m + output_tokens * metadata.output_per_1m) / 1e6


def _result_cost(result: ChatResult, metadata: Backend) -> float | None:
    # Adapters retain the provider response. Missing raw usage must not turn
    # default Usage(0, 0) into a supposedly free billable request.
    if result.raw:
        usage = result.raw.get("usage")
        if (not isinstance(usage, dict)
                or any(isinstance(usage.get(k), bool) or not isinstance(usage.get(k), int)
                       or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens"))
                or usage["prompt_tokens"] == 0):
            return None
    if result.usage.prompt_tokens < 0 or result.usage.completion_tokens < 0:
        raise ValueError("provider returned negative token usage")
    # The adapter's generic pricing table may be stale or may not know a backend.
    # Real raw usage is priced from the same operator-configured schedule as admission.
    charge = None if result.raw else result.cost_usd
    if charge is None:
        charge = (
            result.usage.prompt_tokens * metadata.input_per_1m
            + result.usage.completion_tokens * metadata.output_per_1m
        ) / 1e6
    charge = float(charge)
    if not math.isfinite(charge) or charge < 0:
        raise ValueError("provider returned invalid cost")
    return charge


def _record_unknown_attempt(model: str, reserved_usd: float) -> None:
    runtime.router.usage.record(model, 0, 0, reserved_usd, cost_basis="unknown_bound")


def _complete(body: ChatRequest, tenant: str, budget: TenantBudget, request_id: str) -> tuple[dict[str, Any], str]:
    prompt = "\n".join(message.content or "" for message in body.messages)
    classification = runtime.classifier.classify(prompt)
    if not runtime.router.backends:
        raise HTTPException(503, "no backends configured")
    if budget.remaining() <= 0:
        raise HTTPException(402, "tenant budget exhausted")
    forwarded = _forward_payload(body)
    prompt_tokens = max(
        _prompt_token_bound({**forwarded, "model": item.model})
        for item in runtime.router.backends
    )
    output_tokens = _output_cap(body)
    latency_sla = _metadata_number(body, "latency_sla_ms", None)
    quality_floor = _metadata_number(body, "quality_floor", 0.0)
    attempted: set[str] = set()
    last_error = "no available backend"
    while True:
        for metadata in runtime.router.backends:
            metadata.circuit_open = runtime.breakers[metadata.model].is_open
        try:
            decision = runtime.router.route(Request(
                task_class=classification.task_class,
                difficulty=classification.difficulty,
                est_prompt_tokens=prompt_tokens,
                latency_sla_ms=latency_sla,
                tenant_budget_remaining_usd=budget.remaining(),
                quality_floor=quality_floor or 0.0,
                max_output_tokens=output_tokens,
                request_id=request_id,
                excluded_models=frozenset(attempted),
            ))
        except NoEligibleBackend as error:
            if not attempted:
                affordable = False
                for item in runtime.router.backends:
                    try:
                        affordable = (
                            _call_bound(item, prompt_tokens, output_tokens)
                            <= budget.remaining() + 1e-12
                        )
                    except (TypeError, ValueError):
                        continue
                    if affordable:
                        break
                if not affordable:
                    raise HTTPException(402, "tenant budget exhausted") from error
            detail = f"all backends failed: {last_error}" if attempted else "no eligible backend"
            raise HTTPException(503, detail) from error
        if decision.model in attempted:
            # Defensive guard for custom Router implementations violating the
            # exclusion contract; never allow an unbounded retry loop.
            raise HTTPException(503, f"all backends failed: {last_error}")
        attempted.add(decision.model)
        metadata = next((item for item in runtime.router.backends if item.model == decision.model), None)
        client = runtime.clients.get(decision.model)
        if metadata is None or client is None:
            last_error = f"backend {decision.model} is not configured"
            continue
        try:
            reserved_usd = _call_bound(metadata, prompt_tokens, output_tokens)
        except ValueError as error:
            last_error = str(error)
            continue
        reservation = budget.reserve(reserved_usd)
        if reservation is None:
            # Another in-flight request may have claimed the last funds after
            # routing.  Admission is rechecked atomically immediately before IO.
            raise HTTPException(402, "tenant budget exhausted")
        payload = dict(forwarded)
        payload["model"] = decision.model
        started = time.perf_counter()
        try:
            result = client.chat(payload)
        except httpx.HTTPStatusError as error:
            charged = reservation.amount_usd
            reservation.settle()
            _record_unknown_attempt(decision.model, charged)
            REQUESTS.labels(decision.model, "error").inc()
            status = error.response.status_code
            if 400 <= status < 500:
                # Client/provider rejection is deterministic for this request:
                # do not poison the circuit and do not send it to another model.
                exposed = status if status in {400, 404, 409, 413, 422, 429} else 400
                raise HTTPException(exposed, "provider rejected request") from error
            last_error = f"provider HTTP {status}"
            runtime.breakers[decision.model].failure()
        except Exception as error:  # noqa: BLE001 - backend failures trigger circuit rerouting
            charged = reservation.amount_usd
            reservation.settle()
            _record_unknown_attempt(decision.model, charged)
            last_error = str(error)
            runtime.breakers[decision.model].failure()
            REQUESTS.labels(decision.model, "error").inc()
        else:
            try:
                charge = _result_cost(result, metadata)
            except (TypeError, ValueError) as error:
                charged = reservation.amount_usd
                reservation.settle()
                _record_unknown_attempt(decision.model, charged)
                last_error = str(error)
                runtime.breakers[decision.model].failure()
                REQUESTS.labels(decision.model, "error").inc()
                continue
            within_bound = reservation.settle(charge)
            accounted_cost = reservation.amount_usd if charge is None else charge
            runtime.router.usage.record(
                decision.model,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                accounted_cost,
                cost_basis="unknown_bound" if charge is None else "usage_priced",
            )
            runtime.breakers[decision.model].success()
            REQUESTS.labels(decision.model, "ok").inc()
            LATENCY.labels(decision.model).observe(time.perf_counter() - started)
            if not within_bound:
                raise HTTPException(402, "tenant budget bound exceeded; actual cost recorded")
            wire = _wire(result, request_id)
            if charge is None:
                wire["usage"] = None
            record_completion(wire, decision.model, accounted_cost)
            return wire, decision.model


def _authenticated_tenant(authorization: str | None, x_tenant_id: str | None) -> str | None:
    """With TenantAuth configured, the tenant comes from the bearer key and a conflicting x-tenant-id is refused. Without it, returns None (legacy trusted-edge mode)."""
    auth = getattr(runtime, "auth", None)
    if auth is None:
        return None
    tenant = auth.resolve(authorization)
    if tenant is None:
        raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
    claimed = (x_tenant_id or "").strip()
    if claimed and claimed not in ("default", tenant):
        raise HTTPException(403, "x-tenant-id does not match the authenticated tenant")
    return tenant


@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest, x_tenant_id: str = Header("default"), authorization: str | None = Header(None)) -> Response:
    # Without TenantAuth, x-tenant-id is a trusted identity assertion supplied by
    # the deployment's authenticated edge/proxy (budget/cache isolation, not
    # authentication).  With TenantAuth configured, the tenant is derived from
    # the bearer key and the header can no longer select another tenant.
    authenticated = _authenticated_tenant(authorization, x_tenant_id)
    tenant = authenticated if authenticated is not None else x_tenant_id.strip()
    if not tenant:
        raise HTTPException(400, "x-tenant-id must not be empty")
    budget = runtime.budgets.get(tenant)
    if budget is None:
        raise HTTPException(403, "unknown tenant")
    if not runtime.admission.acquire():
        raise HTTPException(429, "gateway queue full", headers={"Retry-After": "1"})
    release_here = True
    try:
        cache_payload = body.model_dump()
        policy_fingerprint = runtime.router.policy_fingerprint()
        if not body.stream:
            CACHE_REQUESTS.inc()
        cached_entry = runtime.response_cache.get(
            cache_payload,
            tenant=tenant,
            policy_fingerprint=policy_fingerprint,
        ) if not body.stream else None
        if cached_entry is not None and runtime.router.policy_fingerprint() != policy_fingerprint:
            # A rollout changed during lookup.  Bypass this entry rather than
            # serving a response selected under the superseded policy.
            cached_entry = None
        cached = dict(cached_entry) if cached_entry is not None else None
        if cached is not None:
            CACHE_HITS.inc()
            backend = str(cached.pop("_routebench_backend", cached["model"]))
            cached_usage = cached["usage"]
            runtime.router.usage.record(
                backend,
                int(cached_usage["prompt_tokens"]),
                int(cached_usage["completion_tokens"]),
                0.0,
                cache_hit=True,
            )
            record_completion(cached, backend, 0.0, cache_hit=True)
            return Response(
                json.dumps(cached),
                media_type="application/json",
                headers={"x-routebench-cache": "hit", "x-routebench-backend": backend},
            )
        request_id = "chatcmpl-" + uuid.uuid4().hex
        if body.stream:
            from . import streaming

            if streaming.native_streaming_available():
                opened = streaming.open_stream(body, tenant, budget, request_id)
                release_here = False  # the stream generator releases admission when it ends or is closed
                return StreamingResponse(
                    opened.events(runtime.admission.release),
                    media_type="text/event-stream",
                    headers={"x-routebench-backend": opened.backend, "x-routebench-cache": "miss"},
                )
        wire, backend = _complete(body, tenant, budget, request_id)
        if body.stream:
            def events() -> Any:
                try:
                    chunk = {
                        "id": wire["id"], "object": "chat.completion.chunk", "created": wire["created"],
                        "model": wire["model"], "choices": [{"index": 0, "delta": wire["choices"][0]["message"], "finish_reason": None}],
                    }
                    terminal = {
                        "id": wire["id"], "object": "chat.completion.chunk", "created": wire["created"],
                        "model": wire["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": wire["choices"][0]["finish_reason"]}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield f"data: {json.dumps(terminal)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    runtime.admission.release()
            release_here = False
            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={"x-routebench-backend": backend, "x-routebench-cache": "miss"},
            )
        cached_wire = dict(wire)
        cached_wire["_routebench_backend"] = backend
        if wire["usage"] is not None and runtime.router.policy_fingerprint() == policy_fingerprint:
            runtime.response_cache.set(
                cache_payload,
                cached_wire,
                tenant=tenant,
                policy_fingerprint=policy_fingerprint,
            )
        return Response(
            json.dumps(wire),
            media_type="application/json",
            headers={"x-routebench-backend": backend, "x-routebench-cache": "miss"},
        )
    finally:
        if release_here:
            runtime.admission.release()


@app.get("/v1/models")
def models(authorization: str | None = Header(None)) -> dict[str, Any]:
    _authenticated_tenant(authorization, None)
    return {"object": "list", "data": [{"id": item.model, "object": "model", "owned_by": item.provider} for item in runtime.router.backends]}


@app.get("/v1/usage")
def usage(authorization: str | None = Header(None)) -> dict[str, Any]:
    _authenticated_tenant(authorization, None)
    return runtime.router.usage.snapshot()


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readiness() -> Response:
    ready = bool(runtime.evidence_mode == "validated_state" and runtime.router.has_serving_configuration())
    unresolved: int | None = runtime.unresolved_reservations
    if getattr(runtime, "budget_store", None) is not None:
        try:
            unresolved = len(runtime.budget_store.unresolved())  # live: also proves the budget database is reachable
        except BudgetStateError:
            unresolved, ready = None, False
    return Response(json.dumps({"ready": ready, "evidence_mode": runtime.evidence_mode,
                                "configured_backends": len(runtime.router.backends),
                                "budget_mode": runtime.budget_mode,
                                "unresolved_budget_reservations": unresolved}),
                    status_code=200 if ready else 503, media_type="application/json")


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

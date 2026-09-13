"""FastAPI application implementing the OpenAI chat-completions surface."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from aisys.audit import default_audit_log
from aisys.cache import create_cache
from aisys.llm import ChatResult
from aisys.settings import settings
from aisys.tracing import new_trace_id
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .backends import BackendClient
from .cache import ResponseCache
from .classifier import Classifier
from .reliability import AdmissionController, CircuitBreaker, TenantBudget
from .router import Backend, Request, Router

REQUESTS = Counter("routebench_requests_total", "Gateway requests", ["backend", "status"])
LATENCY = Histogram("routebench_request_latency_seconds", "Gateway latency", ["backend"])


class Message(BaseModel):
    role: str
    content: str | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    stream: bool = False
    max_tokens: int = 2048
    temperature: float = 0.0
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Runtime:
    router: Router
    clients: dict[str, BackendClient]
    classifier: Classifier
    response_cache: ResponseCache
    admission: AdmissionController
    breakers: dict[str, CircuitBreaker]
    budgets: dict[str, TenantBudget]


def build_runtime(backends: list[Backend], clients: dict[str, BackendClient], quality: dict[str, dict[str, float]]) -> Runtime:
    cache = create_cache(settings.cache_backend)
    return Runtime(
        router=Router(quality, backends, default_audit_log()),
        clients=clients,
        classifier=Classifier(cache),
        response_cache=ResponseCache(cache),
        admission=AdmissionController(64),
        breakers={backend.model: CircuitBreaker() for backend in backends},
        budgets={},
    )


runtime = build_runtime([], {}, {})
app = FastAPI(title="RouteBench", version="0.1.0")


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
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total,
        },
    }


def _complete(body: ChatRequest, tenant: str) -> tuple[dict[str, Any], str]:
    prompt = "\n".join(message.content or "" for message in body.messages)
    classification = runtime.classifier.classify(prompt)
    budget = runtime.budgets.setdefault(tenant, TenantBudget(limit_usd=100.0))
    if not runtime.router.backends:
        raise HTTPException(503, "no backends configured")
    for metadata in runtime.router.backends:
        metadata.circuit_open = runtime.breakers[metadata.model].is_open
    attempted: set[str] = set()
    last_error = "no available backend"
    while len(attempted) < len(runtime.router.backends):
        decision = runtime.router.route(Request(
            classification.task_class,
            classification.difficulty,
            max(1, len(prompt) // 4),
            body.metadata.get("latency_sla_ms"),
            budget.remaining(),
            float(body.metadata.get("quality_floor", 0)),
        ))
        if decision.model in attempted or runtime.breakers[decision.model].is_open:
            metadata = next(item for item in runtime.router.backends if item.model == decision.model)
            metadata.circuit_open = True
            attempted.add(decision.model)
            continue
        attempted.add(decision.model)
        client = runtime.clients[decision.model]
        payload = body.model_dump(exclude={"metadata"}, exclude_none=True)
        payload["model"] = decision.model
        started = time.perf_counter()
        try:
            result = client.chat(payload)
            runtime.breakers[decision.model].success()
            charge = result.cost_usd or 0.0
            if not budget.charge(charge):
                raise HTTPException(402, "tenant budget exhausted")
            REQUESTS.labels(decision.model, "ok").inc()
            LATENCY.labels(decision.model).observe(time.perf_counter() - started)
            return _wire(result, "chatcmpl-" + uuid.uuid4().hex), decision.model
        except HTTPException:
            raise
        except Exception as error:  # noqa: BLE001 - backend failures trigger circuit rerouting
            last_error = str(error)
            runtime.breakers[decision.model].failure()
            next(item for item in runtime.router.backends if item.model == decision.model).circuit_open = True
            REQUESTS.labels(decision.model, "error").inc()
    raise HTTPException(503, f"all backends failed: {last_error}")


@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest, x_tenant_id: str = Header("default")) -> Response:
    if not runtime.admission.acquire():
        raise HTTPException(429, "gateway queue full", headers={"Retry-After": "1"})
    new_trace_id()
    try:
        cache_payload = body.model_dump()
        cached = runtime.response_cache.get(cache_payload) if not body.stream else None
        if cached is not None:
            return Response(json.dumps(cached), media_type="application/json", headers={"x-routebench-cache": "hit"})
        wire, backend = _complete(body, x_tenant_id)
        if body.stream:
            def events() -> Any:
                chunk = {
                    "id": wire["id"], "object": "chat.completion.chunk", "created": wire["created"],
                    "model": wire["model"], "choices": [{"index": 0, "delta": wire["choices"][0]["message"], "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream", headers={"x-routebench-backend": backend})
        runtime.response_cache.set(cache_payload, wire)
        return Response(json.dumps(wire), media_type="application/json", headers={"x-routebench-backend": backend})
    finally:
        runtime.admission.release()


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": item.model, "object": "model", "owned_by": item.provider} for item in runtime.router.backends]}


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


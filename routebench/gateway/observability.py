"""Request-root tracing using the existing Console + JSONL OpenTelemetry setup."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from aisys.tracing import current_trace_id, init_tracing
from opentelemetry import trace
from prometheus_client import Counter

_SETUP_LOCK = threading.Lock()
HTTP_REQUESTS = Counter("routebench_gateway_requests_total", "Gateway chat HTTP requests", ["status"])


def ensure_tracing() -> None:
    with _SETUP_LOCK:
        if isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
            init_tracing()


class RequestTraceMiddleware:
    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        ensure_tracing()
        token = current_trace_id.set(uuid.uuid4().hex)
        started = time.perf_counter()
        tracer = trace.get_tracer("aisys")
        try:
            with tracer.start_as_current_span("gateway.request") as span:
                span.set_attribute("aisys.kind", "agent")
                span.set_attribute("aisys.trace_id", current_trace_id.get())
                span.set_attribute("http.method", scope["method"])
                span.set_attribute("http.route", scope["path"])

                async def traced_send(message: Any) -> None:
                    if message["type"] == "http.response.start":
                        span.set_attribute("http.status_code", message["status"])
                        if scope["path"] == "/v1/chat/completions":
                            HTTP_REQUESTS.labels(str(message["status"])).inc()
                    await send(message)

                try:
                    await self.app(scope, receive, traced_send)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_attribute("aisys.error", type(exc).__name__)
                    raise
                finally:
                    span.set_attribute("aisys.latency_ms", (time.perf_counter() - started) * 1000)
        finally:
            current_trace_id.reset(token)


def record_completion(wire: dict[str, Any], backend: str, cost: float, *, cache_hit: bool = False) -> None:
    span = trace.get_current_span()
    span.set_attribute("llm.model", wire["model"])
    span.set_attribute("routebench.backend", backend)
    span.set_attribute("routebench.cache_hit", cache_hit)
    span.set_attribute("routebench.usage_known", wire["usage"] is not None)
    if wire["usage"] is not None:
        span.set_attribute("llm.prompt_tokens", wire["usage"]["prompt_tokens"])
        span.set_attribute("llm.completion_tokens", wire["usage"]["completion_tokens"])
    span.set_attribute("llm.cost_usd", cost)

"""OpenTelemetry setup + @traced decorator. Spans carry model/tokens/cost/latency so trace mining
(ForgeCode) and online evals (RouteBench) can query them from Phoenix."""
from __future__ import annotations

import contextvars
import functools
import json
import time
import uuid
from typing import Any, Callable, Literal

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from .settings import settings

Kind = Literal["llm", "tool", "agent", "router", "eval", "approval"]
current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_TRUNC = 4000


def init_tracing(service_name: str | None = None, exporter: SpanExporter | None = None) -> None:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name or settings.service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(exporter or OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)


def new_trace_id() -> str:
    tid = uuid.uuid4().hex
    current_trace_id.set(tid)
    return tid


def _short(v: Any) -> str:
    try:
        s = json.dumps(v, default=str)
    except Exception:
        s = str(v)
    return s if len(s) <= _TRUNC else s[:_TRUNC] + f"...[{len(s) - _TRUNC} more]"


def traced(kind: Kind, name: str | None = None) -> Callable:
    """Wrap a function; record inputs/outputs (truncated), latency, and any usage/cost on the result."""
    def deco(fn: Callable) -> Callable:
        span_name = name or f"{kind}.{fn.__name__}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer("aisys")
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("aisys.kind", kind)
                span.set_attribute("aisys.trace_id", current_trace_id.get() or new_trace_id())
                span.set_attribute("aisys.input", _short({"args": args[1:] if kind == "tool" else args, "kwargs": kwargs}))
                t0 = time.perf_counter()
                try:
                    out = fn(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("aisys.error", type(e).__name__)
                    raise
                span.set_attribute("aisys.latency_ms", (time.perf_counter() - t0) * 1000)
                span.set_attribute("aisys.output", _short(getattr(out, "text", out)))
                usage = getattr(out, "usage", None)
                if usage is not None:
                    span.set_attribute("llm.model", getattr(out, "model", ""))
                    span.set_attribute("llm.prompt_tokens", usage.prompt_tokens)
                    span.set_attribute("llm.completion_tokens", usage.completion_tokens)
                    span.set_attribute("llm.cost_usd", getattr(out, "cost_usd", None) or 0.0)
                return out
        return wrapper
    return deco

"""OpenTelemetry setup + @traced decorator. Spans carry model/tokens/cost/latency so trace mining
(ForgeCode) and online evals (RouteBench) can query them from Phoenix."""
from __future__ import annotations

import contextvars
import functools
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from .settings import settings

Kind = Literal["llm", "tool", "agent", "router", "eval", "approval"]
current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_TRUNC = 4000
P = ParamSpec("P")
R = TypeVar("R")


class JsonlSpanExporter(SpanExporter):
    """Append completed spans as stable, queryable JSON lines."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Any) -> SpanExportResult:
        with self.path.open("a", encoding="utf-8") as stream:
            for span in spans:
                context = span.get_span_context()
                payload = {
                    "name": span.name,
                    "trace_id": format(context.trace_id, "032x"),
                    "span_id": format(context.span_id, "016x"),
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "attributes": dict(span.attributes or {}),
                    "status": span.status.status_code.name,
                }
                stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        return SpanExportResult.SUCCESS


def init_tracing(service_name: str | None = None, exporter: SpanExporter | None = None) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name or settings.service_name}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(settings.trace_jsonl_path)))
    trace.set_tracer_provider(provider)
    return provider


def new_trace_id() -> str:
    tid = uuid.uuid4().hex
    current_trace_id.set(tid)
    return tid


def _short(v: Any) -> str:
    try:
        s = json.dumps(v, default=str)
    except Exception:  # noqa: BLE001 - telemetry serialization must never break the application
        s = str(v)
    return s if len(s) <= _TRUNC else s[:_TRUNC] + f"...[{len(s) - _TRUNC} more]"


def traced(kind: Kind, name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a function; record inputs/outputs (truncated), latency, and any usage/cost on the result."""
    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        span_name = name or f"{kind}.{fn.__name__}"

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
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

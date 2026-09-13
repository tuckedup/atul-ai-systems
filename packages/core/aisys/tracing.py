"""OpenTelemetry setup + @traced decorator. Spans carry model/tokens/cost/latency so trace mining
(ForgeCode) and online evals (RouteBench) can query them from Phoenix.

When Phoenix / OTel collector is unavailable, spans are exported to:
  - the console (ConsoleSpanExporter), and
  - a JSONL file at .local/traces.jsonl

The @traced decorator and span attributes do NOT change. Only the exporter changes.
"""

from __future__ import annotations

import atexit
import contextvars
import functools
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from .settings import settings

Kind = Literal["llm", "tool", "agent", "router", "eval", "approval"]
current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_TRUNC = 4000

# ---------------------------------------------------------------------------
# JSONL file exporter — appends one JSON object per span, opened once.
# ---------------------------------------------------------------------------

_DEFAULT_JSONL = Path(".local") / "traces.jsonl"


class JsonlSpanExporter(SpanExporter):
    """Writes each span as one JSON line to a file. Meant for debugging/audit when no
    OTel backend is reachable — not for high-throughput production."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_JSONL
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self._path.open("a", encoding="utf-8")

    def export(self, spans: list[trace.Span]) -> bool:
        try:
            for s in spans:
                line = _span_to_json(s)
                if line:
                    self._f.write(line + "\n")
            self._f.flush()
            return True
        except Exception:
            return False

    def shutdown(self, timeout_millis: int = 5000) -> bool:
        try:
            self._f.close()
        except Exception:
            return False
        return True


def _span_to_json(s: trace.Span) -> str:
    try:
        attrs = {a.key: a.value for a in s.attributes.keys()}
    except AttributeError:
        attrs = dict(s.attributes)
    return json.dumps({
        "trace_id": s.get_span_context().trace_id,
        "span_id": s.get_span_context().span_id,
        "parent_span_id": s.parent.span_id if s.parent else None,
        "name": s.name,
        "kind": attrs.get("aisys.kind", ""),
        "path": s.resource.attributes.get("service.name", ""),
        "attributes": attrs,
        "status": s.status.status_code.name,
        "duration_ms": 0.0,
    }, default=str, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Public init
# ---------------------------------------------------------------------------

_DEFAULT_EXPORTER: SpanExporter | None = None


def init_tracing(service_name: str | None = None, exporter: SpanExporter | None = None) -> SpanExporter:
    """Set up the global tracer provider. If no exporter is given, use Console + JSONL.

    When you *do* have Phoenix / OTel collector, pass:
        exporter=OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True)
    and the provider will ship spans there instead.
    """
    global _DEFAULT_EXPORTER
    svc = service_name or settings.service_name
    providers: list[SpanExporter] = []
    jsonl_exporter: SpanExporter | None = None
    if exporter is not None:
        providers.append(exporter)
    else:
        providers.append(ConsoleSpanExporter())
        # JSONL is low-throughput debug/audit; use SimpleSpanProcessor so every
        # span is written immediately (no batching, no shutdown race in tests).
        jsonl_exporter = JsonlSpanExporter()
        _DEFAULT_EXPORTER = jsonl_exporter
    provider = TracerProvider(resource=Resource.create({"service.name": svc}))
    for p in providers:
        provider.add_span_processor(BatchSpanProcessor(p))
    # JSONL writer gets a separate SimpleSpanProcessor so spans land on disk
    # immediately — avoids the BatchSpanProcessor shutdown race in tests.
    if jsonl_exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(jsonl_exporter))


    trace.set_tracer_provider(provider)
    return providers[0]


def default_exporter() -> SpanExporter | None:
    """Return the JSONL exporter if it was created by init_tracing(); None otherwise."""
    return _DEFAULT_EXPORTER


def new_trace_id() -> str:
    tid = uuid.uuid4().hex
    current_trace_id.set(tid)
    return tid


def _short(v: Any) -> str:
    try:
        s = json.dumps(v, default=str)
    except Exception:
        s = str(v)
    if len(s) <= _TRUNC:
        return s
    suffix = f"...[{len(s) - _TRUNC} more]"
    return s[:_TRUNC - len(suffix)] + suffix


# ---------------------------------------------------------------------------
# Test hook: a debug span processor that records spans in-process so unit
# tests can assert on attributes without firing the real exporters.
# ---------------------------------------------------------------------------

DEBUG_SPANS: list[dict[str, Any]] = []


def _debug_span_processor(span: Any) -> None:
    try:
        attrs = {a.key: a.value for a in span.attributes.keys()}
    except AttributeError:
        attrs = dict(span.attributes)
    DEBUG_SPANS.append({
        "name": span.name,
        "kind": attrs.get("aisys.kind", ""),
        "aisys.trace_id": attrs.get("aisys.trace_id", ""),
        "aisys.input": attrs.get("aisys.input", ""),
        "aisys.output": attrs.get("aisys.output", ""),
        "aisys.latency_ms": attrs.get("aisys.latency_ms", 0.0),
        "aisys.error": attrs.get("aisys.error", ""),
        "llm.model": attrs.get("llm.model", ""),
        "llm.prompt_tokens": attrs.get("llm.prompt_tokens", 0),
        "llm.completion_tokens": attrs.get("llm.completion_tokens", 0),
        "llm.cost_usd": attrs.get("llm.cost_usd", None),
        "status": span.status.status_code.name,
    })


def init_tracing_for_test(service_name: str | None = None) -> None:
    """Like init_tracing() but adds a debug span processor so tests can inspect spans.
    Does NOT add the real exporters (Console/JSONL) — only the debug collector.

    Call this in test fixtures instead of init_tracing() when you want to assert
    on span attributes directly.
    """
    svc = service_name or settings.service_name
    provider = TracerProvider(resource=Resource.create({"service.name": svc}))
    provider.add_span_processor(_debug_span_processor)
    trace.set_tracer_provider(provider)


def shutdown_provider() -> None:
    """Shut down the global tracer provider so tests can flush + close exporters."""
    prov = trace.get_tracer_provider()
    prov.force_flush()
    prov.shutdown()
    _close_default_jsonl()
    DEBUG_SPANS.clear()
    # Reset the global tracer provider so a subsequent init_tracing() in another
    # test can create a fresh provider. TracerProvider.set_tracer_provider is
    # once-only guarded (a SpinLock), so we must clear the global by creating
    # a throwaway provider and swapping it in, then shut it down immediately.
    _reset_global_tracer_provider()


def _reset_global_tracer_provider() -> None:
    """Replace the global TracerProvider with a fresh no-op one so that the
    next call to init_tracing() (which calls trace.set_tracer_provider) is
    allowed to install its own provider. Without this, the SDK's SpinLock
    rejects the second set_tracer_provider call with a warning and the span
    from the next test is exported through the previous test's provider.
    """
    global _DEFAULT_EXPORTER
    try:
        # A brand-new provider with no processors; shutting it down is a no-op.
        dummy = TracerProvider(resource=Resource.create({}))
        # The SDK's set_tracer_provider is once-only guarded (SpinLock) and
        # will refuse to install 'dummy' if a provider was already set. We
        # bypass the guard by writing directly to the module-level global.
        import opentelemetry.trace as _ot
        _ot._TRACER_PROVIDER = dummy
        dummy.shutdown()
        _DEFAULT_EXPORTER = None
    except Exception:
        # If anything goes wrong, at least clear the module reference so the
        # next init_tracing creates a brand-new exporter + file handle.
        _DEFAULT_EXPORTER = None


def _close_default_jsonl() -> None:
    """Close the default JSONL exporter's file handle and clear the module reference."""
    global _DEFAULT_EXPORTER
    # Close the file handle attached to the active tracer provider's
    # SimpleSpanProcessor (if any) — this is the handle actually writing
    # spans for the current test. Also close _DEFAULT_EXPORTER's handle
    # for backwards compatibility with tests that set it directly.
    try:
        prov = trace.get_tracer_provider()
        if hasattr(prov, "_active_span_processor"):
            smp = prov._active_span_processor
            if hasattr(smp, "_span_processors"):
                for sp in smp._span_processors:
                    if isinstance(sp, SimpleSpanProcessor) and hasattr(sp, "span_exporter"):
                        fe = sp.span_exporter
                        if hasattr(fe, "_f") and fe._f is not None and not fe._f.closed:
                            try:
                                fe._f.close()
                            except Exception:
                                pass
    except Exception:
        pass
    if _DEFAULT_EXPORTER is not None and hasattr(_DEFAULT_EXPORTER, "_f"):
        f = _DEFAULT_EXPORTER._f
        if f is not None and not f.closed:
            try:
                f.close()
            except Exception:
                pass
    _DEFAULT_EXPORTER = None


def get_debug_spans_for_kind(kind: str) -> list[dict[str, Any]]:
    return [s for s in DEBUG_SPANS if s["kind"] == kind]


def debug_span_to_jsonl(span: Any) -> str:
    """Serialize one span as the same JSONL line the JsonlSpanExporter would write."""
    try:
        attrs = {a.key: a.value for a in span.attributes.keys()}
    except AttributeError:
        attrs = dict(span.attributes)
    return json.dumps({
        "trace_id": span.get_span_context().trace_id,
        "span_id": span.get_span_context().span_id,
        "parent_span_id": span.parent.span_id if span.parent else None,
        "name": span.name,
        "kind": attrs.get("aisys.kind", ""),
        "path": span.resource.attributes.get("service.name", ""),
        "attributes": attrs,
        "status": span.status.status_code.name,
        "duration_ms": 0.0,
    }, default=str, separators=(",", ":"))


def traced(kind: Kind, name: str | None = None) -> Callable:
    """Wrap a function; record inputs/outputs (truncated), latency, and any usage/cost on the result.

    Span attributes are identical regardless of exporter — only the destination changes.
    """

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

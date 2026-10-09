"""Existing Console/JSONL setup plus request-root parenting, without a collector."""
import json

from aisys.llm import ChatResult, Usage
from aisys.tracing import current_trace_id, traced
from fastapi.testclient import TestClient
from gateway import observability
from gateway.app import app, build_runtime, configure_runtime
from gateway.router import Backend
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class LocalDouble:
    name = "trace-test"

    @traced(kind="llm")
    def chat(self, payload):
        return ChatResult(text="hello", usage=Usage(prompt_tokens=2, completion_tokens=1),
                          model=payload["model"], provider=self.name, cost_usd=0, latency_ms=1)


def test_root_router_backend_and_cache_share_trace_and_reset_context(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "ensure_tracing", lambda: None)
    monkeypatch.setattr(observability.trace, "get_tracer", lambda *_a, **_k: provider.get_tracer("test"))
    backend = Backend("local", "trace-test", 0, 0, 1)
    configure_runtime(build_runtime([backend], {"local": LocalDouble()}, {"local": {"chat": 1}}))
    client = TestClient(app)
    caller_trace_id = current_trace_id.get()
    body = {"messages": [{"role": "user", "content": "hello"}]}
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert client.post("/v1/chat/completions", json=body).headers["x-routebench-cache"] == "hit"
    spans = exporter.get_finished_spans()
    roots = [s for s in spans if s.name == "gateway.request"]
    assert len(roots) == 2 and roots[0].parent is None and roots[1].parent is None
    children = [s for s in spans if s.name != "gateway.request"]
    assert children and all(s.context.trace_id == roots[0].context.trace_id for s in children)
    assert all(s.parent.span_id == roots[0].context.span_id for s in children)
    assert all(s.attributes["aisys.trace_id"] == roots[0].attributes["aisys.trace_id"] for s in children)
    assert roots[1].attributes["routebench.cache_hit"] is True
    assert roots[1].attributes["llm.cost_usd"] == 0
    assert current_trace_id.get() == caller_trace_id
    assert roots[0].attributes["aisys.trace_id"] != roots[1].attributes["aisys.trace_id"]
    # This is an in-memory exporter test, not a claim of Phoenix verification.
    json.dumps([dict(s.attributes) for s in spans])

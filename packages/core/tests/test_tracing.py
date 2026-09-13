"""Tests for aisys.tracing — verify spans are exported correctly."""
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from aisys.tracing import current_trace_id, init_tracing, new_trace_id, traced


def test_traced_decorator_records_span(_clear_exporter: InMemorySpanExporter) -> None:
    """Run a traced function and assert a span was exported with correct attributes."""

    @traced(kind="tool", name="test.add")
    def add(a: int, b: int) -> int:
        return a + b

    result = add(2, 3)
    assert result == 5

    spans = _clear_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "test.add"
    attrs = span.attributes
    assert attrs is not None
    assert attrs["aisys.kind"] == "tool"
    assert "aisys.trace_id" in attrs
    assert "aisys.input" in attrs
    assert "aisys.output" in attrs
    assert "aisys.latency_ms" in attrs
    latency = attrs["aisys.latency_ms"]
    assert isinstance(latency, (int, float))
    assert latency >= 0


def test_traced_records_error(_clear_exporter: InMemorySpanExporter) -> None:
    """Verify that exceptions are recorded on the span."""

    @traced(kind="agent", name="test.fail")
    def failing() -> None:
        raise ValueError("boom")

    try:
        failing()
    except ValueError:
        pass

    spans = _clear_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = span.attributes
    assert attrs is not None
    assert attrs.get("aisys.error") == "ValueError"
    assert len(span.events) >= 1
    assert any(e.name == "exception" for e in span.events)


def test_new_trace_id_sets_contextvar() -> None:
    """new_trace_id generates a 32-char hex and sets the contextvar."""
    tid = new_trace_id()
    assert len(tid) == 32
    assert current_trace_id.get() == tid


def test_init_tracing_with_in_memory_exporter(_clear_exporter: InMemorySpanExporter) -> None:
    """init_tracing installs a provider; traced functions produce spans."""
    init_tracing(service_name="test-service", exporter=_clear_exporter)

    @traced(kind="router", name="test.route")
    def route() -> str:
        return "ok"

    result = route()
    assert result == "ok"

    spans = _clear_exporter.get_finished_spans()
    assert any(s.name == "test.route" for s in spans)


def test_traced_llm_attributes(_clear_exporter: InMemorySpanExporter) -> None:
    """When output has .usage, llm.* attributes are set on the span."""

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5

    class FakeResult:
        text = "hi"
        usage = FakeUsage()
        model = "gpt-4o-mini"
        cost_usd = 0.001

    @traced(kind="llm", name="test.llm")
    def fake_llm() -> FakeResult:
        return FakeResult()

    fake_llm()

    spans = _clear_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs is not None
    assert attrs["llm.model"] == "gpt-4o-mini"
    assert attrs["llm.prompt_tokens"] == 10
    assert attrs["llm.completion_tokens"] == 5
    assert attrs["llm.cost_usd"] == 0.001

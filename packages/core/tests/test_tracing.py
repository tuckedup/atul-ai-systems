from aisys import tracing
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_traced_function_exports_attributes():
    exporter = InMemorySpanExporter()
    provider = tracing.init_tracing("test", exporter)

    @tracing.traced(kind="tool")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert spans[-1].attributes["aisys.kind"] == "tool"
    assert spans[-1].attributes["aisys.output"] == "5"

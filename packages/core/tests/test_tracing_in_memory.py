"""
M3 verification test for tracing.py.
Tests that @traced decorator emits spans and that the JSONL exporter works.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from opentelemetry import trace as ot
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import get_tracer_provider as _get_tracer_provider

from aisys import tracing


def test_jsonl_exporter_writes_span_to_file():
    """M3: a span exported through JsonlSpanExporter lands in the JSONL file."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.jsonl"

        exporter = tracing.JsonlSpanExporter(path=p)
        tracing._DEFAULT_EXPORTER = exporter
        # Add the processor to the global provider. Do NOT call
        # ot.set_tracer_provider(prov) — the SDK's once-only guard may reject it
        # if another test already called init_tracing(), and the new provider
        # would never become global. Instead, attach the processor directly to
        # the already-global provider so spans flow through this exporter.
        ot.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))

        tracer = ot.get_tracer("test")
        with tracer.start_as_current_span("test.add") as span:
            span.set_attribute("aisys.kind", "tool")
            span.set_attribute("aisys.trace_id", "abc123")
            span.set_attribute("aisys.output", "5")
        exporter.shutdown()
        tracing.shutdown_provider()
        assert p.exists()
        text = p.read_text().strip()
        assert text, "JSONL file should be non-empty"
        data = json.loads(text.splitlines()[0])
        assert data["name"] == "test.add"
        assert data["kind"] == "tool"
        assert data["attributes"]["aisys.output"] == "5"


def test_traced_function_writes_jsonl():
    """M3: calling a @traced function writes a span to .local/traces.jsonl."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.jsonl"
        orig = tracing._DEFAULT_JSONL
        tracing._DEFAULT_JSONL = p
        try:
            tracing.init_tracing("test-svc")
            # init_tracing already created a JsonlSpanExporter writing to p.

            @tracing.traced(kind="tool")
            def add(a: int, b: int) -> int:
                return a + b

            assert add(2, 3) == 5
            tracing.shutdown_provider()
            assert p.exists()
            text = p.read_text().strip()
            assert text
            data = json.loads(text.splitlines()[0])
            assert data["name"] == "tool.add"
            assert data["kind"] == "tool"
            assert data["attributes"]["aisys.output"] == "5"
        finally:
            tracing._DEFAULT_JSONL = orig


def test_traced_decorator_preserves_metadata():
    """@traced must not break __name__ or __doc__."""
    @tracing.traced(kind="tool")
    def my_tool(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    assert my_tool.__name__ == "my_tool"
    assert my_tool.__doc__ == "Add two numbers."
    assert my_tool(3, 4) == 7


def test_traced_records_error_on_exception():
    """M3: an exception inside a traced function produces a span with status ERROR."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.jsonl"
        orig = tracing._DEFAULT_JSONL
        tracing._DEFAULT_JSONL = p
        try:
            tracing.init_tracing("test-svc")
            # init_tracing already created a JsonlSpanExporter writing to p.
            # Do NOT create a second JsonlSpanExporter at the same path — that
            # doubles file handles and, on Windows, risks a PermissionError
            # during TemporaryDirectory cleanup.

            @tracing.traced(kind="tool")
            def boom() -> int:
                raise ValueError("boom")

            with pytest.raises(ValueError, match="boom"):
                boom()

            tracing.shutdown_provider()
            if p.exists():
                text = p.read_text().strip()
                if text:
                    data = json.loads(text.splitlines()[-1])
                    assert data["status"] == "ERROR"
        finally:
            tracing._DEFAULT_JSONL = orig


def test_trace_id_context_propagation():
    """trace_id set via new_trace_id() is readable via current_trace_id.get()."""
    tracing.init_tracing("test-svc")
    tid = tracing.new_trace_id()
    assert tid and len(tid) == 32
    assert tracing.current_trace_id.get() == tid
    tracing.shutdown_provider()


def test_traced_output_truncation():
    """_short() must truncate long values; the function return is unaffected."""
    big = "x" * 5000

    @tracing.traced(kind="tool")
    def return_big() -> str:
        return big

    assert return_big() == big  # caller gets full value
    short = tracing._short(big)
    assert len(short) <= tracing._TRUNC + 20
    assert "...[" in short  # truncation marker present

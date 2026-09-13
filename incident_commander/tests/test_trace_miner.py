"""Tests for Incident Commander M6 — trace miner and report generator."""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tracing_miner import TraceMiner, Span, Trace
from reports.generator import generate_report


_TEST_DIR = os.path.join(os.path.dirname(__file__), "..", ".local", "test_traces")


def _cleanup():
    if os.path.exists(_TEST_DIR):
        shutil.rmtree(_TEST_DIR, ignore_errors=True)


def _make_span(
    name: str = "test.span",
    trace_id: str = "abc123",
    span_id: str = "s001",
    parent_span_id: str | None = None,
    kind: str = "agent",
    start_time: float = 1_000_000_000,
    end_time: float = 2_000_000_000,
    model: str = "",
    tokens: int = 0,
    cost: float = 0.0,
    error: str = "",
    incident_id: str = "",
) -> dict:
    """Build a JSONL record dict."""
    attrs: dict = {"aisys.kind": kind, "aisys.input": "test", "aisys.output": "result"}
    if model:
        attrs["llm.model"] = model
        attrs["llm.prompt_tokens"] = tokens // 2
        attrs["llm.completion_tokens"] = tokens // 2
        attrs["llm.cost_usd"] = cost
    if error:
        attrs["aisys.error"] = error
    if incident_id:
        attrs["aisys.trace_id"] = incident_id
    return {
        "name": name,
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "start_time": start_time,
        "end_time": end_time,
        "attributes": attrs,
        "status": "ERROR" if error else "OK",
        "events": [],
    }


def _write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---- TraceMiner tests ----

def test_miner_loads_empty_file():
    _cleanup()
    try:
        path = os.path.join(_TEST_DIR, "empty.jsonl")
        _write_jsonl(path, [])
        miner = TraceMiner(path)
        miner.load()
        assert miner.all_trace_ids() == []
    finally:
        _cleanup()


def test_miner_loads_single_trace():
    _cleanup()
    try:
        path = os.path.join(_TEST_DIR, "single.jsonl")
        records = [
            _make_span(trace_id="t1", span_id="s1", kind="agent", name="coordinator"),
            _make_span(trace_id="t1", span_id="s2", parent_span_id="s1", kind="llm", name="llm.call", model="gpt-4o-mini", tokens=100, cost=0.001),
            _make_span(trace_id="t1", span_id="s3", parent_span_id="s1", kind="tool", name="prometheus.query"),
        ]
        _write_jsonl(path, records)
        miner = TraceMiner(path)
        miner.load()
        assert len(miner.all_trace_ids()) == 1
        trace = miner.get_trace("t1")
        assert trace is not None
        assert len(trace.spans) == 3
        assert len(trace.root_spans) == 1
        assert trace.root_spans[0].span_id == "s1"
    finally:
        _cleanup()


def test_miner_multiple_traces():
    _cleanup()
    try:
        path = os.path.join(_TEST_DIR, "multi.jsonl")
        records = [
            _make_span(trace_id="t1", span_id="s1", kind="agent"),
            _make_span(trace_id="t2", span_id="s2", kind="agent"),
            _make_span(trace_id="t3", span_id="s3", kind="llm"),
        ]
        _write_jsonl(path, records)
        miner = TraceMiner(path)
        miner.load()
        assert len(miner.all_trace_ids()) == 3
    finally:
        _cleanup()


def test_miner_incident_indexing():
    _cleanup()
    try:
        path = os.path.join(_TEST_DIR, "incident.jsonl")
        records = [
            _make_span(trace_id="t1", span_id="s1", incident_id="inc-001"),
            _make_span(trace_id="t2", span_id="s2", incident_id="inc-001"),
            _make_span(trace_id="t3", span_id="s3", incident_id="inc-002"),
        ]
        _write_jsonl(path, records)
        miner = TraceMiner(path)
        miner.load()
        assert len(miner.traces_for_incident("inc-001")) == 2
        assert len(miner.traces_for_incident("inc-002")) == 1
        assert len(miner.traces_for_incident("inc-999")) == 0
        assert set(miner.all_incident_ids()) == {"inc-001", "inc-002"}
    finally:
        _cleanup()


def test_miner_summary_for_incident():
    _cleanup()
    try:
        path = os.path.join(_TEST_DIR, "summary.jsonl")
        records = [
            _make_span(trace_id="t1", span_id="s1", incident_id="inc-001", kind="agent", name="coordinator"),
            _make_span(trace_id="t1", span_id="s2", parent_span_id="s1", incident_id="inc-001", kind="llm", name="llm.call", model="gpt-4o-mini", tokens=200, cost=0.002),
            _make_span(trace_id="t1", span_id="s3", parent_span_id="s1", incident_id="inc-001", kind="tool", name="prometheus.query"),
        ]
        _write_jsonl(path, records)
        miner = TraceMiner(path)
        summary = miner.summary_for_incident("inc-001")
        assert summary["incident_id"] == "inc-001"
        assert summary["spans"] == 3
        assert summary["total_tokens"] == 200
        assert summary["total_cost_usd"] > 0
        assert "coordinator" in summary["agents_called"]
        assert "gpt-4o-mini" in summary["llm_models_used"]
        assert len(summary["spans_timeline"]) == 3
    finally:
        _cleanup()


def test_miner_errors_detected():
    _cleanup()
    try:
        path = os.path.join(_TEST_DIR, "errors.jsonl")
        records = [
            _make_span(trace_id="t1", span_id="s1", kind="llm", name="llm.call", error="ProviderError"),
            _make_span(trace_id="t1", span_id="s2", kind="agent", name="coordinator"),
        ]
        _write_jsonl(path, records)
        miner = TraceMiner(path)
        summary = miner.summary_for_incident("")
        # No incident_id means no indexed spans, but we can test via get_trace
        trace = miner.get_trace("t1")
        assert trace is not None
        assert len(trace.errors) == 1
        assert trace.errors[0].is_error
    finally:
        _cleanup()


def test_span_properties():
    span = Span(
        name="test", trace_id="t1", span_id="s1", parent_span_id=None,
        start_time=1_000_000_000, end_time=2_000_000_000,
        attributes={"aisys.kind": "llm", "llm.prompt_tokens": 50, "llm.completion_tokens": 50, "llm.cost_usd": 0.005},
    )
    assert span.kind == "llm"
    assert span.token_count == 100
    assert span.cost_usd == 0.005
    assert span.duration_ms > 0


def test_trace_timeline_sorted():
    trace = Trace(trace_id="t1", spans=[
        Span(name="b", trace_id="t1", span_id="s2", parent_span_id=None, start_time=3, end_time=4),
        Span(name="a", trace_id="t1", span_id="s1", parent_span_id=None, start_time=1, end_time=2),
    ])
    timeline = trace.timeline
    assert timeline[0].name == "a"
    assert timeline[1].name == "b"


def test_miner_write_incident_traces():
    _cleanup()
    try:
        path = os.path.join(_TEST_DIR, "write.jsonl")
        records = [
            _make_span(trace_id="t1", span_id="s1", incident_id="inc-001"),
            _make_span(trace_id="t2", span_id="s2", incident_id="inc-001"),
            _make_span(trace_id="t3", span_id="s3", incident_id="inc-002"),
        ]
        _write_jsonl(path, records)
        miner = TraceMiner(path)
        out_path = os.path.join(_TEST_DIR, "extracted", "inc-001.jsonl")
        result = miner.write_incident_traces("inc-001", out_path)
        assert result.exists()
        lines = result.read_text().strip().split("\n")
        assert len(lines) == 2
    finally:
        _cleanup()


# ---- Report generator tests ----

def test_report_generation():
    _cleanup()
    try:
        path = os.path.join(_TEST_DIR, "report_traces.jsonl")
        records = [
            _make_span(trace_id="t1", span_id="s1", incident_id="inc-rpt", kind="agent", name="coordinator"),
            _make_span(trace_id="t1", span_id="s2", parent_span_id="s1", incident_id="inc-rpt", kind="llm", name="llm.call", model="gpt-4o-mini", tokens=150, cost=0.0015),
            _make_span(trace_id="t1", span_id="s3", parent_span_id="s1", incident_id="inc-rpt", kind="tool", name="prometheus.query"),
            _make_span(trace_id="t1", span_id="s4", incident_id="inc-rpt", kind="agent", name="root_cause_agent"),
            _make_span(trace_id="t1", span_id="s5", incident_id="inc-rpt", kind="llm", name="llm.call", model="gpt-4o-mini", tokens=200, cost=0.002, error="TimeoutError"),
        ]
        _write_jsonl(path, records)
        miner = TraceMiner(path)

        alert = {
            "alertname": "HighErrorRate",
            "severity": "critical",
            "service": "demo-service",
            "summary": "Error rate exceeded 10%",
            "startsAt": "2026-09-12T04:00:00Z",
        }
        report_dir = os.path.join(_TEST_DIR, "reports")
        report_path = generate_report("inc-rpt", miner, alert=alert, output_dir=report_dir)
        assert report_path.exists()
        content = report_path.read_text()
        assert "inc-rpt" in content
        assert "HighErrorRate" in content
        assert "coordinator" in content
        assert "TimeoutError" in content
    finally:
        _cleanup()


def test_report_no_alert():
    _cleanup()
    try:
        path = os.path.join(_TEST_DIR, "report_no_alert.jsonl")
        records = [
            _make_span(trace_id="t1", span_id="s1", incident_id="inc-na", kind="agent", name="coordinator"),
        ]
        _write_jsonl(path, records)
        miner = TraceMiner(path)
        report_dir = os.path.join(_TEST_DIR, "reports_na")
        report_path = generate_report("inc-na", miner, output_dir=report_dir)
        assert report_path.exists()
        content = report_path.read_text()
        assert "inc-na" in content
        assert "Triggering Alert" not in content
    finally:
        _cleanup()


def test_report_empty_incident():
    _cleanup()
    try:
        path = os.path.join(_TEST_DIR, "report_empty.jsonl")
        _write_jsonl(path, [])
        miner = TraceMiner(path)
        report_dir = os.path.join(_TEST_DIR, "reports_empty")
        report_path = generate_report("nonexistent", miner, output_dir=report_dir)
        assert report_path.exists()
        content = report_path.read_text()
        assert "nonexistent" in content
    finally:
        _cleanup()

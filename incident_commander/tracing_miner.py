"""JSONL trace miner — reads local trace files, indexes spans by trace_id, reconstructs incident trajectories."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_TRACE_PATH = os.path.join(os.path.dirname(__file__), "..", "packages", "core", ".local", "traces.jsonl")


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_time: float
    end_time: float
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "UNSET"
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Span:
        return cls(
            name=record["name"],
            trace_id=record["trace_id"],
            span_id=record["span_id"],
            parent_span_id=record.get("parent_span_id"),
            start_time=record.get("start_time", 0),
            end_time=record.get("end_time", 0),
            attributes=record.get("attributes", {}),
            status=record.get("status", "UNSET"),
            events=record.get("events", []),
        )

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1e-9 if self.start_time and self.end_time else 0.0

    @property
    def kind(self) -> str:
        return self.attributes.get("aisys.kind", "unknown")

    @property
    def is_error(self) -> bool:
        return self.status == "ERROR" or bool(self.attributes.get("aisys.error"))

    @property
    def token_count(self) -> int:
        return int(self.attributes.get("llm.prompt_tokens", 0)) + int(self.attributes.get("llm.completion_tokens", 0))

    @property
    def cost_usd(self) -> float:
        return float(self.attributes.get("llm.cost_usd", 0.0))


@dataclass
class Trace:
    trace_id: str
    spans: list[Span] = field(default_factory=list)

    @property
    def root_spans(self) -> list[Span]:
        return [s for s in self.spans if s.parent_span_id is None]

    @property
    def timeline(self) -> list[Span]:
        return sorted(self.spans, key=lambda s: s.start_time)

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def total_tokens(self) -> int:
        return sum(s.token_count for s in self.spans)

    @property
    def total_duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        starts = [s.start_time for s in self.spans if s.start_time]
        ends = [s.end_time for s in self.spans if s.end_time]
        if not starts or not ends:
            return 0.0
        return (max(ends) - min(starts)) * 1e-9

    @property
    def errors(self) -> list[Span]:
        return [s for s in self.spans if s.is_error]

    @property
    def kind_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for s in self.spans:
            counts[s.kind] += 1
        return dict(counts)


class TraceMiner:
    """Reads JSONL trace files, indexes spans by trace_id and incident_id."""

    def __init__(self, trace_path: str | Path | None = None):
        self.trace_path = Path(trace_path or _DEFAULT_TRACE_PATH)
        self._traces: dict[str, Trace] = defaultdict(lambda: Trace(trace_id=""))
        self._incident_index: dict[str, set[str]] = defaultdict(set)  # incident_id -> {trace_ids}
        self._loaded = False

    def load(self) -> None:
        """Parse the JSONL file and build indices."""
        if not self.trace_path.exists():
            return
        with open(self.trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                span = Span.from_record(record)
                tid = span.trace_id
                trace = self._traces[tid]
                trace.trace_id = tid
                trace.spans.append(span)
                # Index by incident_id if present
                incident_id = span.attributes.get("aisys.trace_id", "")
                if incident_id:
                    self._incident_index[incident_id].add(tid)
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get_trace(self, trace_id: str) -> Trace | None:
        self._ensure_loaded()
        return self._traces.get(trace_id)

    def traces_for_incident(self, incident_id: str) -> list[Trace]:
        self._ensure_loaded()
        tids = self._incident_index.get(incident_id, set())
        return [self._traces[tid] for tid in tids if tid in self._traces]

    def all_trace_ids(self) -> list[str]:
        self._ensure_loaded()
        return list(self._traces.keys())

    def all_incident_ids(self) -> list[str]:
        self._ensure_loaded()
        return list(self._incident_index.keys())

    def spans_for_incident(self, incident_id: str) -> list[Span]:
        self._ensure_loaded()
        tids = self._incident_index.get(incident_id, set())
        spans = []
        for tid in tids:
            spans.extend(self._traces.get(tid, Trace(trace_id="")).spans)
        return sorted(spans, key=lambda s: s.start_time)

    def summary_for_incident(self, incident_id: str) -> dict[str, Any]:
        self._ensure_loaded()
        traces = self.traces_for_incident(incident_id)
        all_spans = [s for t in traces for s in t.spans]
        if not all_spans:
            return {"incident_id": incident_id, "traces": 0, "spans": 0}

        by_kind: dict[str, list[Span]] = defaultdict(list)
        for s in all_spans:
            by_kind[s.kind].append(s)

        agent_spans = by_kind.get("agent", [])
        llm_spans = by_kind.get("llm", [])
        tool_spans = by_kind.get("tool", [])

        return {
            "incident_id": incident_id,
            "traces": len(traces),
            "spans": len(all_spans),
            "total_cost_usd": sum(s.cost_usd for s in all_spans),
            "total_tokens": sum(s.token_count for s in all_spans),
            "total_duration_ms": sum(t.total_duration_ms for t in traces),
            "error_count": sum(1 for s in all_spans if s.is_error),
            "kind_counts": {k: len(v) for k, v in by_kind.items()},
            "agents_called": [s.name for s in agent_spans],
            "llm_models_used": list({s.attributes.get("llm.model", "") for s in llm_spans if s.attributes.get("llm.model")}),
            "spans_timeline": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "start_time": s.start_time,
                    "duration_ms": s.duration_ms,
                    "is_error": s.is_error,
                    "tokens": s.token_count,
                    "cost_usd": s.cost_usd,
                }
                for s in sorted(all_spans, key=lambda x: x.start_time)
            ],
        }

    def write_incident_traces(self, incident_id: str, output_path: str | Path) -> Path:
        """Write all spans for an incident to a new JSONL file for offline analysis."""
        self._ensure_loaded()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        spans = self.spans_for_incident(incident_id)
        with open(output, "w", encoding="utf-8") as f:
            for s in spans:
                record = {
                    "name": s.name,
                    "trace_id": s.trace_id,
                    "span_id": s.span_id,
                    "parent_span_id": s.parent_span_id,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "attributes": s.attributes,
                    "status": s.status,
                    "events": s.events,
                }
                f.write(json.dumps(record, default=str) + "\n")
        return output

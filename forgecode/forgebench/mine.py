"""Mine failed, correlated trajectories from the local OpenTelemetry JSONL export.

The exporter writes one span per line. A single agent run can contain many spans
whose OpenTelemetry trace IDs differ, so ``aisys.trace_id`` is the durable
application-level correlation key. The miner emits one regression seed per
failed correlated trajectory and intentionally omits raw span inputs/outputs;
those fields can contain source code, prompts, paths, or credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from aisys.settings import settings

HERE = Path(__file__).parent
EXPECTED_CONTROL_FLOW_ERRORS = frozenset({"GraphInterrupt"})


@dataclass(frozen=True)
class MiningSummary:
    spans_read: int
    malformed_lines: int
    ignored_control_flow_spans: int
    failed_trajectories: int
    cases_written: int


def _status(span: dict[str, Any]) -> str:
    """Accept both the compact JSONL and OpenTelemetry console status shapes."""
    status = span.get("status", "")
    if isinstance(status, dict):
        status = status.get("status_code", "")
    return str(status).upper()


def _attributes(span: dict[str, Any]) -> dict[str, Any]:
    attrs = span.get("attributes", {})
    return attrs if isinstance(attrs, dict) else {}


def _correlation_id(span: dict[str, Any]) -> str:
    attrs = _attributes(span)
    value = attrs.get("aisys.trace_id") or span.get("trace_id")
    return str(value or "uncorrelated")


def _error_reason(span: dict[str, Any]) -> str | None:
    attrs = _attributes(span)
    error = attrs.get("aisys.error")
    if error:
        # Keep the useful exception summary bounded and single-line. Raw model
        # inputs and outputs never enter the mined case.
        return str(error).replace("\r", " ").replace("\n", " ")[:240]
    if _status(span) == "ERROR":
        return "status=ERROR"
    return None


def _timestamp(span: dict[str, Any], key: str) -> int | float | str | None:
    value = span.get(key)
    return value if isinstance(value, (int, float, str)) else None


def _span_summary(span: dict[str, Any]) -> dict[str, object]:
    attrs = _attributes(span)
    summary: dict[str, object] = {
        "name": str(span.get("name", "unknown")),
        "status": _status(span) or "UNSET",
        "kind": str(attrs.get("aisys.kind", "unknown")),
    }
    reason = _error_reason(span)
    if reason:
        summary["error"] = reason
    latency = attrs.get("aisys.latency_ms")
    if isinstance(latency, (int, float)):
        summary["latency_ms"] = round(float(latency), 3)
    return summary


def _input_fingerprint(spans: list[dict[str, Any]]) -> str | None:
    """Retain replay identity without copying potentially sensitive inputs."""
    for span in spans:
        value = _attributes(span).get("aisys.input")
        if isinstance(value, str) and value:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return None


def _read_spans(source: Path) -> tuple[list[dict[str, Any]], int]:
    spans: list[dict[str, Any]] = []
    malformed = 0
    if not source.exists():
        return spans, malformed
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict):
                spans.append(value)
            else:
                malformed += 1
    return spans, malformed


def _failed_trajectories(
    spans: list[dict[str, Any]],
) -> tuple[list[tuple[str, list[dict[str, Any]]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ignored = 0
    for span in spans:
        grouped[_correlation_id(span)].append(span)

    failed: list[tuple[str, list[dict[str, Any]]]] = []
    for trace_id, trajectory in grouped.items():
        reasons = [_error_reason(span) for span in trajectory]
        real_errors = [
            reason
            for reason in reasons
            if reason and reason not in EXPECTED_CONTROL_FLOW_ERRORS
        ]
        ignored += sum(reason in EXPECTED_CONTROL_FLOW_ERRORS for reason in reasons)
        if real_errors:
            def span_order(span: dict[str, Any]) -> tuple[int, float, str]:
                value = _timestamp(span, "start_time")
                if isinstance(value, (int, float)):
                    return (0, float(value), "")
                return (1, 0.0, str(value or ""))

            ordered = sorted(
                trajectory,
                key=span_order,
            )
            failed.append((trace_id, ordered))

    # Mine the newest failures first. String ordering works for ISO timestamps,
    # while numeric exporters retain natural numeric ordering via zero padding.
    def newest(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, str]:
        timestamps = [_timestamp(span, "end_time") for span in item[1]]
        numeric = [float(value) for value in timestamps if isinstance(value, (int, float))]
        textual = [str(value) for value in timestamps if isinstance(value, str)]
        if numeric:
            return (1, f"{max(numeric):030.3f}")
        return (0, max(textual, default=""))

    failed.sort(key=newest, reverse=True)
    return failed, ignored


def _case(trace_id: str, spans: list[dict[str, Any]]) -> dict[str, object]:
    errors = [
        reason
        for span in spans
        if (reason := _error_reason(span))
        and reason not in EXPECTED_CONTROL_FLOW_ERRORS
    ]
    failed_spans = [
        str(span.get("name", "unknown"))
        for span in spans
        if (_error_reason(span) or "") not in EXPECTED_CONTROL_FLOW_ERRORS
        and _error_reason(span)
    ]
    identity = json.dumps(
        {"trace_id": trace_id, "errors": errors, "failed_spans": failed_spans},
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    payload: dict[str, object] = {
        "trace_id": trace_id,
        "failure_reasons": list(dict.fromkeys(errors)),
        "failed_spans": list(dict.fromkeys(failed_spans)),
        "trajectory": [_span_summary(span) for span in spans],
    }
    fingerprint = _input_fingerprint(spans)
    if fingerprint:
        payload["input_fingerprint"] = fingerprint
    return {
        "id": f"mined-{digest}",
        "input": payload,
        "expected": {"status": "no_error"},
        "tags": ["mined", "failed-trajectory"],
        "grader": "exact",
    }


def mine_traces(source: Path, output: Path, report: Path, *, limit: int = 10) -> MiningSummary:
    """Write deterministic, privacy-bounded cases for up to ``limit`` failures."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    spans, malformed = _read_spans(source)
    failures, ignored = _failed_trajectories(spans)
    selected = failures[:limit]
    cases = [_case(trace_id, trajectory) for trace_id, trajectory in selected]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cases, sort_keys=False), encoding="utf-8")

    reasons = Counter(
        reason
        for _trace_id, trajectory in failures
        for span in trajectory
        if (reason := _error_reason(span))
        and reason not in EXPECTED_CONTROL_FLOW_ERRORS
    )
    lines = [
        "# Mined failed trajectories",
        "",
        f"Source: `{source.as_posix()}`",
        f"Spans read: {len(spans)}",
        f"Malformed lines skipped: {malformed}",
        f"Expected control-flow spans ignored: {ignored}",
        f"Failed trajectories found: {len(failures)}",
        f"Cases written: {len(cases)}",
        "",
        "## Failure reasons",
        "",
    ]
    lines.extend(f"- {reason}: {count}" for reason, count in reasons.most_common())
    if not reasons:
        lines.append("- none")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MiningSummary(
        spans_read=len(spans),
        malformed_lines=malformed,
        ignored_control_flow_spans=ignored,
        failed_trajectories=len(failures),
        cases_written=len(cases),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(settings.trace_jsonl_path))
    parser.add_argument("--output", type=Path, default=HERE / "mined_tasks" / "mined.yaml")
    parser.add_argument("--report", type=Path, default=HERE / "MINED.md")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    summary = mine_traces(args.source, args.output, args.report, limit=args.limit)
    print(
        f"mined {summary.cases_written}/{summary.failed_trajectories} failed trajectories "
        f"from {summary.spans_read} spans; malformed={summary.malformed_lines}; "
        f"ignored_control_flow={summary.ignored_control_flow_spans}"
    )


if __name__ == "__main__":
    main()

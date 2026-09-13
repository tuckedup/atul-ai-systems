"""Report generator — produces reports/<incident_id>.md from JSONL trace data."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracing_miner import TraceMiner, Trace, Span


_DEFAULT_REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def _ts_str(epoch_ns: float) -> str:
    """Convert nanosecond epoch to ISO string."""
    if not epoch_ns:
        return "N/A"
    return datetime.fromtimestamp(epoch_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_cost(cost: float) -> str:
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _format_duration(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def generate_report(
    incident_id: str,
    miner: TraceMiner,
    alert: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate a Markdown report for an incident from trace data.

    Returns the path to the written report.
    """
    output_dir = Path(output_dir or _DEFAULT_REPORTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = miner.summary_for_incident(incident_id)
    traces = miner.traces_for_incident(incident_id)
    all_spans = miner.spans_for_incident(incident_id)

    # Build markdown
    lines: list[str] = []
    w = lines.append

    w(f"# Incident Report: {incident_id}")
    w("")
    w(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    w("")

    # Alert section
    if alert:
        w("## Triggering Alert")
        w("")
        w(f"- **Alertname:** {alert.get('alertname', 'N/A')}")
        w(f"- **Severity:** {alert.get('severity', 'N/A')}")
        w(f"- **Service:** {alert.get('service', 'N/A')}")
        w(f"- **Summary:** {alert.get('summary', 'N/A')}")
        w(f"- **Started:** {alert.get('startsAt', 'N/A')}")
        w("")

    # Summary metrics
    w("## Summary Metrics")
    w("")
    w(f"| Metric | Value |")
    w(f"|--------|-------|")
    w(f"| Traces | {summary.get('traces', 0)} |")
    w(f"| Total spans | {summary.get('spans', 0)} |")
    w(f"| Total cost | {_format_cost(summary.get('total_cost_usd', 0.0))} |")
    w(f"| Total tokens | {summary.get('total_tokens', 0)} |")
    w(f"| Duration | {_format_duration(summary.get('total_duration_ms', 0.0))} |")
    w(f"| Errors | {summary.get('error_count', 0)} |")
    w("")

    # Agents called
    agents = summary.get("agents_called", [])
    if agents:
        w("## Agent Pipeline")
        w("")
        for i, name in enumerate(agents, 1):
            w(f"{i}. `{name}`")
        w("")

    # LLM models used
    models = summary.get("llm_models_used", [])
    if models:
        w("## LLM Models")
        w("")
        for m in models:
            w(f"- `{m}`")
        w("")

    # Span timeline
    timeline = summary.get("spans_timeline", [])
    if timeline:
        w("## Span Timeline")
        w("")
        w("| # | Name | Kind | Duration | Tokens | Cost | Error |")
        w("|---|------|------|----------|--------|------|-------|")
        for i, s in enumerate(timeline, 1):
            err = "ERR" if s.get("is_error") else ""
            w(f"| {i} | `{s['name']}` | {s['kind']} | {_format_duration(s.get('duration_ms', 0))} | {s.get('tokens', 0)} | {_format_cost(s.get('cost_usd', 0))} | {err} |")
        w("")

    # Errors detail
    errors = [s for s in all_spans if s.is_error]
    if errors:
        w("## Errors")
        w("")
        for s in errors:
            w(f"### `{s.name}` ({s.kind})")
            w("")
            error_msg = s.attributes.get("aisys.error", "unknown")
            w(f"- **Error type:** `{error_msg}`")
            w(f"- **Trace ID:** `{s.trace_id}`")
            w(f"- **Span ID:** `{s.span_id}`")
            w(f"- **Time:** {_ts_str(s.start_time)}")
            if s.attributes.get("aisys.input"):
                w(f"- **Input:** `{s.attributes['aisys.input'][:200]}`")
            w("")

    # Agent outputs
    agent_spans = [s for s in all_spans if s.kind == "agent"]
    if agent_spans:
        w("## Agent Outputs")
        w("")
        for s in agent_spans:
            w(f"### `{s.name}`")
            w("")
            output = s.attributes.get("aisys.output", "")
            if output:
                # Truncate long outputs
                if len(output) > 500:
                    output = output[:500] + f"... [{len(output) - 500} more chars]"
                w("```")
                w(output)
                w("```")
            else:
                w("*No output captured*")
            w("")

    # Cost breakdown by kind
    kind_costs: dict[str, float] = {}
    kind_tokens: dict[str, int] = {}
    for s in all_spans:
        k = s.kind
        kind_costs[k] = kind_costs.get(k, 0) + s.cost_usd
        kind_tokens[k] = kind_tokens.get(k, 0) + s.token_count
    if kind_costs:
        w("## Cost Breakdown")
        w("")
        w("| Kind | Cost | Tokens |")
        w("|------|------|--------|")
        for k in sorted(kind_costs, key=kind_costs.get, reverse=True):  # type: ignore
            w(f"| {k} | {_format_cost(kind_costs[k])} | {kind_tokens.get(k, 0)} |")
        w("")

    w("---")
    w("*Report generated from JSONL trace data by Incident Commander trace miner.*")
    w("")

    report_path = output_dir / f"{incident_id}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

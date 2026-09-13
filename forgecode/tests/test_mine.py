import json
from pathlib import Path

import yaml
from forgebench.mine import mine_traces


def _span(
    trace_id: str,
    name: str,
    status: str,
    timestamp: int,
    *,
    error: str | None = None,
    raw_input: str = "",
) -> dict[str, object]:
    attributes: dict[str, object] = {
        "aisys.trace_id": trace_id,
        "aisys.kind": "agent",
        "aisys.input": raw_input,
        "aisys.latency_ms": 1.23456,
    }
    if error:
        attributes["aisys.error"] = error
    return {
        "trace_id": f"otel-{name}",
        "name": name,
        "status": status,
        "start_time": timestamp,
        "end_time": timestamp + 1,
        "attributes": attributes,
    }


def _write_jsonl(path: Path, spans: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(span) for span in spans) + "\n", encoding="utf-8")


def test_miner_groups_failed_spans_and_ignores_approval_interrupts(tmp_path: Path):
    source = tmp_path / "traces.jsonl"
    output = tmp_path / "mined.yaml"
    report = tmp_path / "MINED.md"
    secret = "do-not-copy-this-prompt"
    spans = [
        _span("failed-run", "planner", "UNSET", 1, raw_input=secret),
        _span("failed-run", "tool.call", "ERROR", 2, error="ValueError"),
        _span("failed-run", "implementer", "ERROR", 3, error="ValueError"),
        _span("paused-run", "approval_gate", "ERROR", 4, error="GraphInterrupt"),
        _span("successful-run", "patch", "UNSET", 5),
    ]
    _write_jsonl(source, spans)
    source.write_text(source.read_text(encoding="utf-8") + "not-json\n", encoding="utf-8")

    summary = mine_traces(source, output, report)

    assert summary.spans_read == 5
    assert summary.malformed_lines == 1
    assert summary.ignored_control_flow_spans == 1
    assert summary.failed_trajectories == 1
    assert summary.cases_written == 1
    cases = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert len(cases) == 1
    assert cases[0]["input"]["trace_id"] == "failed-run"
    assert cases[0]["input"]["failed_spans"] == ["tool.call", "implementer"]
    assert [step["name"] for step in cases[0]["input"]["trajectory"]] == [
        "planner",
        "tool.call",
        "implementer",
    ]
    assert secret not in output.read_text(encoding="utf-8")
    assert "Expected control-flow spans ignored: 1" in report.read_text(encoding="utf-8")


def test_miner_is_stable_and_limits_newest_failures(tmp_path: Path):
    source = tmp_path / "traces.jsonl"
    output = tmp_path / "mined.yaml"
    report = tmp_path / "MINED.md"
    _write_jsonl(
        source,
        [
            _span("older", "reviewer", "ERROR", 10, error="RuntimeError"),
            _span("newer", "tester", "ERROR", 20),
        ],
    )

    first = mine_traces(source, output, report, limit=1)
    first_text = output.read_text(encoding="utf-8")
    second = mine_traces(source, output, report, limit=1)

    assert first == second
    assert output.read_text(encoding="utf-8") == first_text
    assert yaml.safe_load(first_text)[0]["input"]["trace_id"] == "newer"
    assert first.failed_trajectories == 2
    assert first.cases_written == 1


def test_miner_handles_missing_source_and_rejects_negative_limit(tmp_path: Path):
    output = tmp_path / "mined.yaml"
    report = tmp_path / "MINED.md"
    summary = mine_traces(tmp_path / "missing.jsonl", output, report)
    assert summary.spans_read == 0
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == []

    try:
        mine_traces(tmp_path / "missing.jsonl", output, report, limit=-1)
    except ValueError as error:
        assert str(error) == "limit must be non-negative"
    else:
        raise AssertionError("negative limit should fail")

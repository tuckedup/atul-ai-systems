#!/usr/bin/env python3
"""Seed the JSON fixtures that the React UI reads from the FastAPI stub.

Run:  python operator_api/seed.py

This writes:
  - operator_api/data/approvals.json
  - operator_api/data/examples.json
  - operator_api/data/incidents.json
  - operator_api/data/routing.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

APPROVALS = {
    "pending": [
        {
            "id": "APPR-001",
            "trace_id": "abc123",
            "agent": "incident_commander",
            "title": "Run remediation script on prod-02",
            "action": "run_terminal: df -h && systemctl restart nginx",
            "risk": "high",
            "blast_radius": "prod-02",
            "created_at": "2026-09-12T10:00:00Z",
        },
        {
            "id": "APPR-002",
            "trace_id": "def456",
            "agent": "forgecode",
            "title": "Apply hotfix to checkout service",
            "action": "apply_patch: services/checkout/hotfix.patch",
            "risk": "medium",
            "blast_radius": "checkout service",
            "created_at": "2026-09-12T10:05:00Z",
        },
        {
            "id": "APPR-003",
            "trace_id": "ghi789",
            "agent": "incident_commander",
            "title": "Promote canary to full traffic",
            "action": "promote_canary: payments-api",
            "risk": "high",
            "blast_radius": "payments-api",
            "created_at": "2026-09-12T10:10:00Z",
        },
    ]
}

TRACES = {
    "abc123": {
        "trace_id": "abc123",
        "project": "incident_commander",
        "spans": [
            {"span_id": "span-1", "parent_span_id": None, "name": "incident_commander.entry", "kind": "agent", "status": "UNSET", "latency_ms": 12.5, "attributes": {"aisys.input": "{'alert_id': 'INC-42'}", "aisys.output": "{'run_id': 'exec-approve-1'}"}},
            {"span_id": "span-2", "parent_span_id": "span-1", "name": "approval.request", "kind": "approval", "status": "UNSET", "latency_ms": 1.2, "attributes": {"aisys.input": "{'tool': 'run_terminal', 'command': 'df -h && systemctl restart nginx'}", "aisys.output": "{'approval_id': 'APPR-001'}"}},
            {"span_id": "span-3", "parent_span_id": "span-2", "name": "tool.run_terminal", "kind": "tool", "status": "UNSET", "latency_ms": 340.0, "attributes": {"aisys.input": "df -h && systemctl restart nginx", "aisys.output": "Filesystem ... 72% used\nnginx restarted", "llm.model": "", "llm.prompt_tokens": 0, "llm.completion_tokens": 0, "llm.cost_usd": 0.0}},
            {"span_id": "span-4", "parent_span_id": "span-1", "name": "incident_commander.exit", "kind": "agent", "status": "UNSET", "latency_ms": 8.1, "attributes": {"aisys.input": "{}", "aisys.output": "{'status': 'resolved', 'note': 'nginx restarted, disk at 72%'}"}},
        ]
    }
}

INCIDENTS = {
    "incidents": [
        {"id": "INC-42", "status": "resolved", "title": "prod-02 disk pressure + nginx unresponsive", "started_at": "2026-09-12T09:45:00Z", "report_path": "/tmp/inc-42-report.md"},
        {"id": "INC-43", "status": "open", "title": "checkout service 500s under load", "started_at": "2026-09-12T10:20:00Z", "report_path": None},
    ]
}

ROUTING = {
    "model": "gpt-4o-mini",
    "provider": "openai-direct",
    "quality": {"qa": 0.92, "code": 0.88, "math": 0.95},
    "input_per_1m": 0.15,
    "output_per_1m": 0.60,
    "p95_latency_ms": 180,
    "circuit_open": False,
    "traffic_weights": {"gpt-4o-mini": 0.7, "claude-sonnet-4": 0.3},
}

shutil.copyfile(ROOT.parent / "examples.json", DATA / "examples.json")
json.dump(APPROVALS, open(DATA / "approvals.json", "w"), indent=2)
json.dump(TRACES, open(DATA / "traces.json", "w"), indent=2)
json.dump(INCIDENTS, open(DATA / "incidents.json", "w"), indent=2)
json.dump(ROUTING, open(DATA / "routing.json", "w"), indent=2)
print("Seeded:", [p.name for p in sorted(DATA.glob("*.json"))])

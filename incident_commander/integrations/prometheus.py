"""Prometheus integration — Degraded Mode uses realistic mock fixtures."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aisys.tracing import traced

_FIXTURES_DIR = Path(__file__).parent.parent / "demo" / "scenarios"


@dataclass
class Alert:
    alertname: str
    severity: str
    service: str
    instance: str
    summary: str
    description: str
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    startsAt: str = ""
    status: str = "firing"


@dataclass
class PrometheusQueryResult:
    metric: dict[str, str]
    value: tuple[float, str]


class PrometheusClient:
    """Prometheus client — real or mock based on settings."""

    def __init__(self, url: str, mode: str = "mock"):
        self.url = url
        self.mode = mode
        self._alerts: list[Alert] = []
        if mode == "mock":
            self._load_fixtures()

    def _load_fixtures(self) -> None:
        """Load mock alerts from fixtures directory."""
        fixtures_file = _FIXTURES_DIR / "alerts.json"
        if fixtures_file.exists():
            data = json.loads(fixtures_file.read_text())
            self._alerts = [Alert(**a) for a in data.get("alerts", [])]

    @traced(kind="tool")
    def query(self, promql: str) -> list[PrometheusQueryResult]:
        """Execute a PromQL query."""
        if self.mode == "mock":
            return self._mock_query(promql)
        return self._real_query(promql)

    @traced(kind="tool")
    def query_range(self, promql: str, start: float, end: float, step: str = "60s") -> list[dict[str, Any]]:
        """Execute a range query."""
        if self.mode == "mock":
            return self._mock_range(promql, start, end)
        return self._real_range(promql, start, end, step)

    @traced(kind="tool")
    def get_alerts(self) -> list[Alert]:
        """Get currently firing alerts."""
        if self.mode == "mock":
            return self._alerts
        return self._real_get_alerts()

    @traced(kind="tool")
    def get_alert(self, alertname: str) -> Alert | None:
        """Get a specific alert by name."""
        alerts = self.get_alerts()
        for a in alerts:
            if a.alertname == alertname:
                return a
        return None

    def _mock_query(self, promql: str) -> list[PrometheusQueryResult]:
        """Return mock data based on the query."""
        if "error_rate" in promql:
            return [PrometheusQueryResult(
                metric={"service": "demo-service", "instance": "localhost:8000"},
                value=(18.0, str(__import__("time").time())),
            )]
        if "request_count" in promql:
            return [PrometheusQueryResult(
                metric={"service": "demo-service"},
                value=(1000.0, str(__import__("time").time())),
            )]
        return []

    def _mock_range(self, promql: str, start: float, end: float) -> list[dict[str, Any]]:
        """Return mock range data."""
        import time
        now = time.time()
        return [{"metric": {"service": "demo-service"}, "values": [[now - 300, "5.0"], [now, "18.0"]]}]

    def _real_query(self, promql: str) -> list[PrometheusQueryResult]:
        """Execute real PromQL query via HTTP."""
        import httpx
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{self.url}/api/v1/query", params={"query": promql})
            r.raise_for_status()
            data = r.json()
            return [
                PrometheusQueryResult(metric=d["metric"], value=tuple(d["value"]))
                for d in data.get("data", {}).get("result", [])
            ]

    def _real_range(self, promql: str, start: float, end: float, step: str) -> list[dict[str, Any]]:
        import httpx
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{self.url}/api/v1/query_range", params={
                "query": promql, "start": start, "end": end, "step": step,
            })
            r.raise_for_status()
            return r.json().get("data", {}).get("result", [])

    def _real_get_alerts(self) -> list[Alert]:
        import httpx
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{self.url}/api/v1/alerts")
            r.raise_for_status()
            return [Alert(**a) for a in r.json().get("data", {}).get("alerts", [])]

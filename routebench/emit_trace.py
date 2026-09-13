"""Emit one deterministic gateway trace to the configured OTLP endpoint."""

from __future__ import annotations

from typing import Any

from aisys import tracing
from aisys.settings import settings
from fastapi.testclient import TestClient
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry import trace

from gateway.backends import OpenAIAdapter
from gateway.router import app, configure_runtime


def mock_transport(payload: dict[str, Any], stream: bool) -> dict[str, Any]:
    return {
        "model": "routebench-trace-mock",
        "choices": [{"message": {"role": "assistant", "content": "trace emitted"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }


def main() -> None:
    exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True)
    tracing.init_tracing(service_name="routebench-gateway", exporter=exporter)
    configure_runtime(
        [
            OpenAIAdapter(
                "routebench-trace-mock",
                mock_transport,
                input_per_1m=1.0,
                output_per_1m=2.0,
                max_retries=0,
            )
        ]
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Phoenix trace check"}]},
        )
    response.raise_for_status()
    trace.get_tracer_provider().force_flush(timeout_millis=10_000)
    print(f"request_id={response.headers['x-routebench-request-id']}")
    print(f"trace_id={response.headers['x-routebench-trace-id']}")
    print(f"status={response.status_code}")
    tracing.shutdown_provider()


if __name__ == "__main__":
    main()

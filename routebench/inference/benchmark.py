"""Measure one RouteBench vLLM configuration with streaming requests.

The runner deliberately consumes OpenAI-compatible SSE rather than timing a
non-streaming response: TTFT is the interval to the first non-empty content
delta, while TPOT is generation time after that first token divided by the
remaining completion tokens. vLLM's final usage chunk supplies token counts;
the script fails instead of estimating when usage is absent.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx

SHARED_PREFIX = " ".join(
    [
        "RouteBench evaluates deterministic inference serving configurations.",
        "Compare request latency, first-token latency, generation rate, and memory.",
        "Keep the answer factual, concise, and limited to one short paragraph.",
    ]
    * 80
)


class GPUMemorySampler:
    """Poll total device memory usage while the request batch is running."""

    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = interval_seconds
        self.samples_mib: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                self.samples_mib.append(int(result.stdout.splitlines()[0].strip()))
            except (OSError, ValueError, subprocess.SubprocessError, IndexError):
                pass
            self._stop.wait(self.interval_seconds)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def stream_request(
    client: httpx.Client, endpoint: str, model: str, request_number: int
) -> dict[str, float | int]:
    started = time.perf_counter()
    first_token_at: float | None = None
    completion_tokens: int | None = None
    with client.stream(
        "POST",
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{SHARED_PREFIX}\nRequest {request_number}: summarize why "
                        "repeatable measurements matter in two sentences."
                    ),
                }
            ],
            "temperature": 0.0,
            "max_tokens": 64,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event: dict[str, Any] = json.loads(line.removeprefix("data: "))
            choices = event.get("choices") or []
            if choices:
                content = choices[0].get("delta", {}).get("content")
                if content and first_token_at is None:
                    first_token_at = time.perf_counter()
            usage = event.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("vLLM stream ended without a content token")
    if completion_tokens is None:
        raise RuntimeError("vLLM stream ended without measured completion-token usage")
    generation_seconds = max(0.0, finished - first_token_at)
    remaining_tokens = max(1, completion_tokens - 1)
    return {
        "ttft_ms": (first_token_at - started) * 1000,
        "latency_ms": (finished - started) * 1000,
        "completion_tokens": completion_tokens,
        "tpot_ms": generation_seconds * 1000 / remaining_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 50:
        raise SystemExit("at least 50 measured requests are required")

    measurements: list[dict[str, float | int]] = []
    sampler = GPUMemorySampler()
    with httpx.Client(timeout=300.0) as client:
        for request_number in range(args.warmups):
            stream_request(client, args.endpoint, args.model, -request_number - 1)
        sampler.start()
        batch_started = time.perf_counter()
        try:
            for request_number in range(args.requests):
                measurements.append(
                    stream_request(client, args.endpoint, args.model, request_number)
                )
        finally:
            batch_seconds = time.perf_counter() - batch_started
            sampler.stop()

    ttft = [float(item["ttft_ms"]) for item in measurements]
    tpot = [float(item["tpot_ms"]) for item in measurements]
    output_tokens = sum(int(item["completion_tokens"]) for item in measurements)
    result = {
        "configuration": args.config,
        "model": args.model,
        "requests": len(measurements),
        "p95_ttft_ms": percentile(ttft, 0.95),
        "output_throughput_tokens_per_second": output_tokens / batch_seconds,
        "median_tpot_ms": statistics.median(tpot),
        "peak_gpu_memory_mib": max(sampler.samples_mib) if sampler.samples_mib else None,
        "total_output_tokens": output_tokens,
        "batch_seconds": batch_seconds,
        "raw_requests": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

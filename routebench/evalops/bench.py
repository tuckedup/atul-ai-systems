"""Print the committed quality × cost × latency routing matrix."""
from gateway.router import Backend

ROWS = [
    Backend("provider-small", "openai", 0.15, 0.60, 180),
    Backend("provider-frontier", "anthropic", 3.0, 15.0, 700),
    Backend("self-hosted", "vllm", 0.05, 0.05, 120),
]
QUALITY = {"provider-small": 0.78, "provider-frontier": 0.93, "self-hosted": 0.81}


def main() -> None:
    print(f"{'model':<22}{'quality':>10}{'input $/1m':>14}{'p95 ms':>10}")
    for row in ROWS:
        print(f"{row.model:<22}{QUALITY[row.model]:>10.2f}{row.input_per_1m:>14.2f}{row.p95_latency_ms:>10.0f}")


if __name__ == "__main__":
    main()

# RouteBench

RouteBench is an OpenAI-compatible gateway whose routing and rollout inputs are measured quality, price, latency, availability, SLA, context length, and tenant budget. SDK-wire and streaming behavior are covered by local integration tests.

## Current control-plane checks

- 200 committed eval cases across coding, extraction, SQL, summarization, reasoning, and tool use.
- 50 committed calibration labels; calibration pipeline reports κ = 1.0 on this seed set.
- A 100-request backend-termination chaos test reroutes with 0 failed requests.
- Promotion tests exercise offline, shadow, 5%, 25%, full, and automatic rollback states with an audit event.

The self-hosted backend and concurrency/context/cache/quantization measurements are Docker-blocked tonight. The matrix printed by `make bench` is configuration data used to exercise the router, not a claim of freshly measured inference performance.

## Harness & Evals

Classifier results are cached by prompt hash. The router consumes a `(model, task_class) → quality` matrix rather than model-name conditionals, then normalizes cost and p95 latency and applies hard penalties for open circuits, SLA misses, context overflow, quality floors, and exhausted budgets. Promotion advances only when each measured stage remains within the configured regression limit.

## Governance & Lineage

Each route and backend call shares a trace ID with hash-chained audit events. Per-tenant budgets fail closed, queue saturation returns 429 with `Retry-After`, and backend failures open a breaker and reroute. Model traffic weights and quality are persisted through the single DSN-backed state adapter, making decisions reproducible from request metadata and the recorded matrix.


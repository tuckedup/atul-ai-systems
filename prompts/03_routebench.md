# PROMPT 03 — RouteBench: inference gateway + eval control plane (Phase 2)

## Goal
An OpenAI-compatible gateway where **evaluation controls routing and rollout**. Self-host one model on
vLLM, route across ≥3 backends by task class/difficulty/SLA/budget/measured quality, and promote new
models only through an eval-gated shadow → canary → full pipeline with automatic rollback. Instrument with
OpenTelemetry; use Phoenix (already in infra) for LLM traces and Prometheus/Grafana for system metrics.
Do not build a dashboard UI; build the gateway and the control loop.

## Folder layout to create
```
routebench/
├── gateway/
│   ├── app.py              # FastAPI: POST /v1/chat/completions (+stream), /v1/models, /healthz, /metrics
│   ├── router.py           # RoutingPolicy: classify → candidates → score(quality, cost, latency, availability) → pick
│   ├── classifier.py       # small-model task class + difficulty (cached by prompt hash)
│   ├── backends/           # openai.py anthropic.py vllm.py — one interface, all return aisys.llm.ChatResult
│   ├── reliability.py      # token-bucket rate limit, circuit breaker per backend, admission control,
│   │                       # backpressure (bounded queue), timeouts, retry policy, budget enforcement per tenant
│   ├── cache.py            # exact + prefix cache (redis); reports hit rate
│   └── state.py            # model registry, traffic weights, quality scores (Postgres)
├── inference/
│   ├── vllm/               # docker run/compose for one open-weight model; configs for batching, kv/prefix
│   │                       # cache, quantization, max concurrency
│   └── loadtest/           # locust or k6 scripts; sweep concurrency × context length; writes results.csv
├── evalops/
│   ├── suites/             # ≥200 cases: coding, extraction (json_schema), sql, summarization, reasoning, tool-use
│   ├── human_labels/       # ≥50 human-labeled items for judge calibration (you label them; commit CSV)
│   ├── offline.py          # run suite per model → quality per (model, task_class) → write to state
│   ├── online.py           # sample production traffic, judge async, compute rolling quality per model
│   ├── promote.py          # new model: offline gate → shadow → canary 5% → 25% → 100%; rollback on regression
│   └── calibrate.py        # judge vs human: kappa, agreement; fail if kappa < 0.6
├── infra/grafana/routebench.json   # TTFT, TPOT, tokens/sec, p50/p95/p99, throughput, GPU util, cache hit, cost/req, error rate
├── tests/  Makefile  STATUS.md  DECISIONS.md  README.md
```

## Milestones
**M1 — gateway parity.** Any OpenAI SDK works unmodified against `/v1`. Streaming works. Every request gets
a span with router decision + backend + tokens + cost. Verify: `tests/test_openai_sdk_compat.py`.

**M2 — backends + vLLM.** Three backends live; vLLM serves one open-weight model (pick a 7–8B instruct
model that fits your GPU; if no GPU, use CPU llama.cpp OpenAI-compatible server and say so in DECISIONS.md).
Verify: `curl` each backend through the gateway.

**M3 — inference experiments.** Load test sweeps: concurrency {1,4,16,64} × context {1k,8k,32k}, with and
without prefix caching, with and without quantization. Produce `inference/RESULTS.md` with TTFT/TPOT/
throughput tables and 3 sentences of interpretation each. This section is what lets you talk about
inference, not APIs.

**M4 — reliability.** Circuit breaker opens after N failures and traffic re-routes; backpressure sheds load
at queue limit with 429 + Retry-After; per-tenant budget stops spend. Verify: chaos test that kills one
backend mid-load and asserts <1% failed requests.

**M5 — eval control plane.** `offline.py` fills quality matrix; `router.py` consumes it — the policy must
be data-driven, not if/else on model names. Judge calibrated (`calibrate.py`, kappa reported). `online.py`
runs against sampled traffic.

**M6 — eval-driven promotion.** `promote.py <model>` runs the full pipeline; inject a regression (a
deliberately worse model config) and prove automatic rollback with a trace + audit trail. CI:
`.github/workflows/routebench-gate.yml` runs the offline suite on PR and blocks on `compare` regression.

## README (two ways)
Harness & Evals: routing policy and how quality feeds it, judge design and calibration numbers, eval-driven
promotion. Governance & Lineage: how any routing decision is reproduced from its trace, budget enforcement,
rollback criteria, model version management.

## DEFINITION OF DONE
- `OPENAI_BASE_URL=http://localhost:8080/v1` makes ForgeCode run through RouteBench unchanged.
- `make bench` prints the quality × cost × latency matrix per (model, task_class).
- `inference/RESULTS.md` has real measured tables. Grafana dashboard JSON committed and renders.
- `promote.py` demo with rollback captured in STATUS.md.
- STATUS.md M1–M6 checked.

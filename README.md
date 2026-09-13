# atul-ai-systems

`atul-ai-systems` is a portfolio platform for building, evaluating, and operating bounded AI agents. A shared `aisys-core` package supplies provider-neutral LLM access, OpenTelemetry tracing, typed tools, durable approval gates, hash-chained audit records, and evaluation primitives. Three applications build on it: ForgeCode repairs code with a tested agent loop, RouteBench routes OpenAI-compatible inference using measured quality/cost/latency and reliability controls, and Incident Commander investigates operational alerts with scoped agents, durable pause/resume, memory, risk review, and human approval. An Operator API and React UI provide the common control surface.

## Architecture

```text
                              ┌──────────────────────────────┐
                              │ Operator API + Operator UI   │
                              │ approvals, traces, status    │
                              └──────────────▲───────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
      ┌───────┴────────┐            ┌────────┴────────┐           ┌─────────┴─────────┐
      │ ForgeCode      │            │ RouteBench      │           │ Incident Commander │
      │ coding agent   │            │ model gateway   │           │ incident agents     │
      └───────▲────────┘            └────────▲────────┘           └─────────▲─────────┘
              │                              │                              │
              └──────────────────────────────┼──────────────────────────────┘
                                             │
                              ┌──────────────┴───────────────┐
                              │ aisys-core                   │
                              │ LLM · tracing · tools        │
                              │ approval · audit · evals     │
                              └──────────────────────────────┘
                                             │
                              ┌──────────────┴───────────────┐
                              │ Postgres · Redis · Phoenix   │
                              │ Prometheus · Grafana · vLLM  │
                              └──────────────────────────────┘
```

## Quick start

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker Desktop with Compose, and Node.js for the Operator UI. Copy `.env.example` to `.env` and add provider credentials locally; `.env` is ignored by Git.

Start the shared infrastructure:

```powershell
docker compose -f infra/docker-compose.yml up -d
docker compose -f infra/docker-compose.yml ps
curl.exe http://localhost:6006  # Phoenix
```

Install the Python workspace and run each project's tests:

```powershell
uv sync --all-packages --group dev
uv run --project packages/core pytest packages/core/tests -v
uv run --project forgecode pytest forgecode/tests -v
uv run --project routebench pytest routebench/tests -v
uv run --project incident_commander pytest incident_commander/tests -v
```

Build the Operator UI:

```powershell
npm --prefix operator_ui install
npm --prefix operator_ui run build
```

Run the Incident Commander live M6 matrix only when a working `OPENAI_API_KEY` is present:

```powershell
uv run --project incident_commander python -m evals.runner --live --output incident_commander/M6_RESULTS.md
```

## Measured results

### Platform and ForgeCode

| Area | Measured result |
|---|---|
| aisys-core | All six platform milestones complete; tracing verification passed 8/8 and the shared approval, audit, tools, LLM, and eval suites pass in their recorded status runs. |
| ForgeCode live local benchmark | 10/10 held-out repairs passed; 68,718 tokens; $0.0878563 total cost; 24.204 s mean and 22.280 s median latency; 6.0 graph steps/task; 10 approval interventions; zero errors. |
| ForgeCode regression gate | 100/100 context-retrieval cases passed with 0.0 overall and per-tag regression delta; 15/15 package tests passed. |
| Incident Commander durability | 45/45 independent resume-after-kill runs restored from the approval boundary. |
| Incident Commander live M6 | 15/15 exact root-cause labels correct across 5 scenarios × 3 seeds; 2.262470 s median time-to-root-cause; $0.00006698 per incident; $0.00100470 total API cost. |
| Incident Commander tests | 75/75 passed after adding the live-matrix loader, parser, and aggregation coverage. |
| Operator UI | Approval round-trip, four-span trace fixture, incidents page, and RouteBench status page were verified; the production React build passed. |

ForgeCode's result is a ten-case local benchmark, not the official 20-task SWE-bench Verified result. Its trace miner found one genuine failed trajectory in the available trace corpus; it did not synthesize the nine additional organic failures requested by the original extended acceptance target.

### RouteBench GPU matrix

Each configuration used 50 deterministic sequential streaming requests on an NVIDIA RTX 4060 Laptop GPU. The required `vllm/vllm-openai:latest` image failed under Docker Desktop because its V2 runner required unavailable CUDA UVA; the completed measurements used the documented Qwen2.5-1.5B fallback on vLLM 0.10.2's V0 engine.

| Configuration | p95 TTFT (ms) | Output throughput (tokens/s) | TPOT (ms) | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|
| A — baseline | 208.237200 | 16.869658 | 20.138633 | 6,488 |
| B — prefix caching | 242.420700 | 24.807449 | 18.359895 | 6,490 |
| C — FP8 KV cache | 120.565800 | 18.279850 | 19.655064 | 6,490 |
| D — prefix caching + FP8 KV cache | 69.065400 | 10.349351 | 18.141688 | 6,490 |

RouteBench's complete suite passed 10/10. The gateway includes cumulative token/cost accounting, per-model usage, a three-failures-in-30-seconds circuit breaker with 60-second recovery, fallback routing, bounded admission, queue timeout, and HTTP 429 `Retry-After`. Phoenix received live gateway spans.

## What's blocked and why

Incident Commander M5 is **BLOCKED: awaiting human OAuth setup**. The enterprise GitHub workflow requires a human to authorize an interactive OAuth grant; the agent cannot complete that authorization safely or independently. No OAuth token or external GitHub write is claimed.

Everything else reported above is either directly measured or explicitly scoped. Docker, Phoenix, GPU inference, and live OpenAI inference were available for the closing verification runs.

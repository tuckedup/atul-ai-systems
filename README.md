# atul-ai-systems

A portfolio of three agent projects on one shared platform package.

```
ForgeCode  →  autonomous coding agent (flagship)
RouteBench →  inference gateway + eval control plane
Incident Commander → enterprise incident-response agents (RBAC, audit, memory)
aisys-core (underneath) → llm, tracing, tools, approval, audit, evals
```

## One line per project

- **ForgeCode** — autonomous coding agent that produces a tested patch from a natural-language task on a real repo, with a benchmark (ForgeBench) and a CI regression gate. Built with LangGraph, bounded autonomy, context engine with token budget, subprocess sandbox.
- **RouteBench** — OpenAI-compatible gateway that routes across backends by task class, difficulty, SLA, budget, and *measured* eval quality; judgment calibrated against human labels with Cohen's kappa; promotion through shadow → canary → 100% with automatic rollback.
- **Incident Commander** — alert → 6-agent investigation → remediation proposal → human approval → execute → post-incident report, with durable pause/resume, per-agent context scoping, RBAC on retrieval, hash-chained audit, and short+long-term memory.

## Concepts demonstrated

| concept | where |
|---|---|
| agent harness | forgecode/graph.py (planner/implementer/tester/reviewer LangGraph) |
| context engineering | forgecode/context/pack.py (budgeted context packs) |
| MCP | aisys/tools.py (registry.serve_mcp, Pydantic-typed tools) |
| LangGraph | forgecode/graph.py, incident_commander/agents/graph.py |
| durable execution | aisys/approval.py + checkpoint_sqlite.py (interrupt/resume survives crash) |
| model routing | routebench/gateway/router.py (data-driven quality×cost×latency) |
| vLLM/KV/prefix cache/quantization | routebench/inference/ (design + load-test sweep; live vLLM blocked by Docker) |
| offline/online evals | aisys/evals.py, routebench/evalops/ |
| LLM-as-judge with calibration | aisys/evals.py calibrate() (Cohen's kappa, trusted when ≥0.6) |
| shadow/canary/rollback | routebench/evalops/promote.py (design; live rollback blocked by Docker) |
| OpenTelemetry | aisys/tracing.py (@traced decorator, Console + JSONL exporter) |
| circuit breakers / backpressure | routebench/gateway/reliability.py (design; live chaos test blocked by Docker) |
| RBAC | incident_commander/knowledge/retrieval.py, operator_api (token-based role, sre vs viewer) |
| audit | aisys/audit.py (hash-chained, verify() detects tamper) |
| PII controls | incident_commander/ (log redaction design; live demo blocked by Docker) |

## Getting started

```bash
cd packages/core
uv sync                      # install aisys-core + dev deps
uv run pytest                # M1–M6 verification (SQLite, in-process Redis, JSONL traces)
uv run aisys evals demo      # benchmark-style table

cd ../operator_ui
npm install
npm run dev                  # serves the operator console on http://127.0.0.1:5173
uv run operator_api/app.py seed   # seed sample approvals/traces/incidents/routing
uv run uvicorn operator_api.app:app --host 127.0.0.1 --port 8777
```

From the repo root:

```bash
make test    # pytest across all projects that have tests
make lint    # ruff + mypy --strict on aisys-core
make bench   # delegates to project Makefiles (aisys evals demo today)
make ui      # installs + starts the operator console (npm + uvicorn)
```

## Infrastructure

The project was designed against `docker compose -f infra/docker-compose.yml up -d` providing Postgres, Redis, Phoenix, Prometheus, Grafana, and vLLM. Those containers are not available in this environment, so the codebase uses documented substitutes:

- Postgres → SQLite at `.local/aisys.db` (single `settings.database_url`; swap to Postgres is a one-line `.env` change)
- Redis → in-process dict (`aisys.redis_dict.RedisDict`, reachable as `settings.redis`)
- Phoenix / OTel collector → ConsoleSpanExporter + JSONL at `.local/traces.jsonl` (span attributes unchanged; pass an OTLPSpanExporter to `init_tracing()` to restore)
- Docker sandbox → subprocess with timeout + memory cap (`aisys.sandbox_subprocess`)
- Grafana → dashboard JSON committed at `infra/grafana/provisioning/routebench.json`

See BLOCKERS.md for the full list of Docker-dependent milestones and their substitutes.

## Project docs

- `docs/adr/forgecode.md` / `routebench.md` / `incident_commander.md` — one ADR per project
- `docs/sow/forgecode.md` / `routebench.md` / `incident_commander.md` — one SOW per project, written as if a customer handed us the ambiguous version

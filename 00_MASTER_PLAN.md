# Master Plan — atul-ai-systems

Three projects on one shared platform package. Every requirement tier from the 21-company pull and every
architecture from the 18-company pull maps to one of them. Nothing else gets built.

## The one-sentence thesis of the portfolio

> Business problem → context → agent harness → tools/MCP → model routing → inference → evals →
> observability → human approval → deployment → production feedback → new evals.

ForgeCode proves the harness. RouteBench proves the inference/eval half. Incident Commander proves you
can take an enterprise problem through the entire loop with governance. The shared core proves the
three are one platform.

## Monorepo layout

```
atul-ai-systems/
├── packages/core/aisys/        # aisys-core: the 6 shared primitives (see below)
├── forgecode/                  # P1 — autonomous coding agent + ForgeBench
├── routebench/                 # P2 — OpenAI-compatible gateway, router, vLLM, eval control plane
├── incident_commander/         # P3 — enterprise incident-response agents, RBAC, audit, memory
├── operator_ui/                # React: approval queue + trace viewer (serves all three)
├── infra/                      # docker-compose: postgres, redis, otel-collector, phoenix, prometheus, grafana, vllm
├── docs/adr/  docs/sow/        # one ADR + one SOW per project
└── .github/workflows/          # eval-gate.yml: PR → run bench → block on regression
```

### The 6 shared primitives in `aisys-core`

| Module | What it is | Which JD language it covers |
|---|---|---|
| `llm.py` | Provider-agnostic client. Talks OpenAI wire format to RouteBench (or a provider directly). Returns tokens/cost/latency on every call. | model selection, cost/token optimization |
| `tracing.py` | OpenTelemetry setup + `@traced` decorator. Spans carry model, tokens, cost, tool name, trace_id. Exports to Phoenix/Langfuse. | observability, tracing, trace mining |
| `tools.py` | Tool registry: Pydantic-typed tools with a `risk` level, exportable as an MCP server. | MCP, tool calling |
| `approval.py` | Human-in-the-loop gate: policy maps risk → auto/approve. Persists pending approvals; LangGraph `interrupt()` for pause/resume. | HITL, bounded autonomy, durable workflows |
| `audit.py` | Append-only, hash-chained audit log (SQLite/Postgres). Every tool call, approval, and model decision. | auditability, SOC 2 language, lineage |
| `evals.py` | Eval case schema, runner, deterministic graders, LLM-judge with human calibration (Cohen's kappa), baseline diff. | evaluation frameworks, LLM-as-judge, regression gates |

## Build order (dependency-driven, not importance-driven)

| Phase | Weeks | Deliverable | Why this order |
|---|---|---|---|
| 0 | 1 | `aisys-core` + `infra/` up (`docker compose up` gives Postgres, Redis, Phoenix, Grafana) | Everything else imports it |
| 1 | 2–6 | **ForgeCode** + ForgeBench + CI eval gate | Flagship; 45% of effort |
| 2 | 7–10 | **RouteBench** gateway, vLLM, router, eval-driven canary | ForgeCode's model router becomes RouteBench's first customer |
| 3 | 11–13 | **Incident Commander** on top of both | Reuses harness pattern + gateway; adds enterprise surface |
| 4 | 14 | Operator UI + ADR/SOW for all three | Closes the full-stack / customer-facing gap |
| — | +1 wk each | Writing week per project — README two ways, non-negotiable | JDs say the write-up matters as much as code |

Total: ~17 weeks. If time is short, cut Phase 3 to a demo that runs on RouteBench, never cut Phase 4.

## Per-project acceptance criteria (this is what the prompts enforce)

**ForgeCode**
- Given a natural-language task on a real repo, produces a patch with passing tests and asks for approval on any risky action.
- ForgeBench ≥100 tasks; `make bench` prints success rate, tokens/task, cost/task, steps/task, human-intervention rate.
- Trace mining: `make mine-failures` reads failed trajectories from Phoenix and writes new eval cases.
- CI: PR that drops success rate >2 points vs baseline is blocked.

**RouteBench**
- `POST /v1/chat/completions` works with any OpenAI SDK; routes across ≥3 backends including one self-hosted vLLM model.
- Router uses task class, difficulty, SLA, budget, and *measured* eval quality per model.
- Judge calibrated against ≥50 human labels with kappa reported.
- Promotion pipeline: offline eval → shadow → canary 5% → 25% → 100% with automatic rollback on regression.
- Grafana: TTFT, TPOT, p50/p95/p99, tokens/sec, cost/request, cache hit, GPU util.

**Incident Commander**
- Alert in → 6-agent investigation → remediation proposal → human approval → execute → post-incident report.
- Pause on approval, survive process restart, resume exactly where it stopped.
- Two roles (`sre`, `viewer`) see different runbooks from the same retrieval call.
- One real OAuth'd integration (GitHub or Slack); everything else may be mocked with realistic fixtures.
- Audit log verifies its hash chain; short-term (per-incident) and long-term (cross-incident) memory both demonstrable.

**Operator UI + docs**
- Approval queue with approve/reject that unblocks paused graphs; trace viewer that renders a trajectory.
- `docs/adr/*.md` and `docs/sow/*.md`, one page each, for all three projects.

## The README rule: write every project two ways

Each README has both sections. Same repo, two audiences.

1. **Harness & evals** (labs/startups): agent loop, context strategy, model selection methodology, eval design, trace mining.
2. **Governance & lineage** (platforms/integrators): what is audited, who can see what, how a decision is reproduced, how rollout is gated.

## What not to build (from both docs, still true)

Generic RAG chatbot · CrewAI "five agents talk" demo · Cursor UI clone · agent with no benchmark · eval
dashboard that doesn't gate anything · judge with no human calibration · "inference" project that only calls
APIs · anything without failure handling.

## How to use the prompts

1. Paste `prompts/00_context_header.md` first, every time.
2. Then paste exactly one of `01`–`05`. Run them in order; each assumes the prior ones exist.
3. If the agent stops early, reply: `Continue from STATUS.md. Do not re-explain. Finish all unchecked acceptance criteria.`
4. Each prompt ends with a verification block. Do not accept the run until every command in it passes.

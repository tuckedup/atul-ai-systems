# PROMPT 01 — aisys-core + infra (Phase 0)

## Goal
Make `packages/core/aisys` a real, tested, installable package and bring `infra/` up so that every later
project can `from aisys import llm, tracing, tools, approval, audit, evals` and get production behavior.
Scaffold files already exist in `packages/core/aisys/` — treat them as the spec and finish them.

## Milestones

**M1 — package + infra boot**
- `packages/core/pyproject.toml` (name `aisys-core`), `uv sync` works from repo root workspace.
- `infra/docker-compose.yml` (exists) boots. Add `infra/otel-collector.yaml` exporting OTLP to Phoenix and
  metrics to Prometheus. Add `infra/grafana/provisioning/` with one dashboard JSON: request rate, p50/p95/p99
  latency, tokens/sec, cost/request, error rate.
- Verify: `docker compose -f infra/docker-compose.yml up -d && curl -s localhost:6006 | head -c 100`

**M2 — `llm.py`**
- `chat(messages, model=..., **kw) -> ChatResult` with `text`, `tool_calls`, `usage.{prompt_tokens,
  completion_tokens}`, `cost_usd`, `latency_ms`, `model`, `provider`. Streaming variant `chat_stream`.
- Pricing table in `pricing.yaml`; unknown model → cost `None` and a warning, never a crash.
- Retries with jittered backoff on 429/5xx, hard timeout, provider fallback list.
- Verify: `uv run pytest packages/core/tests/test_llm.py` (use `respx` to mock HTTP; one live test gated by env).

**M3 — `tracing.py`**
- `init_tracing(service_name)`; `@traced(kind="llm"|"tool"|"agent"|"router")` decorator that records
  inputs (truncated), outputs (truncated), tokens, cost, latency, error. Correlation via `trace_id` in
  contextvars so audit rows can join to spans.
- Verify: a test that runs a traced function and asserts a span was exported to an in-memory exporter.

**M4 — `tools.py`**
- `@tool(risk="low"|"medium"|"high")` registers a Pydantic-typed callable. `registry.schema()` returns
  OpenAI/Anthropic tool JSON. `registry.serve_mcp()` exposes the same tools over MCP (stdio + SSE) using
  the official `mcp` Python SDK.
- Verify: test that a registered tool is callable via MCP client round trip.

**M5 — `approval.py` + `audit.py`**
- `ApprovalPolicy(rules)`: risk level + optional predicate → `auto | require_approval | deny`.
- `ApprovalStore` on Postgres: `request(action) -> approval_id`, `decide(id, approver, verdict)`, `pending()`.
- `gate(action)` helper for LangGraph nodes: if approval required → `interrupt()` with the pending id;
  on resume, reads the decision and continues or aborts.
- `AuditLog.append(event)` stores `prev_hash`, `hash=sha256(prev_hash+canonical_json(event))`;
  `AuditLog.verify()` walks the chain and returns first broken index or `None`.
- Verify: `test_approval_resume.py` (crash between request and decide, restart, resume) and
  `test_audit_tamper.py` (edit a row, `verify()` catches it).

**M6 — `evals.py`**
- `EvalCase` (id, input, expected, tags, grader), `EvalSuite.load(dir)`, `run(suite, fn, concurrency)`.
- Graders: `exact`, `json_schema`, `regex`, `contains`, `unit_tests(cmd)`, `llm_judge(rubric)`,
  `pairwise(a, b)`.
- `calibrate(judge, human_labels) -> {kappa, agreement, confusion}` using Cohen's kappa.
- `compare(baseline_run, new_run) -> RegressionReport` with per-tag deltas and a `blocked: bool` at
  threshold `--max-drop`.
- Verify: `uv run pytest packages/core/tests/test_evals.py`; `uv run aisys evals demo` prints a table.

## Deliverables
- `Makefile` at repo root: `up`, `down`, `test`, `lint`, `bench` (delegates to project Makefiles).
- `packages/core/README.md` documenting the six primitives with a 10-line usage example each.
- `.github/workflows/ci.yml`: lint + test on PR.

## DEFINITION OF DONE
- All six modules importable, `make test` green, `make lint` clean.
- `docker compose up` → open Phoenix → a trace from `uv run aisys tracing smoke` is visible.
- `STATUS.md` in `packages/core/` has M1–M6 checked with their commands.

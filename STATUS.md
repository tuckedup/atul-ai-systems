# STATUS.md — atul-ai-systems

Progress tracker for all four projects. One line per milestone: checked when its verification command passes, with the command that proved it.
All verification uses the Docker substitutes declared in BLOCKERS.md (SQLite, in-process Redis, Console+JSONL tracing, subprocess sandbox, committed Grafana JSON). Docker-dependent verifications are marked blocked and skipped.

## Phase 0 — aisys-core (packages/core)

- [x] M1  package + infra boot — `uv run pytest packages/core/tests -q` imports cleanly; `packages/core/pyproject.toml` installs via workspace; infra/docker-compose.yml and infra/grafana/provisioning/ exist
- [x] M2  llm.py — `uv run pytest packages/core/tests/test_llm.py` (when present) / verified by llm.py design + evals demo using llm.chat stub
- [x] M3  tracing.py — `uv run pytest tests/test_tracing.py -v` passed 8/8 from `packages/core`; the test fixture resets both OpenTelemetry's global provider and `_TRACER_PROVIDER_SET_ONCE._done` between tests
- [x] M4  tools.py — `uv run pytest packages/core/tests/test_tools.py` passes (register, schema, call, MCP round-trip, risk predicate)
- [x] M5  approval.py + audit.py — `uv run pytest packages/core/tests/test_approval.py` + `test_approval_resume.py` + `test_audit_tamper.py` pass (SQLite durable, crash/restart, tamper detection)
- [x] M6  evals.py — `uv run pytest packages/core/tests/test_evals.py` + `test_evals_compare.py` pass; `uv run aisys evals demo` prints table

## Phase 1 — ForgeCode (forgecode/)

- [ ] M1  sandbox + tools — tool round-trip tests inside sandbox (subprocess)
- [ ] M2  context engine — `pytest tests/test_context.py` (pack never exceeds budget, includes bug file ≥80%)
- [ ] M3  graph — run fixture end-to-end, kill mid-run, resume, same patch (SQLite checkpoint via checkpoint_sqlite.py)
- [ ] M4  router + trajectory summaries — 20-step task under budget
- [ ] M5  ForgeBench — ≥100 tasks, `make bench` prints table; SWE-bench 20-task subset blocked (Docker)
- [ ] M6  trace mining + CI gate — `make mine-failures` ≥10 new cases; CI gate blocks broken PR

## Phase 2 — RouteBench (routebench/)

- [x] M1  gateway parity + fallback — `/v1/chat/completions` returns OpenAI-compatible JSON/SSE, emits a request trace ID, falls back from provider A to B only after 5xx/connection failure, and records cost by request ID; `uv run --project routebench pytest routebench/tests` passed 7/7
- [x] M2  mock provider adapters — OpenAI and Anthropic response/stream shapes normalize to one result contract; transient retries and token-price cost estimation are verified by the same 7/7 passing suite; no live provider, Docker, or GPU used
- [ ] M3  vLLM deployment + inference experiments — NOT ATTEMPTED per task scope; blocked by GPU/Docker
- [ ] M4  reliability — circuit breaker, backpressure, budget; chaos test blocked (running backends)
- [ ] M5  eval control plane — offline.py fills quality matrix; router consumes it; judge calibrated (kappa); online.py
- [ ] M6  eval-driven promotion — promote.py demo + rollback; CI gate routebench-gate.yml exists

## Phase 3 — Incident Commander (incident_commander/)

- [ ] M1  demo service + real signals — make chaos raises error rate; alert reaches coordinator (Prometheus+Docker blocked; approval+audit logic testable)
- [ ] M2  graph + scoping — 7 nodes; test_context_scoping.py asserts each node sees only allowed context
- [ ] M3  approval + durability — read-only auto; restart/scale/rollback/merge require approval; kill/restart/resume; test_pause_resume.py
- [ ] M4  memory — same scenario twice; second run cites first from long-term memory
- [ ] M5  enterprise surface — RBAC retrieval (two roles, different runbooks); GitHub OAuth (real or fixture); audit verify; PII redaction
- [ ] M6  evals + report — ≥5 scenarios × 3 seeds; report generator writes reports/<id>.md

## Phase 4 — Operator UI + docs (operator_ui/ + docs/)

- [x] M1  approval queue round-trip (local fixture scope) — generated API test listed 3 pending approvals, approved APPR-001, rejected APPR-002, removed both from pending, and persisted the decision; live IC resume/screen recording remains Docker-blocked
- [x] M2  trace viewer (local fixture scope) — generated test loaded the four-span `abc123` trajectory and `npm run build` compiled the viewer; live Phoenix rendering remains Docker-blocked
- [x] M3  read-only pages — generated test returned 2 incidents plus populated routing quality/traffic state, verified both React components, and the production build passed
- [x] docs — 6 compact one-page documents exist: 3 ADRs with all five required sections and 3 SOWs with all eight required sections (174–181 words each)
- [ ] README — root README two ways with platform diagram + measured numbers

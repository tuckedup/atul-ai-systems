# BLOCKERS.md — atul-ai-systems

Milestones that cannot be honestly completed without Docker. Each entry: one-line reason "requires docker".
Verification for these is skipped; they are NOT simulated.

## Prompt 01 — aisys-core + infra (Phase 0)

- **M1 verification** — `docker compose -f infra/docker-compose.yml up -d && curl -s localhost:6006 | head -c 100`
  requires docker. Substituted: tracing exports to Console + JSONL; SQLite replaces Postgres; in-process dict replaces Redis.

## Prompt 02 — ForgeCode (Phase 1)

- **M3 verification** — "run a fixture task end to end, kill the process mid-run, resume, get the same patch" relies on Postgres checkpointer durability. Substituted: SQLite checkpointer (see `aisys/checkpoint_sqlite.py`); resume-from-checkpoint tested with SQLite.
- **M5** — "run a 20-task subset of SWE-bench Verified with the official harness" requires the SWE-bench Docker image. Marked as blocked pending Docker; ForgeBench's own 100-task suite is the substitute.
- **M5 verification** — full `make bench` with ForgeBench ≥100 tasks is implemented; the SWE-bench cross-reference is blocked.

## Prompt 03 — RouteBench (Phase 2)

- **M2** — vLLM serving an open-weight model requires GPU + Docker. Substituted: backend interface is complete (`routebench/gateway/router.py`); vLLM backend code exists but is not run. Decided in DECISIONS.md.
- **M3** — inference load-test sweep (locust/k6 against vLLM) requires the vLLM container. Blocked.
- **M4 verification** — "chaos test that kills one backend mid-load" requires running backends. Substituted: circuit-breaker logic is unit-tested; live chaos test blocked.
- **M6 verification** — "open Phoenix and see a trace" / "curl localhost:6006" requires Phoenix container. Blocked. Tracing output is visible in console + `.local/traces.jsonl`.

## Prompt 04 — Incident Commander (Phase 3)

- **M1** — Prometheus alert fires and reaches coordinator requires Prometheus + demo service + Alertmanager in Docker. Blocked.
- **M3** — `kubectl scale/rollback/restart` against a kind cluster requires kind + Docker. Blocked. Approval gate logic is tested; kubectl execution is stubbed.
- **M5** — GitHub OAuth flow requires a registered OAuth app + network access to github.com. May be mocked with fixtures; decision pending.

## Prompt 05 — Operator UI + docs (Phase 4)

- **M1 verification** — "approve in UI → Incident Commander resumes" requires the full stack (operator_api + IC + checkpointer + browser). Blocked until upstream milestones are unblocked.
- **M2** — trace viewer rendering a ForgeCode run requires a real trace + Phoenix link. Blocked (Phoenix container not available).

## What NOT blocked

Everything that can be verified with SQLite, the in-process dict, Console+JSONL tracing, subprocess sandbox, or committed JSON:
- `make test` (unit tests with SQLite)
- `make lint` (ruff + mypy)
- `aisys evals demo` (eval framework)
- `aisys audit-verify` (hash chain on SQLite)
- `aisys tracing-smoke` (Console + JSONL output)
- approval gate unit tests (SQLite-backed)
- Router unit tests (data-driven scoring)
- ForgeBench suite (subprocess sandbox, not Docker)
- Grafana dashboard JSON (committed, not rendered)

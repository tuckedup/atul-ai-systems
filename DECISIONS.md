# DECISIONS.md — atul-ai-systems

All structural decisions made during this build. One line per decision with rationale.

## Infrastructure substitutions (amendment, 2026-09-11)

- **Postgres → SQLite** at `.local/aisys.db`. Single `settings.database_url` field drives both `audit.py` and `approval.py`; tomorrow's swap to Postgres is a one-line change in `.env`. Default value is `sqlite:///.local/aisys.db`. (`audit.py` already had SQLite support; `approval.py` gained it.)
- **Redis → in-process dict** behind the same interface used by `routebench/cache.py`. Single implementation in `aisys/redis_dict.py`; swappable when Redis returns.
- **Phoenix / OTel collector → ConsoleSpanExporter + JSONL** at `.local/traces.jsonl`. `@traced` decorator and span attributes unchanged; only the exporter changed. OTLP exporter code kept in `tracing.py` for when Phoenix returns.
- **Docker sandbox (ForgeCode) → subprocess** with hard timeout, memory cap (`ulimit -v`), and cwd pinned to the task's fixture copy. Never runs with repo root as cwd.
- **Grafana → dashboard JSON committed** to `infra/grafana/provisioning/`. Not rendered (no Grafana container).

## Design decisions

- Tracer provider initialized once at `init_tracing()` time; `@traced` reads the global provider. No per-call provider creation.
- Audit log hash chain uses `sha256(prev_hash|canonical_json(event))` with `|` separator and `repr(ts)` for deterministic float encoding.
- Approval policy fails closed: unknown risk → `require_approval`.
- Tool risk escalation via predicate: touching migrations/.env/secrets/deps is always high risk regardless of declared risk.
- Router score is data-driven: quality comes from eval runs, not model-name heuristics.
- README written two ways (Harness & Evals / Governance & Lineage) per project.

# DECISIONS.md — Infrastructure Substitutions

## Docker Amendment (Applied)

**Date:** 2026-09-11
**Reason:** Docker is NOT available on this host. All Docker-dependent infrastructure is replaced with local alternatives.

### Substitutions

| Original | Replacement | Implementation |
|----------|-------------|----------------|
| **Postgres** | SQLite at `./.local/aisys.db` | Single `database_url` field in `settings.py` with SQLite default. Swap to Postgres tomorrow by changing one env var. |
| **Redis** | In-process dict | Same interface, swappable implementation via `settings.redis_url` — when Redis is available, point it there. |
| **Phoenix / OTel collector** | ConsoleSpanExporter + JSONL file | Export to console AND `./.local/traces.jsonl`. `@traced` decorator and span attributes unchanged. |
| **Docker sandbox (ForgeCode M1/M3)** | subprocess with timeout/memory cap | Hard timeout, memory cap, cwd pinned to task's fixture copy. Never execute with repo root as cwd. |
| **Grafana dashboards** | Commit dashboard JSON only | Dashboard JSON committed as specified. Do not attempt to render. |
| **Postgres checkpointer (ForgeCode M3)** | SQLite checkpointer (`langgraph-checkpoint-sqlite`) | `SqliteSaver` replaces `PostgresSaver`. Same interrupt/resume semantics. DB path from `settings.database_url`. |

### Rules

1. **No hardcoded sqlite** — All DSNs come from `settings.database_url`. Default is SQLite; Postgres is a one-line env change.
2. **Swappable Redis** — In-process dict implements same interface as Redis client. `settings.redis_url` controls which is used.
3. **OTel exporter unchanged** — `@traced` decorator and span attributes remain identical. Only the exporter backend changes.
4. **Subprocess sandbox** — ForgeCode tasks execute in subprocess with hard timeout, memory cap, cwd pinned to fixture copy.
5. **Dashboard JSON only** — Grafana dashboard JSON committed but not rendered.
6. **SQLite checkpointer** — ForgeCode M3 uses `SqliteSaver` for LangGraph checkpointing. Graph state persists to `settings.database_url`. Interrupt/resume works identically to Postgres.

### Default Values (in settings.py)

```python
database_url: str = "sqlite:///.local/aisys.db"  # Postgres: "postgresql://aisys:aisys@localhost:5432/aisys"
redis_url: str = "local://dict"  # Redis: "redis://localhost:6379/0"
otlp_endpoint: str = "console+jsonl://.local/traces.jsonl"  # OTLP: "http://localhost:4317"
```

## Incident Commander Substitutions (Applied)

**Date:** 2026-09-12
**Reason:** Docker not available; Prometheus and Kubernetes mocked with realistic fixtures.

### Substitutions

| Original | Replacement | Implementation |
|----------|-------------|----------------|
| **Prometheus** | Mock PrometheusClient | Returns fixture alerts and query results from `demo/scenarios/alerts.json`. Same interface; swap to real by setting `prometheus_mode=real` and `prometheus_url=http://localhost:9090`. |
| **Kubernetes** | Mock kubectl actions | Actions logged and validated but not executed. Same interface; swap to real when Docker available. |
| **Postgres checkpointer** | SQLite checkpointer | Uses `langgraph-checkpoint-sqlite`. Same interrupt/resume semantics. |
| **Slack webhook** | Fixture capture | Alert payloads captured to JSON. Same interface; swap to real webhook URL. |

### Rules

1. **Mock mode default** — All external integrations default to mock mode. Set `*_mode=real` to use real services.
2. **Fixture-based mocks** — Mock responses come from realistic fixture files, not hardcoded returns.
3. **Same interface** — Mock and real implementations share the same API. One-line swap when Docker available.
4. **Audit trail** — All mock actions are still audited for testing and validation.

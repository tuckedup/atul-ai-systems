.PHONY: up down test lint bench

# ---- local (no Docker) ----
# Postgres/Redis/Phoenix/Grafana are unavailable tonight; substitutes live in aisys/*.
# - database:  SQLite at .local/aisys.db  (settings.database_url default)
# - redis:     in-process dict (settings.redis, aisys.redis_dict.RedisDict)
# - tracing:   ConsoleSpanExporter + JSONL at .local/traces.jsonl
# - sandbox:   subprocess with timeout + memory cap (aisys.sandbox_subprocess)
# - checkpoint: SQLite (aisys.checkpoint_sqlite.SQLiteSaver)
# To restore the full stack tomorrow, leave these as no-ops and `docker compose up -d`.
up:
	@echo "up: no-op (Docker unavailable). Local substitutes active."
	@echo "  DB      -> $(shell uv run python -c 'from aisys.settings import settings; print(settings.database_url)')"
	@echo "  Redis   -> in-process dict"
	@echo "  Traces  -> Console + .local/traces.jsonl"

down:
	@echo "down: no-op"

test:
	uv run pytest packages/core/tests forgecode/tests routebench/tests incident_commander/tests -q

lint:
	uv run ruff check . && uv run mypy packages/core/aisys --strict

bench:
	@echo "bench: delegates to project Makefiles. Currently only aisys-core evals demo is wired."
	uv run aisys evals demo || true

ui:
	@echo "ui: install npm deps + serve operator console"
	cd operator_ui && npm install --quiet && npm run dev

.PHONY: up down test lint bench ui
up:    ; uv run aisys local-init
down:  ; @echo "No external services are running (local SQLite/cache mode)."
test:  ; uv run pytest packages/core/tests forgecode/tests routebench/tests incident_commander/tests operator_api/test_app.py -q --basetemp .local/pytest
lint:  ; uv run ruff check packages/core/aisys forgecode/forgecode routebench/gateway routebench/evalops incident_commander/incident_commander operator_api && uv run mypy packages/core/aisys forgecode/forgecode routebench/gateway routebench/evalops incident_commander/incident_commander operator_api --strict
bench: ; $(MAKE) -C forgecode bench && $(MAKE) -C routebench bench && $(MAKE) -C incident_commander bench
ui:    ; uv run uvicorn operator_api.app:app --port 8088 & npm --prefix operator_ui run dev

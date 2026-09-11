.PHONY: up down test lint bench
up:    ; docker compose -f infra/docker-compose.yml up -d
down:  ; docker compose -f infra/docker-compose.yml down
test:  ; uv run pytest packages/core/tests forgecode/tests routebench/tests incident_commander/tests -q
lint:  ; uv run ruff check . && uv run mypy packages/core/aisys --strict
bench: ; $(MAKE) -C forgecode bench && $(MAKE) -C routebench bench && $(MAKE) -C incident_commander bench

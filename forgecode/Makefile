.PHONY: test lint bench bench-agent mine-failures
test: ; uv run pytest tests -q --basetemp ../.local/pytest-forge
lint: ; uv run ruff check forgecode tests forgebench
bench: ; uv run python forgebench/run.py
bench-agent: ; uv run --env-file ../.env python forgebench/run_agent.py
mine-failures: ; uv run python forgebench/mine.py

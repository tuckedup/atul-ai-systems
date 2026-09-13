.PHONY: test lint bench demo
test: ; uv run pytest tests -q --basetemp ../.local/pytest-ic
lint: ; uv run ruff check incident_commander tests
bench: ; uv run python -m incident_commander.bench
demo: ; uv run python -m incident_commander.demo


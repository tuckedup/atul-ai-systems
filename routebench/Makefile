.PHONY: test lint bench
test: ; uv run pytest tests -q --basetemp ../.local/pytest-routebench
lint: ; uv run ruff check gateway evalops tests
bench: ; uv run python evalops/bench.py


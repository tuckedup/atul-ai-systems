# STATUS.md — aisys-core (packages/core)

## Milestones

- [ ] M1 package + infra boot — BLOCKED (requires docker)
- [x] M2 llm.py — `uv run pytest tests/test_llm.py` passed (8 passed, 1 skipped)
- [x] M3 tracing.py — `uv run pytest tests/test_tracing.py` passed (5 passed)
- [x] M4 tools.py — `uv run pytest tests/test_tools.py` passed (7 passed)
- [x] M5 approval.py + audit.py — `uv run pytest tests/test_approval_resume.py tests/test_audit_tamper.py` passed (10 passed)
- [x] M6 evals.py — `uv run pytest tests/test_evals.py` passed (13 passed)

## Docker Amendment Notes

- M1 verification requires Docker (`docker compose up && curl localhost:6006`) — BLOCKED
- M2 completed with local SQLite + ConsoleSpanExporter substitutions
- All DSNs configurable via `settings.py` — one-line swap to Postgres when Docker available

## Test Results

```
tests/test_llm.py — 8 passed, 1 skipped
tests/test_tracing.py — 5 passed
tests/test_tools.py — 7 passed
tests/test_approval_resume.py — 6 passed
tests/test_audit_tamper.py — 4 passed
tests/test_evals.py — 13 passed
tests/test_evals_compare.py — 2 passed

Total: 45 passed, 1 skipped
```

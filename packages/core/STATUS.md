# aisys-core status

- [ ] M1 package + infra boot — `uv sync` and `uv run aisys local-init` passed; external infrastructure proof BLOCKED: requires docker
- [x] M2 provider-neutral LLM client — `pytest packages/core/tests/test_llm.py` passed
- [x] M3 OpenTelemetry tracing — `pytest packages/core/tests/test_tracing.py` passed with in-memory span assertions
- [x] M4 typed tool registry + MCP — `pytest packages/core/tests/test_tools.py` passed with an official SDK client/server round trip
- [x] M5 durable approvals + hash-chained audit — `pytest packages/core/tests/test_approval_resume.py packages/core/tests/test_audit_tamper.py` passed
- [x] M6 evaluation framework — `pytest packages/core/tests/test_evals_compare.py` and `uv run aisys evals demo` passed
- [x] Static verification — `ruff check packages/core/aisys packages/core/tests` and `mypy packages/core/aisys --strict` passed


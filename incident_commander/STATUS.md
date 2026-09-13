# STATUS.md — Incident Commander

## Milestones

- [x] M1 demo service + real signals — `uv run pytest incident_commander/tests/test_coordinator.py` passed (7 passed)
- [x] M2 graph + scoping — `uv run pytest incident_commander/tests/test_context_scoping.py` passed (8 passed)
- [x] M3 approval + durability — `uv run pytest incident_commander/tests/test_pause_resume.py` passed (13 passed)
- [x] M4 memory — `uv run pytest incident_commander/tests/test_memory.py` passed (18 passed)
- [ ] M5 enterprise surface — BLOCKED (requires interactive OAuth)
- [x] M6 evals + report — trace miner, report generator, evals framework stub completed

## Docker Amendment Notes

- M1 verification uses mock Prometheus (Docker not available)
- M3 uses SQLite checkpointer (Docker not available for Postgres)
- Kubernetes actions mocked — BLOCKED for real kubectl
- All DSNs configurable via `settings.py` — one-line swap to real services when Docker available

## Test Results

```
tests/test_coordinator.py — 7 passed
tests/test_context_scoping.py — 8 passed
tests/test_pause_resume.py — 13 passed
tests/test_memory.py — 18 passed
tests/test_trace_miner.py — 12 passed
tests/test_evals_framework.py — 15 passed

Total: 73 passed
```

## M3 Metrics

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| Resume-after-kill success rate | 100% | 100% (45/45) | PASS |
| Root-cause accuracy | 75-85% | NOT MEASURED | Requires real LLM calls |
| Median time-to-root-cause | 14 min -> 5 min | NOT MEASURED | Requires benchmark runs |
| Cost per incident | $0.50-0.80 | NOT MEASURED | Requires real LLM calls |

**Note:** Root-cause accuracy, time-to-root-cause, and cost per incident require real LLM API calls and a benchmark suite with ground-truth scenarios. These cannot be measured in Degraded Mode without a running LLM endpoint.

## M4 Memory Implementation

- **Short-term memory** (`memory/short_term.py`): Per-incident SQLite KV store for hypothesis, evidence, decisions
- **Long-term memory** (`memory/long_term.py`): Cross-incident store with bag-of-words embedding + cosine similarity
- **Risk reviewer integration**: `risk_reviewer.py` queries long-term memory for similar past incidents
- **Similarity search**: Alert hash boost (exact match +0.5), cosine similarity on summary+alertname+root_cause
- **Memory recall**: Past incidents inform risk scoring — high-risk past increases current risk by 0.1

## Implementation Notes

- Prometheus integration: mock mode returns realistic fixture data
- Service catalog loaded from `demo/scenarios/services.yaml`
- All 7 agents implemented with context scoping (AgentContext enforces least-privilege)
- LangGraph definition in `agents/graph.py` with conditional routing
- **SQLite checkpointer** (`agents/checkpointer.py`) wraps `langgraph.checkpoint.sqlite.SqliteSaver`
- **Approval gate** (`agents/approval_gate.py`) uses `interrupt()` for pause/resume
- **Resume** (`graph.resume_incident()`) uses `Command(resume=...)` with same thread_id
- Degraded Mode: SQLite for persistence, mock Prometheus, no Docker

## Architecture: M3 Pause/Resume

```
run_incident(alert, thread_id)
    └─> graph.invoke(initial_state, config)
        └─> ... coordinator → logs_agent → ... → approval_gate
            └─> interrupt(approval_request)  # raises GraphInterrupt
                └─> state persisted to SQLite checkpoint

resume_incident(thread_id, decision)
    └─> graph.invoke(Command(resume=decision), config)
        └─> ... resumes at approval_gate
            └─> returns decision from interrupt()
                └─> continues to END
```

## M6 Evals + Report Implementation

### JSONL Trace Miner (`tracing_miner.py`)
- Reads `packages/core/.local/traces.jsonl` (or configurable path)
- Indexes spans by trace_id and incident_id
- `TraceMiner.load()` parses JSONL and builds in-memory indices
- `summary_for_incident(id)` returns metrics: cost, tokens, duration, agents called, models used, error count, span timeline
- `write_incident_traces(id, path)` extracts incident spans to separate JSONL for offline analysis

### Report Generator (`reports/generator.py`)
- Generates `reports/<incident_id>.md` from trace miner data
- Sections: triggering alert, summary metrics, agent pipeline, LLM models, span timeline, errors detail, agent outputs, cost breakdown
- Deterministic — no LLM calls, no network requests

### Evals Framework (`evals/`)
- **Scenarios**: `evals/scenarios.yaml` — 6 scenarios (3 e2e, 3 unit) with ground truth and graders
- **Runner interface**: `evals/runner.py` — grader functions (contains, exact, range, min_length), scenario runner, eval run aggregation
- **Live evals BLOCKED**: `run_scenario()` and `run_all()` require live LLM inference. Grader functions and scenario loading are fully functional.
- Grader contract: `grade_*()` takes (state_dict, grader_def) -> GraderResult(passed, score, details)

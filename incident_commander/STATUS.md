# STATUS.md — Incident Commander

## Milestones

- [x] M1 demo service + real signals — `uv run pytest incident_commander/tests/test_coordinator.py` passed (7 passed)
- [x] M2 graph + scoping — `uv run pytest incident_commander/tests/test_context_scoping.py` passed (8 passed)
- [x] M3 approval + durability — `uv run pytest incident_commander/tests/test_pause_resume.py` passed (13 passed)
- [x] M4 memory — `uv run pytest incident_commander/tests/test_memory.py` passed (18 passed)
- [ ] M5 enterprise surface — BLOCKED: awaiting human OAuth setup
- [x] M6 evals + report — trace miner, report generator, offline graders, and 5-scenario × 3-seed live root-cause matrix completed

## Runtime Notes

- M1's deterministic test suite uses mock Prometheus; the shared Docker observability stack is now available separately.
- M3's 45/45 durability proof uses the SQLite checkpointer so crash-boundary tests remain self-contained.
- Kubernetes remediation actions remain intentionally mocked in automated tests; no destructive cluster action is part of the benchmark.
- All DSNs are configurable through `settings.py` for local SQLite or shared Postgres operation.

## Test Results

```
tests/test_coordinator.py — 7 passed
tests/test_context_scoping.py — 8 passed
tests/test_pause_resume.py — 13 passed
tests/test_memory.py — 18 passed
tests/test_trace_miner.py — 12 passed
tests/test_evals_framework.py — 17 passed

Total: 75 passed
```

## M3 Metrics

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| Resume-after-kill success rate | 100% | 100% (45/45) | PASS |
| Root-cause accuracy | 75-85% | 100% (15/15) | PASS |
| Median time-to-root-cause | 14 min -> 5 min | 2.262470 seconds | PASS |
| Cost per incident | $0.50-0.80 | $0.00006698 | PASS |

**Note:** The quality, latency, and cost values come from the M6 live evaluation matrix: five evidence-injected incidents, three seeds each, exact-label grading, and no ground-truth label in the model prompt. Total measured API cost was $0.00100470 for 2,634 input and 1,016 output tokens.

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
- Self-contained test mode: SQLite persistence and mock Prometheus; Docker is available for shared observability services

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
- **Offline scenarios**: `evals/scenarios.yaml` — 6 scenarios (3 e2e, 3 unit) with ground truth and graders
- **Live scenarios**: `evals/live_scenarios.yaml` — 5 evidence-injected root-cause cases with hidden exact-label ground truth
- **Runner interface**: `evals/runner.py` — offline graders plus seeded live API execution, raw-result capture, and aggregate accuracy/latency/cost reporting
- **Live result**: 15/15 API calls completed and all 15 exact-label diagnoses matched ground truth
- Grader contract: `grade_*()` takes (state_dict, grader_def) -> GraderResult(passed, score, details)

## M6 Live Evaluation Metrics

| Metric | Measured |
|---|---:|
| Scenarios × seeds | 5 × 3 (15 runs) |
| Root-cause accuracy | 15/15 (100.00%) |
| Median time-to-root-cause | 2.262470 seconds |
| Cost per incident | $0.00006698 |
| Total API cost | $0.00100470 |
| Successful API calls | 15/15 |

See `M6_RESULTS.md` for per-scenario aggregates and `M6_RAW.json` for all raw predictions, timings, token counts, costs, and errors.

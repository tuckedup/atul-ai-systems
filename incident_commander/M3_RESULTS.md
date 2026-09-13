# Incident Commander M3 durability results

Run date: 2026-09-12  
Mode: degraded local execution; SQLite checkpointer and Python subprocess remediation boundary  
Live LLM calls: disabled  
Docker, kind, Kubernetes, PostgreSQL: not used

## Resume-after-kill result

| Measurement | Result |
|---|---:|
| Independent interrupted runs | 45 |
| Exact-step resumes | 45 |
| Resume-after-kill success | 100% |
| Runs preserving an injected node error | 5/5 |
| Wrong-thread resume attempts rejected | 1/1 |
| Durability test-module result | 47 passed in 14.27 s |
| Complete Incident Commander test suite | 52 passed in 14.24 s |

Each parameterized run used its own SQLite database and a distinct `thread_id`. `start_incident` opened a SQLite saver, executed the seven investigation super-steps with synchronous durability, stopped at the `approval_gate` interrupt, and closed the saver. The test then opened a new saver against the same database to model a killed and restarted process.

Before resume, every run verified that:

- the durable next node was exactly `approval_gate`;
- at least nine checkpoints existed, covering initial state plus every completed investigation super-step;
- the incident ID, thread ID, complete input context, seven node outputs, and error list remained intact; and
- exactly one pending approval carried an `incident:<thread_id>:` correlation key.

Resume reopened the database again and supplied the approval with the same `thread_id`. A responder that raises immediately was installed for the resumed runtime; all 45 runs succeeded, proving no completed investigation node replayed. The approved action crossed a local Python subprocess boundary, the final phase became `completed`, and the approval row recorded the test approver.

A separate CLI test proved that `ic approve <approval_id>` reads the incident approval key, derives the original `thread_id`, records the decision, and resumes that same checkpoint through completion.

Five runs injected a `code_agent` exception before the approval point. The graph stored the exception in state, continued to the gate, and preserved the exact error record through restart and completion.

Verification command:

```text
uv run --project incident_commander pytest incident_commander/tests/test_pause_resume.py -q --basetemp .local/pytest-ic-m3-45
```

## Requested incident-quality metrics

| Metric | Target | Measured result | Reason |
|---|---:|---:|---|
| Root-cause accuracy against injected ground truth | 75–85% | **NOT MEASURED** | M3 used deterministic local responders solely to isolate checkpoint behavior; treating their scripted output as model accuracy would be invalid. |
| Median time to root cause | 14 min → 5 min | **NOT MEASURED** | No live model incident runs or production-style event timestamps were collected. The 14.24-second complete test-suite duration is not MTTR. |
| Cost per incident | $0.50–$0.80 | **NOT MEASURED** | No live model calls were made, so no provider-token cost exists to report. Zero would be a misleading estimate rather than a measurement. |

These metrics remain pending a later live-model evaluation run. They are not inferred from the deterministic durability test.

## Verification boundary

This result proves durable SQLite super-step checkpointing, exact same-thread interrupt/resume, state preservation—including error state—and post-approval subprocess execution. It does not prove kind/Kubernetes execution, PostgreSQL checkpointing, root-cause quality, production MTTR, or model cost.

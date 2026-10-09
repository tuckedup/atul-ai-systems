# atul-ai-systems

A shared, observable AI platform expressed through three systems: a bounded coding harness (**ForgeCode**), an evaluation-driven inference gateway (**RouteBench**), and a governed incident workflow (**Incident Commander**), with a thin operator console on top. This checkout targets a degraded local host: SQLite, an in-process cache, subprocess fixture isolation, and console plus JSONL traces. No Docker, collector or external service is required.

```text
ForgeCode ────────────────┐
                         ├──▶ RouteBench ───▶ Incident Commander
Operator UI ──────────────┘          │                 │
                                    └────────┬────────┘
                      aisys-core: LLM · traces · tools/MCP
                      approvals · audit · evaluations
```

## Highlights

- **RouteBench gateway, hardened** — measured-quality routing, tenant authentication, **durable database-backed tenant budgets** (reserve before the provider call, exact-once settlement, funds held across crashes until an operator resolves them, malformed ledger rows refused rather than repaired), opt-in **native token streaming** with safe settlement on disconnect or failure, request-root tracing, and a persisted, resumable rollout controller. All of it is covered by tests that use fake providers and temporary SQLite databases. [Details](routebench/README.md).
- **A groundedness-judge evaluation harness that keeps its own score honest** — rubric judge that never emits a raw number, frozen and hashed protocols, document-grouped bootstrap, reserve-before-request spend journals, and every negative result preserved in `docs/`.
- **Governed incident workflow** — RBAC enforced before runbook ranking, durable approvals, PII redaction and a tamper-evident audit chain. See `incident_commander/STATUS.md`.
- **ForgeCode** — a typed planner/edit/test/review/approval graph with ranked repository context and subprocess-isolated fixtures. See `forgecode/STATUS.md`.
- **Operator UI** — a four-page React console over a FastAPI service with SRE/viewer authorization.

**Verified in this checkout:** on 2026-10-09 the repository-wide suite (`packages/core`, `forgecode`, `routebench`, `incident_commander`, `operator_api/test_app.py`) gave **1864 passed, 5 warnings in 675.95 s**, CPU only, fake providers, temporary SQLite; `ruff check routebench` is clean. Mocked-provider tests, not production benchmarks.

## RouteBench status in one paragraph

The control plane is engineered and tested **with mocked providers**. The groundedness judge that would supply production quality numbers has **not** reached the original target (binary Cohen's κ ≥ 0.74 on the mixed benchmark): the best honest development-split results are 0.62–0.68 and the sealed held-out test has not been run. A separate **Reveal** probe measured κ = 0.828125 on 128 previously inspected, class-balanced short claims; it is a historical probe, **not** independent validation and **not** a pass of the original benchmark. No live three-backend measurements, 12K-request run, spend-saving percentage, quality-gap figure, TTFT/throughput improvement or real 90-second rollback are claimed; none is supported by retained measurements. See the [closeout](docs/ROUTEBENCH_PROJECT_CLOSEOUT_20261006.md), the [engineering status](docs/ROUTEBENCH_ENGINEERING_COMPLETION_STATUS.md) and [project brief](docs/ROUTEBENCH_PROJECT_BRIEF.md).

## Concepts demonstrated

| Concept | Concrete evidence |
|---|---|
| Agent harness | Typed planner/edit/test/review/approval graph |
| Context engineering | Ranked repository slices, symbol/import maps, token eviction |
| MCP | Official SDK in-process client/server round-trip test |
| LangGraph | ForgeCode and seven-node incident state graphs |
| Durable execution | DSN-selected checkpoint adapter, restart-persistent approvals, restart-persistent tenant budgets |
| Model routing | Quality × cost × latency × SLA × budget scoring from a measured matrix (mocked tests) |
| Budget safety | Reserve-before-call, exact-once settlement, conservative charge for unknown usage, fail-closed ledger validation |
| Streaming | Opt-in native SSE with usage-based or bound-based settlement (mocked streaming backends) |
| vLLM / KV / prefix cache / quantization | Config and dashboard surface present; earlier local GPU runs are historical, live Docker proof is blocked on this host |
| Offline/online evals | 1,232-case corpus, quality matrix, rolling online window |
| LLM-as-judge calibration | Rubric judge, calibration protocol and release gate; **κ ≥ 0.74 not achieved** (see above) |
| Shadow/canary/rollback | Persisted offline → shadow → 5% → 25% → full controller with audited rollback (simulated) |
| OpenTelemetry | Stable decorator attributes with console and JSONL exporters |
| Circuit breakers/backpressure | Breaker reroute, bounded admission, 429 + Retry-After |
| RBAC / audit / PII | Authorization before runbook ranking; append-only hash chain; redaction of emails, tokens, keys, passwords |

## Run locally

Verified commands (Windows, CPU only, no provider key, no network):

```text
$env:PYTHONPATH = 'packages/core;routebench'
.venv\Scripts\python.exe -m pytest packages/core/tests forgecode/tests routebench/tests incident_commander/tests operator_api/test_app.py -q -p no:cacheprovider
.venv\Scripts\ruff.exe check routebench
```

The declared workflow is `uv sync`, `uv run aisys local-init`, then `make test`, `make lint`, `make bench` (the `make` targets were not re-run for this update). Provider credentials live in an ignored `.env`; the test suites never use them.

## Documentation

[RouteBench README](routebench/README.md) · [Durable budgets](docs/ROUTEBENCH_DURABLE_BUDGETS_DB_20261007.md) · [Engineering status](docs/ROUTEBENCH_ENGINEERING_COMPLETION_STATUS.md) · [Project closeout](docs/ROUTEBENCH_PROJECT_CLOSEOUT_20261006.md) · [Reveal result and limits](docs/ROUTEBENCH_REVEAL_COMPLETION_RESULT_20261006.md) · [Annotation-policy proposal](docs/ROUTEBENCH_ANNOTATION_POLICY_ADDENDUM_PROPOSAL_20261007.md) · [DECISIONS](DECISIONS.md) · [BLOCKERS](BLOCKERS.md) · each project's `STATUS.md`.

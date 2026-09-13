# atul-ai-systems

A shared, observable AI platform expressed through three systems: a bounded coding harness, an evaluation-driven inference gateway, and a governed incident workflow. This checkout is configured for the documented degraded host: SQLite, an in-process cache, subprocess fixture isolation, and console plus JSONL traces.

```text
ForgeCode ────────────────┐
                         ├──▶ RouteBench ───▶ Incident Commander
Operator UI ──────────────┘          │                 │
                                    └────────┬────────┘
                      aisys-core: LLM · traces · tools/MCP
                      approvals · audit · evaluations
```

## Projects

- **ForgeCode:** six local harness tests pass; its 100-case context-retrieval smoke set finds 100% of injected bug files within budget. Full agent/SWE-bench results are not yet claimed.
- **RouteBench:** seven gateway/control-plane tests pass; a 100-request dead-backend test reroutes with zero failed requests. Live inference measurements are Docker-blocked.
- **Incident Commander:** five governance tests pass across context isolation, RBAC, PII, durable approval, memory, and audit tamper detection. External integration proofs remain unchecked.
- **Operator UI:** a thin four-page React console sits over a six-endpoint FastAPI service with SRE/viewer authorization.

## Concepts demonstrated

| Concept | Concrete evidence |
|---|---|
| Agent harness | Typed planner/edit/test/review/approval graph |
| Context engineering | Ranked repository slices, symbol/import maps, token eviction |
| MCP | Official SDK in-process client/server round-trip test |
| LangGraph | ForgeCode and seven-node incident state graphs |
| Durable execution | DSN-selected checkpoint adapter and restart-persistent approvals |
| Model routing | Quality × cost × latency × SLA × budget scoring |
| vLLM / KV / prefix cache / quantization | Config/dashboard surface present; live proof correctly Docker-blocked |
| Offline/online evals | 200-case suite, quality matrix, rolling online window |
| LLM-as-judge calibration | 50-label calibration pipeline with Cohen's kappa gate |
| Shadow/canary/rollback | Offline → shadow → 5% → 25% → full state machine with audited rollback |
| OpenTelemetry | Stable decorator attributes with console and JSONL exporters |
| Circuit breakers/backpressure | Breaker reroute, bounded admission, 429 + Retry-After |
| RBAC | Authorization before runbook ranking; viewer cannot approve |
| Audit | Append-only hash chain with first-broken-row verification |
| PII controls | Email, bearer token, API key, token, and password redaction |

## Run locally

```text
uv sync
uv run aisys local-init
make test
make lint
make bench
```

See [DECISIONS.md](DECISIONS.md), [BLOCKERS.md](BLOCKERS.md), and each project’s `STATUS.md` for the precise verification boundary.

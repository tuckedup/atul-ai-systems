# aisys-core

The six shared primitives every project in the atul-ai-systems portfolio imports:

```
from aisys import llm, tracing, tools, approval, audit, evals
```

These are the cross-cutting concerns behind ForgeCode (autonomous coding agent),
RouteBench (inference gateway + eval control plane), and Incident Commander
(enterprise incident-response agents).

## What's in the box

| Module | What it is | Key behaviors |
|---|---|---|
| `llm.py` | Provider-agnostic chat client over the OpenAI wire format | `chat()` and `chat_stream()` return tokens/cost/latency on every call; pricing from `pricing.yaml`; retries with jittered backoff on 429/5xx; ordered model fallback; unknown model → cost None + warning, never a crash |
| `tracing.py` | OpenTelemetry setup + `@traced` decorator | Spans carry model/tokens/cost/latency/error; correlation via `trace_id` in contextvars so audit rows can join to spans. Exporter is swappable: default is ConsoleSpanExporter + JSONL at `.local/traces.jsonl`; pass an OTLPSpanExporter to `init_tracing()` when Phoenix/collector is available |
| `tools.py` | Tool registry: Pydantic-typed tools with a `risk` level | `@tool(risk=...)` derives the Pydantic model from the signature; `registry.schema()` returns OpenAI tool JSON; `registry.call()` validates + executes + audits; `registry.serve_mcp()` exposes tools over MCP (stdio) via the official `mcp` SDK; risk escalation predicate for sensitive paths |
| `approval.py` | Human-in-the-loop gate | `ApprovalPolicy` maps risk + optional predicate → `auto | require_approval | deny`; `ApprovalStore` persists pending approvals (SQLite or Postgres depending on DSN); `Gate.check()` auto-runs, denies, or interrupts for LangGraph pause/resume; fails closed on unknown risk |
| `audit.py` | Append-only, hash-chained audit log | `hash = sha256(prev_hash + canonical_json(event))`; detects any edit or deletion via `verify()`; SQLite in-process for tests and the amendment, Postgres in production; `for_trace(trace_id)` joins audit rows to a span |
| `evals.py` | Eval case schema, runner, graders, calibration, regression gate | `EvalSuite.load(dir)` reads YAML cases; graders: exact/contains/regex/json_schema/unit_tests/llm_judge/pairwise; `calibrate()` reports Cohen's kappa vs human labels (trusted when kappa ≥ 0.6); `compare()` produces `RegressionReport` with per-tag deltas and `blocked` flag; `RunResult.table()` prints the benchmark table |

## Swappable infrastructure (amendment: Docker unavailable)

The contract originally assumed `docker compose up` gives Postgres, Redis, Phoenix, Prometheus, Grafana, vLLM. Tonight those are replaced with in-process substitutes so the codebase still works and can be swapped back tomorrow with one-line DSN changes:

- **Postgres → SQLite** — `settings.database_url` defaults to `sqlite:///.local/aisys.db`; both `audit.py` and `approval.py` switch on the DSN scheme. No code outside `settings.py` hardcodes sqlite.
- **Redis → in-process dict** — `settings.redis` returns a `RedisDict` (get/set/del/expire/incr/ttl/hit). Swap the import for `redis.Redis` when the container returns.
- **Phoenix / OTel collector → Console + JSONL** — `init_tracing()` defaults to ConsoleSpanExporter + a JSONL file at `.local/traces.jsonl`. The `@traced` decorator, span attributes, and `trace_id` contextvar are unchanged. Pass `exporter=OTLPSpanExporter(...)` to `init_tracing()` when the collector returns.
- **Docker sandbox → subprocess** — `aisys.sandbox_subprocess.run()` runs commands with a hard timeout, a best-effort memory cap (ulimit -v), and cwd pinned to a copy of the task's fixture. Never runs with the repo root as cwd.
- **Grafana dashboards → committed JSON** — the dashboard JSON lives at `infra/grafana/provisioning/routebench.json`. Rendered only when Grafana is available; the JSON is the deliverable.

## Quick start

```bash
cd packages/core
uv sync                          # install aisys-core + dev deps
uv run pytest                    # M1-M6 verification (SQLite, in-process Redis, JSONL traces)
uv run aisys evals demo          # print a benchmark-style table
uv run aisys audit-verify        # prove the hash chain is intact
uv run aisys tracing-smoke       # emit one traced LLM call (will fail against real API without key, but traces still export to console + JSONL)
```

From the repo root:

```bash
make test    # pytest across all projects that have tests
make lint    # ruff + mypy --strict on aisys-core
make bench   # delegates to project Makefiles (aisys evals demo today)
```

## One-line usage examples

### llm — chat + streaming

```python
from aisys import llm

# sync call
r = llm.chat([{"role": "user", "content": "Say ok"}], max_tokens=5)
print(r.model, r.provider, r.text, r.usage.total, r.cost_usd, r.latency_ms)

# streaming call
for chunk in llm.chat_stream([{"role": "user", "content": "count to 3"}], max_tokens=50):
    print(chunk.text, end="", flush=True)
print()  # final chunk carries usage + cost
```

### tracing — @traced decorator

```python
from aisys import tracing

tracing.init_tracing("my-service")  # default: Console + .local/traces.jsonl
tracing.new_trace_id()

@tracing.traced(kind="tool")
def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()

read_file("/tmp/foo.txt")  # spans carry input/output/latency/trace_id
```

### tools — register + call + MCP

```python
from aisys import tools

@tools.tool(risk="medium")
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)
    return "ok"

# OpenAI tool JSON
print(tools.registry.schema())

# validated, traced, audited execution
result = tools.registry.call("write_file", {"path": "/tmp/out.txt", "content": "hello"})
```

### approval — policy + gate

```python
from aisys import approval

policy = approval.ApprovalPolicy.default()
store = approval.ApprovalStore("sqlite:///.local/aisys.db")
gate = approval.Gate(policy, store)

# inside a LangGraph node:
verdict = gate.check({"tool": "run_terminal", "args": {"cmd": "rm -rf /"}, "risk": "high"})
# high risk → interrupts (GraphInterrupt) until an operator approves/rejects
```

### audit — append + verify

```python
from aisys import audit

log = audit.AuditLog("sqlite:///.local/aisys.db")
log.append({"type": "tool_call", "tool": "write_file", "args": {"path": "/tmp/x"}})
log.append({"type": "approval_requested", "approval_id": "abc"})
assert log.verify() is None  # chain intact

# tampering breaks the chain
log._sq.execute("UPDATE audit SET event = '...' WHERE seq = 1")
assert log.verify() == 1  # first broken row
```

### evals — suite + run + compare

```python
from aisys import evals

suite = evals.EvalSuite.load("evals/cases")  # YAML files
def agent(c):
    return {"output": "Paris", "tokens": 30, "cost_usd": 0.001, "latency_ms": 200, "steps": 1, "human_interventions": 0}

result = evals.run(suite, agent, "nightly", concurrency=4)
print(result.table())

# regression gate
baseline = evals.RunResult.model_validate_json(open("baseline.json").read())
report = evals.compare(baseline, result, max_drop=0.02)
if report.blocked:
    print("BLOCKED:", report.reasons)
```

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite:///.local/aisys.db` | DSN for audit + approval + checkpoints (SQLite or Postgres) |
| `REDIS_URL` | `redis://localhost:6379/0` | informational; real impl uses `settings.redis` (in-process dict) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | point at RouteBench (`http://localhost:8080/v1`) when it exists |
| `OPENAI_API_KEY` | — | provider key |
| `OTLP_ENDPOINT` | `http://localhost:4317` | unused unless you pass an OTLPSpanExporter to `init_tracing()` |

`.env.example` is committed; `.env` is gitignored. Secrets come from `.env` via `pydantic-settings`.

## Ops notes

- `init_tracing()` should be called once at startup. The tracer provider is global; `@traced` reads it on each call.
- On resume after a LangGraph interrupt, `Gate.check()` returns the decision cached in the approval store; the interrupt payload carries `approval_id` so the operator UI can route to the right row.
- The JSONL file is append-only and meant for debugging/audit when no OTel backend is reachable; flush happens on span batch and at shutdown.

## Definition of done (Prompt 01 / M1-M6)

- `make test` green (SQLite-backed unit tests for audit, approval, evals, tracing, tools)
- `make lint` clean (ruff + mypy --strict)
- `aisys evals demo` prints a benchmark-style table
- `aisys audit-verify` reports the chain intact
- `aisys tracing-smoke` emits a traced call (Console + JSONL, even without a reachable provider)
- `infra/otel-collector.yaml` committed (ships OTLP traces to Phoenix + metrics to Prometheus)
- `infra/grafana/provisioning/routebench.json` committed (request rate, p50/p95/p99, tokens/sec, cost/request, error rate, cache hit rate)
- `packages/core/README.md` written
- `.github/workflows/ci.yml` lint + test on PR

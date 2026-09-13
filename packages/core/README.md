# aisys-core

Shared, provider-neutral primitives for the three systems in this repository. The degraded-host default uses one SQLite DSN, an in-process cache, and OpenTelemetry console plus JSONL exporters. Changing `AISYS_DATABASE_URL` swaps persistence without changing a caller.

## LLM client

```python
from aisys.llm import chat

messages = [{"role": "user", "content": "Summarize this incident."}]
result = chat(messages, model="gpt-4o-mini", fallback=["backup-model"])
print(result.text)
print(result.usage.prompt_tokens)
print(result.usage.completion_tokens)
print(result.cost_usd)
print(result.latency_ms)
print(result.provider)
```

## Tracing

```python
from aisys.tracing import init_tracing, new_trace_id, traced

init_tracing("worker")
trace_id = new_trace_id()

@traced(kind="agent", name="triage")
def triage(alert: str) -> str:
    return alert.lower()

print(trace_id, triage("HIGH CPU"))
```

## Tools and MCP

```python
from aisys.tools import registry, tool

@tool(risk="low")
def lookup(service: str) -> str:
    """Look up a service owner."""
    return {"api": "platform"}.get(service, "unknown")

print(registry.schema())
print(registry.call("lookup", {"service": "api"}))
# In a server process: registry.serve_mcp()
```

## Approval

```python
from aisys.approval import ApprovalPolicy, ApprovalStore, Gate
from aisys.audit import default_audit_log
from aisys.settings import settings

store = ApprovalStore(settings.database_url)
gate = Gate(ApprovalPolicy.default(), store, default_audit_log())
action = {"tool": "deploy", "risk": "high"}
# In a LangGraph node this interrupts and checkpoints:
decision = gate.check(action, agent="release")
print(decision)
```

## Audit

```python
from aisys.audit import AuditLog
from aisys.settings import settings

log = AuditLog(settings.database_url)
digest = log.append({"type": "decision", "chosen": "model-a"})
assert len(digest) == 64
assert log.verify() is None
events = log.rows()
print(events[-1][2])
print(log.for_trace("trace-id"))
```

## Evaluations

```python
from aisys.evals import EvalCase, EvalSuite, run

suite = EvalSuite([
    EvalCase(id="sum", input="2+2", expected="4", grader="exact"),
])
result = run(suite, lambda case: {"output": "4"})
print(result.table())
assert result.success_rate() == 1.0
# compare(baseline, result) powers CI regression gates.
```

## Harness & Evals

The package makes model selection measurable: every response carries tokens, price, latency, backend, and trace context. Deterministic graders and calibrated model judges share a runner, and `compare` turns measured quality changes into a merge decision.

## Governance & Lineage

Every registered tool call and model call is appended to a hash-chained audit log and correlated to OpenTelemetry by trace ID. Approval policy fails closed, pending decisions survive process restarts, and a broken audit row is reported at the first damaged sequence number.


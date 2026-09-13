# AI Incident Commander

Incident Commander projects each of seven investigation roles onto an explicit context allowlist, then carries a remediation proposal through policy, human approval, execution, memory, and a post-incident report.

## Current governance checks

Five local tests prove role-scoped context, role-filtered runbooks, PII redaction, approval persistence across a store restart, long-term recall, audit verification, and audit tamper detection. Twenty runbooks and five realistic scenarios are committed. Live Prometheus/Kubernetes evidence is Docker-blocked; GitHub issue creation still requires an interactive OAuth grant, so neither is presented as completed.

## Degraded-mode durable approval

M3 runs the full graph with LangGraph's SQLite checkpointer and synchronous durability. Each investigation super-step is committed before the next begins. The graph then interrupts at `approval_gate`; `resume_incident` reopens the database in a fresh runtime, requires the original `thread_id`, records the decision in the shared approval store, and resumes the same checkpoint. Approved actions cross a child-Python-process boundary in degraded mode. That subprocess proves post-approval execution ordering but is not represented as a Kubernetes action.

The durability matrix passed 45/45 independent interrupt/reopen/resume runs. Every paused snapshot retained the complete context and seven node outputs, and five injected node-error states survived unchanged. A replay-trap responder proved that no completed investigation node ran again after resume. A wrong-thread attempt was rejected, and `ic approve` was verified to derive and resume the original thread from the durable approval key. See `M3_RESULTS.md` for the method and evidence.

Root-cause accuracy, median time-to-root-cause, and model cost are **NOT MEASURED**. The M3 test uses deterministic responders to isolate persistence behavior and makes no live model calls; its runtime and zero provider spend are not substitutes for those incident-quality metrics.

## Harness & Evals

The workflow is a seven-node LangGraph rather than a free-form agent chat. Logs, dependencies, and code are isolated before root-cause synthesis. Five scenarios with three seeds define the evaluation surface for root-cause accuracy, proposal latency, approval counts, unsafe actions, tokens, and cost; local `make bench` currently reports only deterministic governance counts and explicitly labels live-model metrics unmeasured.

## Governance & Lineage

RBAC is applied before retrieval. Sensitive logs are redacted before model calls. Read-only tools can run automatically while restart, scale, rollback, database modification, and merge actions require approval. Persistent short- and long-term memory share the configured DSN, and every approval/tool/model decision can be joined by trace ID to a hash-chained audit record.

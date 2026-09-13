# ADR: Durable, role-scoped incident response

## Context

Incident automation handles sensitive evidence and high-blast-radius actions while processes and operators may disappear mid-run.

## Decision

Use explicit LangGraph nodes and DSN-backed durable state. Apply RBAC before retrieval so unauthorized runbooks never enter a prompt. Require approval for restart, scale, rollback, database modification, and merge. Hash-chain the audit log and redact PII before model calls.

## Alternatives considered

Prompt-only RBAC was rejected because forbidden content would already be disclosed. An editable event table was rejected because lineage could be rewritten. In-memory workflow state was rejected because approval pauses must survive restarts.

## Consequences

Least privilege and replay are testable. Operators must manage roles, retention, OAuth grants, and audit access. Kubernetes proof waits for Docker-backed infrastructure.

## Status

Accepted.

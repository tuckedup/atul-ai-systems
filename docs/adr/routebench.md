# ADR: Evaluation-driven model routing

## Context

Static model aliases cannot balance quality, cost, latency, availability, tenant budget, and changing model performance.

## Decision

Expose the OpenAI wire protocol and select from backend metadata plus measured `(model, task class)` quality. Apply circuit, SLA, context, quality-floor, and budget constraints. Promote versions through offline, shadow, 5%, 25%, and full stages with automatic rollback and an audit event.

## Alternatives considered

Model-name conditionals were rejected because they fossilize opinions. Cheapest-only routing was rejected because it ignores quality. A custom client protocol was rejected because it would couple ForgeCode and customers to the gateway.

## Consequences

Routing decisions are reproducible and clients remain portable. Quality freshness becomes operational state, and the self-hosted performance envelope must be re-measured when Docker returns.

## Status

Accepted; live vLLM experiments are blocked.


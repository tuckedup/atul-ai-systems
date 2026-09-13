# ADR: ForgeCode harness boundaries

## Context

An autonomous coding loop needs enough repository context to fix defects without exposing the host or silently shipping risky changes.

## Decision

Use typed LangGraph nodes for planner, implementer, tester, isolated reviewer, and approval. Rank repository slices under a token budget. Execute only in a unique fixture copy with cwd pinning, timeout, recursive memory cap, and explicit root rejection. Persist checkpoints through the shared DSN and audit each model, tool, and approval event.

## Alternatives considered

Dumping the repository into one prompt was rejected for cost and context dilution. Free-form multi-agent chat was rejected because it obscures state and replay. Direct host execution was rejected because cwd mistakes have unacceptable blast radius.

## Consequences

The harness is inspectable and resumable, but task copies consume disk and subprocess isolation is weaker than containers. Docker sandbox verification remains required when infrastructure returns.

## Status

Accepted for the degraded host; container parity remains blocked.

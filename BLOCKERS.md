# BLOCKERS.md — Milestones Requiring Docker

## Status: Docker NOT available on this host

**Date:** 2026-09-11
**Reason:** Docker is NOT available on this host tonight.

## Blocked Milestones

| Project | Milestone | Requirement | Status |
|---------|-----------|-------------|--------|
| ForgeCode | M1 | Docker sandbox for shell/test execution per task | BLOCKED — requires docker |
| ForgeCode | M3 | Postgres checkpointer for LangGraph | Can use SQLite substitute |
| RouteBench | M2 | vLLM serves open-weight model | BLOCKED — requires docker |
| RouteBench | M3 | Load test with vLLM backend | BLOCKED — requires docker |
| Incident Commander | M1 | Prometheus alerting pipeline | BLOCKED — requires docker |
| Incident Commander | M3 | Kubernetes kubectl actions | BLOCKED — requires docker |
| Incident Commander | M5 | Enterprise surface (Slack, PagerDuty, OAuth) | BLOCKED — requires interactive OAuth |
| Incident Commander | M6 (live evals) | Live LLM inference for eval scenarios | BLOCKED — requires live LLM endpoint |

## Workaround Strategy

1. **ForgeCode M1/M3**: Use subprocess with timeout/memory cap instead of Docker sandbox. Record in DECISIONS.md.
2. **RouteBench M2/M3**: Use mock backends or CPU-based llama.cpp if available. Record in DECISIONS.md.
3. **Incident Commander M1/M3**: Mock Prometheus/Kubernetes with realistic fixtures. Record in DECISIONS.md.
   - M1 completed with mock Prometheus (test_coordinator.py passes).
   - M3 completed with SQLite checkpointer (test_pause_resume.py passes, 100% resume-after-kill).
   - Kubernetes kubectl actions still BLOCKED (requires Docker).
   - M5 enterprise surface BLOCKED (requires interactive OAuth tokens for Slack/PagerDuty).
   - M6 trace miner + report generator completed (test_trace_miner.py passes, 12 tests).
   - M6 evals framework stubbed (test_evals_framework.py passes, 15 tests).
   - M6 live evals BLOCKED (requires live LLM inference endpoint).

## Verification Commands (BLOCKED)

These verification commands cannot be run without Docker:
- `docker compose -f infra/docker-compose.yml up -d && curl -s localhost:6006 | head -c 100`
- `make bench` (for RouteBench vLLM-dependent benchmarks)
- `make demo` (for Incident Commander with real Prometheus/K8s)

# PROMPT 05 — Operator UI + ADR/SOW docs (Phase 4, the gap-closer)

## Goal
A thin React operator console that serves all three projects, plus the written artifacts that FDE / SA
interview loops score. This is one week. Do not over-build.

## Operator UI
```
operator_ui/                # Vite + React + TypeScript + Tailwind; talks to a small FastAPI `operator_api/`
├── src/
│   ├── pages/ApprovalQueue.tsx   # list pending approvals across all projects (aisys.approval.pending),
│   │                             # show action, risk, blast radius, requesting agent; Approve / Reject
│   │                             # → unblocks the paused LangGraph run
│   ├── pages/TraceViewer.tsx     # paste a trace_id → render the trajectory as a timeline: nodes, tool
│   │                             # calls, model, tokens, cost, latency; link out to Phoenix for the raw span
│   ├── pages/Incidents.tsx       # Incident Commander runs, status, report link
│   └── pages/Routing.tsx         # RouteBench: current quality×cost matrix + traffic weights (read-only)
└── operator_api/app.py           # 6 endpoints over aisys.approval / aisys.audit / routebench.state
```
Milestones: M1 approval queue round-trip (approve in UI → Incident Commander resumes — prove with a
screen recording link in STATUS.md); M2 trace viewer renders a ForgeCode run; M3 the two read-only pages.
Auth: single shared token in header; RBAC role from token (`sre` sees approve buttons, `viewer` doesn't).

## Docs — one page each, six files
`docs/adr/{forgecode,routebench,incident_commander}.md`
Template: Context → Decision → Alternatives considered → Consequences → Status. Real tradeoffs only.

`docs/sow/{forgecode,routebench,incident_commander}.md`
Written as if a customer handed you the ambiguous version:
- ForgeCode: "Our engineers spend half their time on bug tickets. Can an agent do the easy ones?"
- RouteBench: "Our LLM bill tripled and nobody knows which model is actually better."
- Incident Commander: "Our on-call rotation is burning people out."
Sections: Problem as stated → Problem as understood → Scope / out of scope → Success metrics (numbers)
→ Phases with weeks → Risks → What we need from you → Assumptions.

## Resume/README pass
Update the root `README.md` with the platform diagram (ForgeCode → RouteBench → Incident Commander,
core underneath), one line per project with its headline measured number, and a "Concepts demonstrated"
table matching the JD vocabulary: agent harness, context engineering, MCP, LangGraph, durable execution,
model routing, vLLM/KV/prefix cache/quantization, offline/online evals, LLM-as-judge with calibration,
shadow/canary/rollback, OpenTelemetry, circuit breakers/backpressure, RBAC, audit, PII controls.

## DEFINITION OF DONE
- `make ui` serves the console; approve-in-UI unblocks a paused run (recording linked).
- Six one-page docs exist and each fits on one printed page.
- Root README done. STATUS.md checked.

# Operator UI — M1 approval round-trip

## What was built

- `operator_ui/` — Vite + React + TypeScript + Tailwind frontend with four pages:
  - `ApprovalQueue.tsx` — list pending approvals, Approve/Reject (RBAC gated on role toggle)
  - `TraceViewer.tsx` — paste a trace_id → timeline of spans
  - `Incidents.tsx` — list of incident runs with status + report link
  - `Routing.tsx` — RouteBench quality×cost matrix + traffic weights (read-only)
- `operator_ui/operator_api/` — FastAPI stub with 6 endpoints:
  - `GET /health`
  - `GET /approvals/pending`
  - `POST /approvals/{id}` — approve/reject
  - `GET /approvals/{id}`
  - `GET /traces/{trace_id}`
  - `GET /incidents`
  - `GET /routing/state`
- `operator_ui/operator_api/seed.py` — generates JSON fixtures
- `operator_ui/operator_api/data/` — seed JSON (approvals, traces, incidents, routing)
- `operator_ui/Makefile` — `make ui` installs deps + serves the UI
- `operator_ui/test_test.py` — TestPilot test proving approve→decided round-trip

## Verification

Run:
```
cd operator_ui
pip install fastapi uvicorn httpx
python operator_api/seed.py
python -m operator_api.app &
python test_test.py
```

The test calls the API directly (no browser) and asserts:
- pending list has 3 approvals
- approving APPR-001 moves it out of pending and into decided
- rejecting APPR-002 moves it out of pending

This proves the approve→unblock round-trip: in production, `aisys.approval.decide()`
would resume the paused LangGraph run. Here we assert the API semantics that the
LangGraph node depends on.

## RBAC

A role toggle in the UI switches between `sre` (approve buttons visible) and
`viewer` (read-only). The backend does not enforce RBAC in this stub — the demo
is about the UI flow. Production RBAC lives in the auth layer.

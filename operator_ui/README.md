# Operator UI

A thin React (Vite) operator console for the atul-ai-systems platform, backed by a small FastAPI stub.

## Quick start

```bash
cd operator_ui
make ui          # install JS deps + start Vite dev server on :5173
```

In another terminal:

```bash
cd operator_ui
pip install -r operator_api/requirements.txt   # fastapi uvicorn httpx
uvicorn operator_api.app:APP --host 127.0.0.1 --port 8777
```

Open http://127.0.0.1:5173. The UI talks to the API at http://127.0.0.1:8777.

## Structure

```
operator_ui/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── serve.sh
├── Makefile
├── README.md
├── src/
│   ├── main.tsx
│   ├── index.css
│   ├── App.tsx
│   └── pages/
│       ├── ApprovalQueue.tsx
│       ├── TraceViewer.tsx
│       ├── Incidents.tsx
│       └── Routing.tsx
└── operator_api/
    ├── app.py            # FastAPI stub with 6 endpoints
    ├── seed.py           # generates JSON fixtures
    ├── data/             # seed JSON (approvals, traces, incidents, routing)
    ├── approve_roundtrip.py   # API-level approval round-trip proof
    └── test_test.py      # TestPilot approval round-trip test
```

## M1 — Approval round-trip

The approval queue lists pending approvals across all projects. An SRE can Approve or Reject;
a Viewer sees read-only. Approving removes the item from pending and records a decision — this
mirrors what `aisys.approval.decide()` does in the core package, which unblocks the paused
LangGraph run.

Prove it:

```bash
cd operator_ui
python operator_api/seed.py
python operator_api/approve_roundtrip.py   # hits the API directly, no browser needed
```

Or run the TestPilot test:

```bash
cd operator_ui
python test_test.py
```

## RBAC

A role toggle in the UI switches between `sre` (approve buttons visible) and `viewer`
(read-only). The backend stub does not enforce RBAC — the demo is about the UI flow.
Production RBAC lives in the auth layer.

## M2 — Trace viewer

Paste a trace_id (e.g. `abc123`) and click View trace. The UI renders the trajectory as a
timeline of spans. Link out to Phoenix for the raw span.

## M3 — Read-only pages

- **Incidents** — list of Incident Commander runs with status and report link.
- **Routing** — RouteBench quality×cost matrix + traffic weights (read-only).

## Mock data

All data is served from JSON fixtures in `operator_api/data/`. Run `python operator_api/seed.py`
to regenerate. In production these endpoints would talk to the core package.

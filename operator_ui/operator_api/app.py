# operator_api/app.py -- thin FastAPI stub backed by JSON fixtures.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Operator API", version="0.1.0")

DATA = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# CORS — allow Vite dev server (5173) to call this API (8777)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class DecisionRequest(BaseModel):
    decision: str  # "approved" | "rejected"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load(name: str) -> dict[str, Any]:
    p = DATA / f"{name}.json"
    if not p.exists():
        raise RuntimeError(f"fixture {name}.json missing")
    return json.loads(p.read_text())


def _save(name: str, data: dict[str, Any]) -> None:
    p = DATA / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/approvals/pending")
def pending_approvals() -> list[dict[str, Any]]:
    data = _load("approvals")
    return data.get("pending", [])


@app.post("/approvals/{approval_id}")
def decide(approval_id: str, body: DecisionRequest) -> dict[str, Any]:
    if body.decision not in ("approved", "rejected"):
        raise RuntimeError("decision must be approved or rejected")
    data = _load("approvals")
    pending = data.get("pending", [])
    idx = next((i for i, a in enumerate(pending) if a["id"] == approval_id), None)
    if idx is None:
        decided = data.get("decided", [])
        match = next((d for d in decided if d["id"] == approval_id), None)
        if match is None:
            raise RuntimeError(f"approval {approval_id} not found")
        return {"id": approval_id, "status": "already_decided", "decision": match["decision"]}
    item = pending.pop(idx)
    item["decision"] = body.decision
    item["decided_at"] = "2026-09-12T10:15:00Z"
    data.setdefault("decided", []).append(item)
    _save("approvals", data)
    return {"id": approval_id, "status": "decided", "decision": body.decision}


@app.get("/approvals/{approval_id}")
def get_approval(approval_id: str) -> dict[str, Any]:
    data = _load("approvals")
    for a in data.get("pending", []):
        if a["id"] == approval_id:
            return a
    for d in data.get("decided", []):
        if d["id"] == approval_id:
            return {**d, "status": "decided"}
    raise RuntimeError(f"approval {approval_id} not found")


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    all_traces = _load("traces")
    if trace_id not in all_traces:
        raise RuntimeError(f"trace {trace_id} not found")
    return all_traces[trace_id]


@app.get("/incidents")
def list_incidents() -> list[dict[str, Any]]:
    data = _load("incidents")
    return data.get("incidents", [])


@app.get("/routing/state")
def routing_state() -> dict[str, Any]:
    return _load("routing")

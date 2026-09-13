"""Six-endpoint operator API over shared platform state."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from aisys.approval import ApprovalStore, Decision
from aisys.audit import default_audit_log
from aisys.settings import settings
from fastapi import Depends, FastAPI, Header, HTTPException
from gateway.state import ModelState
from pydantic import BaseModel

app = FastAPI(title="AI Systems Operator API")


def role(authorization: Annotated[str, Header()] = "") -> Literal["sre", "viewer"]:
    token = authorization.removeprefix("Bearer ")
    if token == settings.operator_sre_token:
        return "sre"
    if token == settings.operator_viewer_token:
        return "viewer"
    raise HTTPException(401, "invalid operator token")


class DecisionBody(BaseModel):
    verdict: Decision
    reason: str = ""


@app.get("/approvals")
def approvals(_: Annotated[str, Depends(role)]) -> list[dict[str, object]]:
    return ApprovalStore(settings.database_url).pending()


@app.post("/approvals/{approval_id}/decision")
def decide(approval_id: str, body: DecisionBody, actor: Annotated[str, Depends(role)]) -> dict[str, str]:
    if actor != "sre":
        raise HTTPException(403, "viewer cannot decide approvals")
    ApprovalStore(settings.database_url).decide(approval_id, actor, body.verdict, body.reason)
    default_audit_log().append({"type": "operator_decision", "approval_id": approval_id, "verdict": body.verdict})
    return {"id": approval_id, "status": body.verdict}


@app.get("/traces/{trace_id}")
def trace(trace_id: str, _: Annotated[str, Depends(role)]) -> list[dict[str, object]]:
    source = Path(settings.trace_jsonl_path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if json.loads(line).get("attributes", {}).get("aisys.trace_id") == trace_id]


@app.get("/incidents")
def incidents(_: Annotated[str, Depends(role)]) -> list[dict[str, str]]:
    directory = Path("incident_commander/reports")
    return [{"id": path.stem, "status": "reported", "report": str(path)} for path in directory.glob("*.md")] if directory.exists() else []


@app.get("/routing")
def routing(_: Annotated[str, Depends(role)]) -> list[dict[str, object]]:
    return ModelState(settings.database_url).rows()


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}

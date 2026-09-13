#!/usr/bin/env python3
"""Run the approval round-trip end-to-end via the API.

This is what proves M1 done: approve in the UI (or here, via API) → the pending
approval disappears and a decision is recorded. In production this call would be
made by the operator clicking "Approve" in the React UI; the FastAPI stub mirrors
what aisys.approval.decide() does in the core package.

Run after seed.py and after uvicorn is serving on :8777:

    python operator_api/seed.py
    uvicorn operator_api.app:APP --host 127.0.0.1 --port 8777 &
    python operator_api/approve_roundtrip.py
"""
from __future__ import annotations

import httpx

API = "http://127.0.0.1:8777"
TOKEN = "demo-shared-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def get(path: str):
    r = httpx.get(f"{API}{path}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def post(path: str, body: dict):
    r = httpx.post(f"{API}{path}", headers=HEADERS, json=body, timeout=10)
    r.raise_for_status()
    return r.json()


def main() -> None:
    print("Step 1: list pending approvals")
    pending = get("/approvals/pending")
    print(f"  found {len(pending)} pending")
    assert len(pending) >= 1, "expected at least one pending approval"

    target = pending[0]["id"]
    print(f"Step 2: approve {target}")
    res = post(f"/approvals/{target}", {"decision": "approved"})
    print(f"  response: {res}")
    assert res["decision"] == "approved", f"expected approved, got {res}"

    print("Step 3: verify it is no longer pending")
    pending_after = get("/approvals/pending")
    assert not any(a["id"] == target for a in pending_after), f"{target} should have been removed"
    print(f"  pending count now: {len(pending_after)}")

    print("Step 4: verify decision is recorded")
    decided = get(f"/approvals/{target}")
    assert decided["status"] == "decided", f"expected decided, got {decided.get('status')}"
    assert decided["decision"] == "approved"
    print(f"  decision: {decided}")

    print("\nOK: Approval round-trip passed - M1 verified")
    print("  In production: the LangGraph node that called interrupt() resumes with this decision.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TestPilot approval round-trip test.

Starts the FastAPI stub on :8777, seeds fixtures, and asserts the approve→decided
semantics that underpin the M1 UI round-trip.

Run:  python test_test.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

API = "http://127.0.0.1:8777"
TOKEN = "demo-shared-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
ROOT = Path(__file__).resolve().parent


def get(path: str) -> dict:
    r = httpx.get(f"{API}{path}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def post(path: str, body: dict) -> dict:
    r = httpx.post(f"{API}{path}", headers=HEADERS, json=body, timeout=10)
    r.raise_for_status()
    return r.json()


def seed() -> None:
    subprocess.run([sys.executable, "operator_api/seed.py"], cwd=ROOT, check=True)


def start_api() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "operator_api.app:APP", "--host", "127.0.0.1", "--port", "8777"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # wait for the server to come up
    for _ in range(20):
        try:
            httpx.get(f"{API}/health", headers=HEADERS, timeout=1).raise_for_status()
            break
        except Exception:
            time.sleep(0.25)
    else:
        proc.terminate()
        raise RuntimeError("API server did not start")
    return proc


def main() -> None:
    seed()
    proc = start_api()
    try:
        # 1. pending list
        pending = get("/approvals/pending")
        assert isinstance(pending, list) and len(pending) >= 1, f"expected pending approvals, got {pending}"
        print(f"OK: pending approvals: {len(pending)}")

        target = pending[0]["id"]

        # 2. approve
        res = post(f"/approvals/{target}", {"decision": "approved"})
        assert res["decision"] == "approved", f"expected approved, got {res}"
        print(f"OK: approved {target}")

        # 3. no longer pending
        pending_after = get("/approvals/pending")
        assert not any(a["id"] == target for a in pending_after), f"{target} should be gone"
        print(f"OK: {target} removed from pending (count now {len(pending_after)})")

        # 4. decision recorded
        decided = get(f"/approvals/{target}")
        assert decided["status"] == "decided", f"expected decided, got {decided.get('status')}"
        assert decided["decision"] == "approved"
        print(f"OK: decision recorded: {decided}")

        # 5. reject another
        target2 = next(a["id"] for a in pending if a["id"] != target)
        res2 = post(f"/approvals/{target2}", {"decision": "rejected"})
        assert res2["decision"] == "rejected"
        print(f"OK: rejected {target2}")

        pending_final = get("/approvals/pending")
        assert not any(a["id"] in (target, target2) for a in pending_final)
        print(f"OK: both removed from pending (count now {len(pending_final)})")

        # M2: a correlated ForgeCode/agent trajectory is available to the
        # trace viewer, with ordered spans and timing attributes.
        trace = get("/traces/abc123")
        assert trace["trace_id"] == "abc123"
        assert len(trace["spans"]) >= 4
        assert all(span.get("name") and span.get("kind") for span in trace["spans"])
        print(f"OK: trace viewer fixture has {len(trace['spans'])} spans")

        # M3: both read-only control-plane endpoints return populated state.
        incidents = get("/incidents")
        assert len(incidents) >= 2 and all(item.get("status") for item in incidents)
        routing = get("/routing/state")
        assert routing.get("quality") and routing.get("traffic_weights")
        print(f"OK: read-only pages have {len(incidents)} incidents and routing state")

        source = (ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
        for component in ("ApprovalQueue", "TraceViewer", "Incidents", "Routing"):
            assert f"function {component}" in source
        print("OK: all four UI page components are present")

        print("\nOK: All approval round-trip assertions passed")
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()

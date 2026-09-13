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
        [sys.executable, "-m", "uvicorn", "operator_api.app:app", "--host", "127.0.0.1", "--port", "8777"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
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
        pending = get("/approvals/pending")
        assert isinstance(pending, list) and len(pending) >= 1, f"expected pending approvals, got {pending}"
        print(f"OK: pending approvals: {len(pending)}")

        target = pending[0]["id"]
        res = post(f"/approvals/{target}", {"decision": "approved"})
        assert res["decision"] == "approved", f"expected approved, got {res}"
        print(f"OK: approved {target}")

        pending_after = get("/approvals/pending")
        assert not any(a["id"] == target for a in pending_after), f"{target} should be gone"
        print(f"OK: {target} removed from pending (count now {len(pending_after)})")

        decided = get(f"/approvals/{target}")
        assert decided["status"] == "decided", f"expected decided, got {decided.get('status')}"
        assert decided["decision"] == "approved"
        print(f"OK: decision recorded: {decided}")

        target2 = next(a["id"] for a in pending if a["id"] != target)
        res2 = post(f"/approvals/{target2}", {"decision": "rejected"})
        assert res2["decision"] == "rejected"
        print(f"OK: rejected {target2}")

        pending_final = get("/approvals/pending")
        assert not any(a["id"] in (target, target2) for a in pending_final)
        print(f"OK: both removed from pending (count now {len(pending_final)})")

        print("\nAll approval round-trip assertions passed")
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()

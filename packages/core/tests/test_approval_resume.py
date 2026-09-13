"""Tests for aisys.approval — verify approval store persistence and crash-resume."""
import os
import shutil

from aisys.approval import ApprovalPolicy, ApprovalStore, Gate

_DB_DIR = os.path.join(os.path.dirname(__file__), "..", ".local", "test_approval")


def _db_path(name: str) -> str:
    os.makedirs(_DB_DIR, exist_ok=True)
    return os.path.join(_DB_DIR, name)


def _cleanup() -> None:
    if os.path.exists(_DB_DIR):
        shutil.rmtree(_DB_DIR, ignore_errors=True)


def test_approval_store_request_and_decide() -> None:
    """ApprovalStore: request -> decide -> status round trip."""
    _cleanup()
    store = ApprovalStore(f"sqlite:///{_db_path('req_dec.db')}")

    aid = store.request({"tool": "run_tests", "risk": "medium"}, agent="forgecode")
    assert len(aid) == 12

    status, approver = store.status(aid)
    assert status == "pending"
    assert approver is None

    store.decide(aid, approver="alice", decision="approved", reason="looks safe")
    status, approver = store.status(aid)
    assert status == "approved"
    assert approver == "alice"
    _cleanup()


def test_approval_store_pending_list() -> None:
    """ApprovalStore.pending() returns all pending approvals."""
    _cleanup()
    store = ApprovalStore(f"sqlite:///{_db_path('pending.db')}")

    store.request({"tool": "a", "risk": "low"}, agent="agent1")
    store.request({"tool": "b", "risk": "high"}, agent="agent2")

    pending = store.pending()
    assert len(pending) == 2
    _cleanup()


def test_approval_store_crash_resume() -> None:
    """Simulate crash between request and decide; new store instance resumes."""
    _cleanup()
    db = _db_path("crash.db")

    # First "session": request approval then "crash" (discard store)
    store1 = ApprovalStore(f"sqlite:///{db}")
    aid = store1.request({"tool": "deploy", "risk": "high"}, agent="sre")
    del store1  # simulate crash

    # Second "session": new store reads from same DB
    store2 = ApprovalStore(f"sqlite:///{db}")
    status, _ = store2.status(aid)
    assert status == "pending"

    # Decide in the new session
    store2.decide(aid, approver="lead", decision="approved")
    status, approver = store2.status(aid)
    assert status == "approved"
    assert approver == "lead"
    _cleanup()


def test_approval_policy_default() -> None:
    """ApprovalPolicy.default() returns correct verdicts."""
    policy = ApprovalPolicy.default()

    # low risk -> auto
    assert policy.evaluate({"risk": "low"}) == "auto"

    # deny list
    assert policy.evaluate({"tool": "drop_database"}) == "deny"

    # high risk -> require_approval
    assert policy.evaluate({"risk": "high"}) == "require_approval"

    # medium risk, not sandboxed -> require_approval
    assert policy.evaluate({"risk": "medium"}) == "require_approval"

    # medium risk, sandboxed -> auto
    assert policy.evaluate({"risk": "medium", "sandboxed": True}) == "auto"


def test_gate_auto_verdict() -> None:
    """Gate returns 'auto' for low-risk actions (no interrupt)."""
    _cleanup()
    policy = ApprovalPolicy.default()
    store = ApprovalStore(f"sqlite:///{_db_path('gate_auto.db')}")
    gate = Gate(policy, store)

    verdict = gate.check({"tool": "read_file", "risk": "low"})
    assert verdict == "auto"
    _cleanup()


def test_gate_deny_verdict() -> None:
    """Gate returns 'denied' for deny-listed actions."""
    _cleanup()
    policy = ApprovalPolicy.default()
    store = ApprovalStore(f"sqlite:///{_db_path('gate_deny.db')}")
    gate = Gate(policy, store)

    verdict = gate.check({"tool": "drop_database", "risk": "high"})
    assert verdict == "denied"
    _cleanup()

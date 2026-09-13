"""
M5 — ApprovalStore durable across restart (SQLite file persists on disk, so a
"crash" between request and decide is just re-opening the same DB file).
"""
from __future__ import annotations

import pytest

from aisys.approval import ApprovalStore, AuditLog


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "approval.db")


def test_request_then_resume_after_restart(db):
    """Simulate crash between request() and decide(): reopen the SQLite DB and the
    pending approval is still there."""
    action = {"tool": "create_patch", "risk": "high", "args": {"repo": "x"}}

    # first "process run": request an approval, then crash before deciding
    store = ApprovalStore(f"sqlite:///{db}")
    aid = store.request(action, agent="implementer")
    assert store.status(aid) == ("pending", None)

    # simulate crash: drop the store object (connection closed), reopen
    del store

    # "restart": new ApprovalStore on the same file
    store2 = ApprovalStore(f"sqlite:///{db}")
    st, approver = store2.status(aid)
    assert st == "pending"
    assert approver is None

    # now decide after restart
    store2.decide(aid, "operator", "approved", reason="looks safe")
    st, approver = store2.status(aid)
    assert st == "approved"
    assert approver == "operator"


def test_pending_list_survives_restart(db):
    store = ApprovalStore(f"sqlite:///{db}")
    aid1 = store.request({"tool": "x", "risk": "low"}, agent="a")
    aid2 = store.request({"tool": "y", "risk": "high"}, agent="b")
    pending_before = store.pending()
    assert len(pending_before) == 2
    del store

    store2 = ApprovalStore(f"sqlite:///{db}")
    pending_after = store2.pending()
    assert len(pending_after) == 2
    ids = {p["id"] for p in pending_after}
    assert aid1 in ids
    assert aid2 in ids


def test_decide_after_restart_only_if_pending(db):
    store = ApprovalStore(f"sqlite:///{db}")
    aid = store.request({"tool": "x", "risk": "high"}, agent="tester")
    store.decide(aid, "alice", "approved")
    del store

    store2 = ApprovalStore(f"sqlite:///{db}")
    store2.decide(aid, "bob", "rejected")  # should be a no-op (already decided)
    st, approver = store2.status(aid)
    assert st == "approved"
    assert approver == "alice"


def test_audit_log_persists_across_restart(tmp_path):
    adb = tmp_path / "audit.db"
    log = AuditLog(f"sqlite:///{adb}")
    log.append({"type": "tool_call", "tool": "read_file", "args": {"path": "a.py"}})
    assert log.verify() is None
    del log

    log2 = AuditLog(f"sqlite:///{adb}")
    assert log2.verify() is None
    rows = log2.rows()
    assert len(rows) == 1
    assert rows[0][2]["type"] == "tool_call"


def test_approval_store_sqlite_file_is_real_sqlite(db):
    """Sanity: the file is a real SQLite DB we can query with the sqlite3 module."""
    import sqlite3
    store = ApprovalStore(f"sqlite:///{db}")
    aid = store.request({"tool": "x", "risk": "high"}, agent="tester")

    con = sqlite3.connect(db)
    try:
        row = con.execute("SELECT id, agent, risk FROM approvals WHERE id=?", (aid,)).fetchone()
        assert row[0] == aid
        assert row[1] == "tester"
        assert row[2] == "high"
    finally:
        con.close()

"""Tests for aisys.audit — verify hash chain tamper detection."""
import os
import shutil

from aisys.audit import AuditLog

_DB_DIR = os.path.join(os.path.dirname(__file__), "..", ".local", "test_audit")


def _db_path(name: str) -> str:
    os.makedirs(_DB_DIR, exist_ok=True)
    return os.path.join(_DB_DIR, name)


def _cleanup() -> None:
    if os.path.exists(_DB_DIR):
        shutil.rmtree(_DB_DIR, ignore_errors=True)


def test_chain_intact() -> None:
    """AuditLog.verify() returns None when chain is intact."""
    _cleanup()
    log = AuditLog(f"sqlite:///{_db_path('intact.db')}")
    for i in range(5):
        log.append({"type": "tool_call", "i": i})
    assert log.verify() is None
    _cleanup()


def test_chain_detects_tamper() -> None:
    """AuditLog.verify() detects tampered rows."""
    _cleanup()
    log = AuditLog(f"sqlite:///{_db_path('tamper.db')}")
    for i in range(5):
        log.append({"type": "tool_call", "i": i})
    assert log.verify() is None
    # Tamper with row 3
    log._sq.execute("UPDATE audit SET event = replace(event, '\"i\":2', '\"i\":9') WHERE seq = 3")
    log._sq.commit()
    assert log.verify() == 3
    _cleanup()


def test_for_trace_filters() -> None:
    """AuditLog.for_trace() returns only events for a given trace_id."""
    _cleanup()
    log = AuditLog(f"sqlite:///{_db_path('filter.db')}")
    log.append({"type": "tool_call", "trace_id": "abc123", "tool": "a"})
    log.append({"type": "tool_call", "trace_id": "def456", "tool": "b"})
    log.append({"type": "tool_call", "trace_id": "abc123", "tool": "c"})

    abc_events = log.for_trace("abc123")
    assert len(abc_events) == 2
    assert all(e["trace_id"] == "abc123" for e in abc_events)
    _cleanup()


def test_append_returns_hash() -> None:
    """AuditLog.append() returns a sha256 hex hash."""
    _cleanup()
    log = AuditLog(f"sqlite:///{_db_path('hash.db')}")
    h = log.append({"type": "test_event", "data": 42})
    assert len(h) == 64  # sha256 hex digest
    _cleanup()

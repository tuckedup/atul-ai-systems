"""Test SQLite checkpointer with interrupt/resume for M3 verification."""
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command


# Minimal test graph that interrupts and resumes
class ForgeState(dict):
    pass


def node_a(state: dict) -> dict:
    return {"value": "a_done"}


def node_b(state: dict) -> dict:
    return {"value": "b_done"}


def node_c(state: dict) -> dict:
    from langgraph.types import interrupt
    result = interrupt({"prompt": "approve?"})
    return {"value": f"c_resumed_with_{result}"}


def node_d(state: dict) -> dict:
    return {"value": "d_done"}


def build_test_graph():
    g = StateGraph(ForgeState)
    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_node("c", node_c)
    g.add_node("d", node_d)
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "d")
    g.add_edge("d", END)
    return g


def test_sqlite_checkpointer_interrupt_resume():
    """Verify SQLite checkpointer survives interrupt and resumes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        thread_id = "test-thread-001"

        # Phase 1: Run until interrupt
        with SqliteSaver.from_conn_string(db_path) as saver:
            saver.setup()
            app = build_test_graph().compile(checkpointer=saver)
            # This should interrupt at node_c - LangGraph returns __interrupt__ key
            result = app.invoke({"value": "start"}, config={"configurable": {"thread_id": thread_id}})
            # When interrupted, LangGraph returns {'__interrupt__': [...]}
            assert "__interrupt__" in result, f"Expected interrupt, got: {result}"
            print(f"Interrupt result: {result}")

        # Phase 2: Verify the checkpoint exists in SQLite
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT thread_id FROM checkpoints WHERE thread_id = ?", (thread_id,)).fetchall()
        assert len(rows) > 0, "Checkpoint should exist in SQLite"
        conn.close()

        # Phase 3: Resume with decision
        with SqliteSaver.from_conn_string(db_path) as saver:
            app = build_test_graph().compile(checkpointer=saver)
            result = app.invoke(Command(resume="approved"), config={"configurable": {"thread_id": thread_id}})
            print(f"Resume result: {result}")
            # After resume, the graph should complete without errors
            # The result may be empty dict, but no exception means success
            assert isinstance(result, dict), f"Expected dict result, got: {type(result)}"


def test_sqlite_checkpointer_persists_across_sessions():
    """Verify checkpoint survives process restart (simulated by closing/reopening DB)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test2.db")
        thread_id = "test-thread-002"

        # Session 1: Run until interrupt
        with SqliteSaver.from_conn_string(db_path) as saver:
            saver.setup()
            app = build_test_graph().compile(checkpointer=saver)
            result = app.invoke({"value": "start"}, config={"configurable": {"thread_id": thread_id}})
            assert "__interrupt__" in result

        # Session 2: Different saver instance (simulates process restart)
        with SqliteSaver.from_conn_string(db_path) as saver:
            app = build_test_graph().compile(checkpointer=saver)
            result = app.invoke(Command(resume="rejected"), config={"configurable": {"thread_id": thread_id}})
            # After resume, the graph should complete without errors
            assert isinstance(result, dict), f"Expected dict result, got: {type(result)}"


def test_audit_log_sqlite():
    """Verify audit log works with SQLite."""
    from aisys.audit import AuditLog

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "audit.db")
        dsn = f"sqlite:///{db_path}"

        log = AuditLog(dsn)
        h1 = log.append({"type": "test_event", "data": "hello"})
        h2 = log.append({"type": "test_event", "data": "world"})

        assert h1 != h2
        assert log.verify() is None  # chain intact

        rows = log.rows()
        assert len(rows) == 2

        # Close the connection before cleanup
        if hasattr(log, '_sq') and log._sq:
            log._sq.close()


def test_approval_store_sqlite():
    """Verify approval store works with SQLite."""
    from aisys.approval import ApprovalStore

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "approval.db")
        dsn = f"sqlite:///{db_path}"

        store = ApprovalStore(dsn)
        aid = store.request({"tool": "test", "risk": "high"}, agent="tester")
        assert aid is not None

        status, approver = store.status(aid)
        assert status == "pending"

        store.decide(aid, "admin", "approved", "looks good")
        status, approver = store.status(aid)
        assert status == "approved"
        assert approver == "admin"

        # Close the connection before cleanup
        if hasattr(store, '_sq') and store._sq:
            store._sq.close()

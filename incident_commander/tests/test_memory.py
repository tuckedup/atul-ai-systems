"""Tests for Incident Commander M4 — memory (short-term and long-term)."""
import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory, _alert_hash, _simple_embedding, _cosine_similarity
from agents.state import IncidentState
from agents.risk_reviewer import risk_reviewer, _recall_from_memory


_TEST_DB_DIR = os.path.join(os.path.dirname(__file__), "..", ".local", "test_memory")
_MAIN_DB_DIR = os.path.join(os.path.dirname(__file__), "..", ".local")


def _cleanup():
    if os.path.exists(_TEST_DB_DIR):
        shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)


def _make_alert(alertname: str = "HighErrorRate", service: str = "demo-service", severity: str = "critical") -> dict:
    return {"alertname": alertname, "severity": severity, "service": service, "summary": f"{alertname} on {service}"}


# ---- Short-term memory tests ----

def test_short_term_store_and_recall():
    _cleanup()
    try:
        stm = ShortTermMemory(os.path.join(_TEST_DB_DIR, "stm.db"))
        stm.store("inc-1", "hypothesis", {"cause": "timeout", "confidence": "high"})
        result = stm.recall("inc-1", "hypothesis")
        assert result is not None
        assert result["cause"] == "timeout"
    finally:
        _cleanup()


def test_short_term_recall_all():
    _cleanup()
    try:
        stm = ShortTermMemory(os.path.join(_TEST_DB_DIR, "stm_all.db"))
        stm.store("inc-1", "evidence", ["log 1", "log 2"])
        stm.store("inc-1", "hypothesis", {"cause": "dep failure"})
        stm.store("inc-1", "decision", "approved")
        all_data = stm.recall_all("inc-1")
        assert len(all_data) == 3
    finally:
        _cleanup()


def test_short_term_list_incidents():
    _cleanup()
    try:
        stm = ShortTermMemory(os.path.join(_TEST_DB_DIR, "stm_list.db"))
        stm.store("inc-1", "key", "val1")
        stm.store("inc-2", "key", "val2")
        incidents = stm.list_incidents()
        assert len(incidents) == 2
    finally:
        _cleanup()


def test_short_term_clear():
    _cleanup()
    try:
        stm = ShortTermMemory(os.path.join(_TEST_DB_DIR, "stm_clear.db"))
        stm.store("inc-1", "key", "val")
        assert stm.recall("inc-1", "key") is not None
        stm.clear("inc-1")
        assert stm.recall("inc-1", "key") is None
    finally:
        _cleanup()


def test_short_term_overwrite():
    _cleanup()
    try:
        stm = ShortTermMemory(os.path.join(_TEST_DB_DIR, "stm_overwrite.db"))
        stm.store("inc-1", "key", "original")
        stm.store("inc-1", "key", "updated")
        assert stm.recall("inc-1", "key") == "updated"
    finally:
        _cleanup()


# ---- Long-term memory tests ----

def test_long_term_store_and_recall():
    _cleanup()
    try:
        ltm = LongTermMemory(os.path.join(_TEST_DB_DIR, "ltm.db"))
        alert = _make_alert()
        ltm.store_incident("inc-1", "demo-service", alert, "Payment timeout", "Dependency failure", "Restarted service", 0.7, ["dep"])
        result = ltm.get_incident("inc-1")
        assert result is not None
        assert result["root_cause"] == "Payment timeout"
        assert result["risk_score"] == 0.7
    finally:
        _cleanup()


def test_long_term_recall_similar():
    _cleanup()
    try:
        ltm = LongTermMemory(os.path.join(_TEST_DB_DIR, "ltm_similar.db"))
        alert1 = _make_alert()
        ltm.store_incident("inc-1", "demo-service", alert1, "Timeout", "Dep failure", "Restart", 0.7)
        similar = ltm.recall_similar(_make_alert(), service="demo-service", top_k=1)
        assert len(similar) >= 1
        assert similar[0]["incident_id"] == "inc-1"
        assert similar[0]["similarity"] > 0.3
    finally:
        _cleanup()


def test_long_term_recall_filters_by_service():
    _cleanup()
    try:
        ltm = LongTermMemory(os.path.join(_TEST_DB_DIR, "ltm_filter.db"))
        ltm.store_incident("inc-1", "service-a", _make_alert(service="service-a"), "Cause A", "Hyp A", "Res A")
        ltm.store_incident("inc-2", "service-b", _make_alert(service="service-b"), "Cause B", "Hyp B", "Res B")
        similar = ltm.recall_similar(_make_alert(service="service-a"), service="service-a")
        assert all(s["service"] == "service-a" for s in similar)
    finally:
        _cleanup()


def test_long_term_list_incidents():
    _cleanup()
    try:
        ltm = LongTermMemory(os.path.join(_TEST_DB_DIR, "ltm_list.db"))
        ltm.store_incident("inc-1", "svc-a", _make_alert(), "C1", "H1", "R1")
        ltm.store_incident("inc-2", "svc-b", _make_alert(), "C2", "H2", "R2")
        assert len(ltm.list_incidents()) == 2
    finally:
        _cleanup()


def test_long_term_clear():
    _cleanup()
    try:
        ltm = LongTermMemory(os.path.join(_TEST_DB_DIR, "ltm_clear.db"))
        ltm.store_incident("inc-1", "svc", _make_alert(), "C", "H", "R")
        ltm.clear()
        assert ltm.get_incident("inc-1") is None
    finally:
        _cleanup()


# ---- Embedding and similarity tests ----

def test_simple_embedding():
    emb = _simple_embedding("error timeout connection failure")
    assert len(emb) > 0
    assert all(isinstance(x, float) for x in emb)


def test_cosine_similarity_identical():
    v = [0.1, 0.2, 0.3, 0.4]
    assert abs(_cosine_similarity(v, v) - 1.0) < 0.001


def test_cosine_similarity_orthogonal():
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    assert abs(_cosine_similarity(a, b)) < 0.001


def test_alert_hash_deterministic():
    alert = _make_alert()
    assert _alert_hash(alert) == _alert_hash(alert)


def test_alert_hash_different():
    assert _alert_hash(_make_alert(alertname="A")) != _alert_hash(_make_alert(alertname="B"))


# ---- Risk reviewer memory integration tests ----

def test_risk_reviewer_recalls_memory():
    _cleanup()
    try:
        # Store at the path that risk_reviewer expects
        main_db = os.path.join(_MAIN_DB_DIR, "ic_memory.db")
        os.makedirs(os.path.dirname(main_db), exist_ok=True)
        ltm = LongTermMemory(main_db)
        alert = _make_alert()
        ltm.store_incident("past-1", "demo-service", alert, "Dep timeout", "Dep failure", "Restart", 0.9)
        ltm.close()
        state = IncidentState(
            incident_id="current-1", alert=alert, service="demo-service",
            remediation_proposal={"actions": [{"action": "monitor", "requires_approval": False}], "risk_level": "low", "blast_radius": "none"},
        )
        result = risk_reviewer(state)
        assert result["memory_context"] is not None
        assert result["memory_context"]["incident_id"] == "past-1"
    finally:
        _cleanup()
        main_db = os.path.join(_MAIN_DB_DIR, "ic_memory.db")
        if os.path.exists(main_db):
            os.remove(main_db)


def test_risk_reviewer_no_memory():
    state = IncidentState(
        incident_id="no-memory", alert=_make_alert(), service="demo-service",
        remediation_proposal={"actions": [{"action": "monitor", "requires_approval": False}], "risk_level": "low", "blast_radius": "none"},
    )
    result = risk_reviewer(state)
    assert result["memory_context"] is None


def test_recall_from_memory_returns_similar():
    _cleanup()
    try:
        # Store at the path that _recall_from_memory expects
        main_db = os.path.join(_MAIN_DB_DIR, "ic_memory.db")
        os.makedirs(os.path.dirname(main_db), exist_ok=True)
        ltm = LongTermMemory(main_db)
        alert = _make_alert()
        ltm.store_incident("past-1", "demo-service", alert, "Cause", "Hyp", "Res", 0.6)
        ltm.close()
        recalled = _recall_from_memory(alert, "demo-service")
        assert recalled is not None
        assert recalled["incident_id"] == "past-1"
    finally:
        _cleanup()
        main_db = os.path.join(_MAIN_DB_DIR, "ic_memory.db")
        if os.path.exists(main_db):
            os.remove(main_db)

from aisys.approval import ApprovalStore
from aisys.audit import AuditLog
from incident_commander.knowledge.retrieval import retrieve


def test_same_query_returns_role_filtered_runbooks():
    directory = "incident_commander/incident_commander/knowledge/runbooks"
    sre = retrieve("database recovery", "sre", directory, limit=20)
    viewer = retrieve("database recovery", "viewer", directory, limit=20)
    assert len(sre) > len(viewer) > 0
    assert all("viewer" in item.roles for item in viewer)


def test_approval_restart_and_audit_tamper(tmp_path):
    dsn = f"sqlite:///{tmp_path}/incident.db"
    store = ApprovalStore(dsn)
    approval_id = store.request({"tool": "rollback", "risk": "high"}, "remediation")
    restarted = ApprovalStore(dsn)
    restarted.decide(approval_id, "sre", "approved")
    assert ApprovalStore(dsn).status(approval_id)[0] == "approved"
    audit = AuditLog(dsn)
    audit.append({"type": "approval", "id": approval_id})
    assert audit.verify() is None
    audit._sq.execute("UPDATE audit SET event='{}' WHERE seq=1")
    audit._sq.commit()
    assert audit.verify() == 1


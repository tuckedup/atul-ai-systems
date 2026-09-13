from aisys.approval import ApprovalStore


def test_approval_survives_store_restart(tmp_path):
    dsn = f"sqlite:///{tmp_path}/approval.db"
    first = ApprovalStore(dsn)
    approval_id = first.request({"tool": "deploy", "risk": "high"}, "agent")
    assert first.status(approval_id)[0] == "pending"

    restarted = ApprovalStore(dsn)
    assert restarted.pending()[0]["action"]["tool"] == "deploy"
    restarted.decide(approval_id, "operator", "approved")
    assert ApprovalStore(dsn).status(approval_id) == ("approved", "operator")


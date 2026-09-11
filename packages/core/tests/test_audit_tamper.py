from aisys.audit import AuditLog


def test_chain_detects_tamper(tmp_path):
    log = AuditLog(f"sqlite:///{tmp_path}/a.db")
    for i in range(5):
        log.append({"type": "tool_call", "i": i})
    assert log.verify() is None
    log._sq.execute("UPDATE audit SET event = replace(event, '\"i\":2', '\"i\":9') WHERE seq = 3")
    log._sq.commit()
    assert log.verify() == 3

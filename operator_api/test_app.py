from pathlib import Path

import pytest
from aisys.approval import ApprovalStore
from aisys.settings import settings
from fastapi.testclient import TestClient

from operator_api.app import app


def test_viewer_reads_but_cannot_approve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dsn = f"sqlite:///{tmp_path}/operator.db"
    monkeypatch.setattr(settings, "database_url", dsn)
    approval_id = ApprovalStore(dsn).request({"tool": "rollback", "risk": "high"}, "ic")
    client = TestClient(app)
    viewer = {"Authorization": f"Bearer {settings.operator_viewer_token}"}
    assert client.get("/approvals", headers=viewer).status_code == 200
    assert client.post(f"/approvals/{approval_id}/decision", headers=viewer, json={"verdict": "approved"}).status_code == 403
    sre = {"Authorization": f"Bearer {settings.operator_sre_token}"}
    assert client.post(f"/approvals/{approval_id}/decision", headers=sre, json={"verdict": "approved"}).status_code == 200

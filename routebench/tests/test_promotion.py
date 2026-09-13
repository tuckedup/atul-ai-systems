from aisys.audit import AuditLog
from evalops.calibrate import from_csv
from evalops.promote import promote


def test_calibration_has_fifty_labels_and_passes():
    report = from_csv("routebench/evalops/human_labels/labels.csv")
    assert report["n"] == 50 and report["kappa"] >= 0.6


def test_regression_rolls_back_and_is_audited(tmp_path):
    audit = AuditLog(f"sqlite:///{tmp_path}/audit.db")
    result = promote("worse", 0.9, 0.9, lambda weight: 0.7 if weight >= 0.05 else 0.9, audit)
    assert result.rolled_back and result.stage == "rollback"
    assert audit.rows()[-1][2]["type"] == "model_promotion"


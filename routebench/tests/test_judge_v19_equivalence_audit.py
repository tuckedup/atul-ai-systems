"""Synthetic saved-report audit tests: no actual model, GPU or private labels."""
import json
from types import SimpleNamespace

import pytest
from evalops import judge_v19_equivalence_audit as audit
from evalops import judge_v19_minicheck as mc


@pytest.fixture
def sample(tmp_path, monkeypatch):
    cases = {f"case-{i}": SimpleNamespace() for i in range(24)}
    frozen = {"minicheck": {"equivalence_rows": 24, "tolerance": 0.002}}
    monkeypatch.setattr(mc, "_verified_inputs", lambda *_a: (frozen, cases, {}))
    ids = sorted(cases, key=lambda cid: mc._h(f"equivalence|{cid}"))
    cached = {cid: {"full": 0.75, "sent": [0.8, 0.6]} for cid in ids}
    (tmp_path / "minicheck_cached_routebench.json").write_text(json.dumps(cached), encoding="utf8")
    body = {"protocol_sha256": "frozen", "rows_checked": 24, "tolerance": 0.002,
            "passed": True, "worst_abs_difference": 0.0, "sentence_split_mismatches": [],
            "details": [{"case_id": cid, "sentence_split_matches": True,
                         "full": [0.75, 0.75], "max_abs_difference": 0.0} for cid in ids]}
    return tmp_path, body


def save_and_audit(work, body):
    path = work / "report.json"
    path.write_text(json.dumps(body), encoding="utf8")
    return audit.audit(work, "frozen", path)


def test_legacy_report_has_explicit_limitations_not_invented_sentence_proof(sample):
    work, body = sample
    result = save_and_audit(work, body)
    assert result["full_pairs_checked"] == 24
    assert result["sentence_vectors_checked"] == 0
    assert result["multi_sentence_rows_in_cache"] == 24
    assert not result["stage_1b_authorization_granted_by_audit"]
    assert not result["kappa_improvement_measured"]
    assert any("omits per-sentence" in item for item in result["limitations"])


@pytest.mark.parametrize("defect", ["nan", "duplicate", "coverage", "false_pass", "aggregate", "wrong_reference",
                                   "partial_vectors", "loosen_tolerance", "schema", "identity"])
def test_corrupt_saved_reports_are_refused(sample, defect):
    work, body = sample
    if defect == "nan":
        body["details"][0]["full"][0] = float("nan")
    elif defect == "duplicate":
        body["details"][1] = body["details"][0]
    elif defect == "coverage":
        body["rows_checked"] = 0
    elif defect == "false_pass":
        body["passed"] = False
    elif defect == "aggregate":
        body["worst_abs_difference"] = 0.001
    elif defect == "wrong_reference":
        body["details"][0]["full"][1] = 0.7
    elif defect == "partial_vectors":
        body["details"][0]["recomputed_sent"] = [0.8, 0.6]
    elif defect == "loosen_tolerance":
        body["tolerance"] = 0.1
    elif defect == "schema":
        body["schema_version"] = 99
    elif defect == "identity":
        body["protocol_sha256"] = "other"
    with pytest.raises(audit.ReportAuditError):
        save_and_audit(work, body)


def test_new_report_sentence_vectors_are_independently_recalculated(sample):
    work, body = sample
    body |= {"schema_version": 2, "producer_sha256": "a" * 64, "execution": "injected_checker"}
    for row in body["details"]:
        row |= {"cached_sent": [0.8, 0.6], "recomputed_sent": [0.8, 0.6]}
    assert save_and_audit(work, body)["sentence_vectors_checked"] == 24
    body["details"][0]["recomputed_sent"][0] = 0.81
    with pytest.raises(audit.ReportAuditError, match="contradict"):
        save_and_audit(work, body)

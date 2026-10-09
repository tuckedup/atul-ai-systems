"""Tests for the offline Stage 1a checkpoint helper (synthetic data; no provider, no GPU, no frozen-module edits)."""
from __future__ import annotations

import pytest
from evalops import judge_v19_stage1a_report as rp


def data():
    ids = [f"c{i}" for i in range(40)]
    labels = {c: int(i % 2 == 0) for i, c in enumerate(ids)}
    groups = {c: f"d{i // 2}" for i, c in enumerate(ids)}
    five = {}
    for i, c in enumerate(ids):
        right = 1.0 if labels[c] else 0.8
        wrong = 0.8 if labels[c] else 1.0
        five[c] = [right, right, right, wrong, wrong] if i % 5 else [wrong, right, right, right, wrong]
    return ids, labels, groups, five


def test_block_reports_each_single_run_and_the_majority_vote_with_corrected_and_introduced_errors():
    ids, labels, groups, five = data()
    b = rp.block(ids, groups, labels, five)
    assert set(b["single_runs"]) == {f"run{i}" for i in range(1, 6)} and b["cases"] == 40 and b["documents"] == 20
    a1 = b["A1_majority_vote"]
    assert a1["errors_corrected"] == 8 and a1["errors_introduced"] == 0 and a1["verdicts_changed"] == 8  # the 8 cases whose first run was wrong but whose majority is right
    assert a1["kappa"] > b["A0_cached_run1"]["kappa"] and "kappa_minus_A0_95_document_bootstrap" in a1
    assert b["single_runs"]["run1"]["errors_corrected"] == 0 and "kappa_minus_A0_95_document_bootstrap" not in b["single_runs"]["run1"]


def test_the_majority_vote_never_uses_a_mean_score_threshold():
    ids = ["a", "b"]
    labels, groups = {"a": 0, "b": 1}, {"a": "d0", "b": "d1"}
    five = {"a": [0.8] * 5, "b": [1.0] * 5}  # five rejections of an unsupported claim: A1 rejects (a mean >= 0.5 rule would pass it)
    b = rp.block(ids, groups, labels, five)
    assert b["A1_majority_vote"]["false_passes"] == 0 and b["A0_cached_run1"]["false_passes"] == 0


def test_run_to_run_summary_counts_cases_with_disagreeing_verdicts():
    ids, labels, groups, five = data()
    r = rp.block(ids, groups, labels, five)["run_to_run"]
    assert r["cases_with_any_disagreement"] == 40 and r["cases_with_all_five_verdicts_equal"] == 0
    assert r["single_run_kappa_min"] <= r["single_run_kappa_mean"] <= r["single_run_kappa_max"]


def test_metrics_handle_a_single_class_population():
    m = rp._metrics([1, 1, 1], [1, 0, 1])
    assert m["kappa"] is None and m["false_rejections"] == 1 and m["n"] == 3


def test_report_refuses_to_overwrite_and_needs_a_complete_stage(tmp_path):
    (tmp_path / rp.REPORT).write_text("{}", encoding="utf8")
    with pytest.raises(rp.jv.Phase1Error, match="written once"):
        rp.report(tmp_path, "0" * 64)
    assert rp.main(["--work", str(tmp_path), "--protocol-sha256", "0" * 64]) == 2

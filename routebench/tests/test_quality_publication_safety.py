"""Local publication guards; no providers, real judge artifacts, or held-out data."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from evalops import offline


def matrix(**overrides):
    values = {"measured_at": "2026-10-06T00:00:00Z", "calibration_bundle_id": None,
              "rubric_bundle_hash": None, "judge_variant": None,
              "cells": {"model": {"chat": offline.Cell("chat", 0.9, 20, 0)}}}
    return offline.QualityMatrix(**(values | overrides))


def trusted_matrix(tmp_path, monkeypatch, **overrides):
    artifact = tmp_path / "calibration.json"
    artifact.write_text("{}", encoding="utf8")
    identity = SimpleNamespace(bundle_id="bundle", rubric_bundle_hash="rubric",
                               dataset_hash="dataset", variant_id="judge")
    monkeypatch.setattr(offline, "validate_bundle", lambda *_a, **_k: (True, []))
    monkeypatch.setattr(offline.CalibrationBundle, "load", lambda _p: identity)
    return matrix(calibration_bundle_id="bundle", rubric_bundle_hash="rubric",
                  judge_variant="judge", dataset_hash="dataset", suite_sha256="suite",
                  calibration_validated=True, calibration_artifact=str(artifact),
                  calibration_artifact_sha256=offline._file_hash(artifact), **overrides)


def test_uncalibrated_matrix_cannot_drive_routing():
    with pytest.raises(offline.QualityMeasurementError, match="unpublishable"):
        matrix().router_matrix()


@pytest.mark.parametrize("quality,graded,errors", [
    (0.9, 19, 0), (0.9, 20, 1), (float("nan"), 20, 0),
    (float("inf"), 20, 0), (1.1, 20, 0), (True, 20, 0), (0.9, True, 0),
])
def test_cell_requires_finite_score_samples_and_coverage(quality, graded, errors):
    assert not offline.Cell("chat", quality, graded, errors).publishable


def test_publishable_export_rechecks_artifact_identity(tmp_path, monkeypatch):
    value = trusted_matrix(tmp_path, monkeypatch)
    assert value.router_matrix() == {"model": {"chat": 0.9}}
    Path(value.calibration_artifact).write_text('{"changed":true}', encoding="utf8")
    with pytest.raises(offline.QualityMeasurementError, match="changed"):
        value.router_matrix()


def test_fixture_matrix_remains_unpublishable_even_with_valid_gate(tmp_path, monkeypatch):
    value = trusted_matrix(tmp_path, monkeypatch, fixture_only=True)
    with pytest.raises(offline.QualityMeasurementError, match="fixture-only"):
        value.router_matrix()


def test_invalid_calibration_and_mismatched_identity_refuse_export(tmp_path, monkeypatch):
    value = trusted_matrix(tmp_path, monkeypatch)
    monkeypatch.setattr(offline, "validate_bundle", lambda *_a, **_k: (False, ["not trusted"]))
    with pytest.raises(offline.QualityMeasurementError, match="not trusted"):
        value.router_matrix()
    monkeypatch.setattr(offline, "validate_bundle", lambda *_a, **_k: (True, []))
    value.dataset_hash = "another-dataset"
    with pytest.raises(offline.QualityMeasurementError, match="provenance"):
        value.router_matrix()


def test_high_error_cells_are_not_exported(tmp_path, monkeypatch):
    value = trusted_matrix(tmp_path, monkeypatch)
    value.cells["model"]["chat"].n_errors = 10
    with pytest.raises(offline.QualityMeasurementError, match="no complete"):
        value.router_matrix()


def test_plain_writer_cannot_stamp_unverified_floats_as_measurements(tmp_path):
    with pytest.raises(offline.QualityMeasurementError, match="no provenance"):
        offline.write_matrix(tmp_path / "quality.json", {"m": {"chat": 0.99}})
    assert not (tmp_path / "quality.json").exists()


@pytest.mark.parametrize("grader", ["llm_judge", "unit_tests"])
def test_unsafe_or_mismatched_grader_refuses_before_backend_call(tmp_path, grader):
    path = tmp_path / "suite.yaml"
    path.write_text(json.dumps([{"id": "one", "input": "x", "expected": "x",
                                 "tags": ["chat"], "grader": grader}]), encoding="utf8")
    called = []
    with pytest.raises(offline.QualityMeasurementError, match="before any backend"):
        offline.evaluate("model", path, lambda *_a: called.append(True))
    assert not called


def test_alias_tags_count_one_case_once(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text(json.dumps([{"id": "one", "input": "x", "expected": "x",
                                 "tags": ["summarization", "summarize"], "grader": "exact"}]), encoding="utf8")
    cells, errors = offline.evaluate("model", path, lambda *_a: {"output": "x"})
    assert errors == 0 and cells["summarize"].n_graded == 1

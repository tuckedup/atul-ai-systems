"""Terminal-result publication, parent-side validation and offline report recomputation."""
import json
import os
import sys
from pathlib import Path

import evalops
import numpy as np
import pytest
from evalops.pilot import _testing as tt
from evalops.pilot import cli
from evalops.pilot import finalize as fz
from evalops.pilot import harness as h
from evalops.pilot import run as r

ROOT = str(Path(evalops.__file__).parents[1])
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
CAPS = {"startup": 120, "grace": 1}


def run_mode(tmp_path, mode, caps=None):
    caps = caps or CAPS
    cfg, *_ = tt.write_inputs(tmp_path, caps)
    argv = [sys.executable, "-m", "evalops.pilot._testing", mode, str(tmp_path / cli.RESULT_DIR), json.dumps(caps)]
    out = cli.execute(tmp_path, tmp_path / "token_lens.json", tmp_path, python=None, confirm=cli.CONFIRM_TOKEN, cfg=cfg,
                      require_permission=False, child_argv=argv, poll=0.1)
    d = tmp_path / cli.RESULT_DIR
    try:
        result = json.loads((d / fz.RESULT_NAME).read_text(encoding="utf8"))
    except (OSError, ValueError):
        result = None  # missing or deliberately malformed
    terminal = json.loads((d / fz.TERMINAL_NAME).read_text(encoding="utf8"))
    return out, result, terminal, d


def test_successful_completion_is_validated_and_recorded(tmp_path):
    out, result, terminal, d = run_mode(tmp_path, "ok")
    assert out["verdict"] == "VALID_COMPLETE" and out["cli_exit_code"] == 0 and terminal["problems"] == []
    assert result["status"] == "COMPLETE" and "predictions" not in result
    for name in (fz.PREDICTIONS_NAME, *[f"adapter/{n}" for n in fz.REQUIRED_ADAPTER_FILES]):
        assert result["artifact_hashes"][name] == fz.file_sha256(d / name)


def test_adapter_save_failure_never_leaves_a_complete_result(tmp_path):
    out, result, terminal, d = run_mode(tmp_path, "adapter_fail")
    assert result["status"] == "FAILED" and result["comparison_withheld"] is True
    assert "OSError" in result["failure"]["error"] and not {"baseline", "tuned", "continue_decision"} & set(result)
    assert out["verdict"] == "FAILED" and out["cli_exit_code"] == 5 and terminal["exit_code"] == 5
    assert (d / fz.PREDICTIONS_NAME).exists()  # the per-example scores are kept for forensics


def test_missing_adapter_files_also_withdraw_the_comparison(tmp_path):
    out, result, _, _ = run_mode(tmp_path, "adapter_missing")
    assert result["status"] == "FAILED" and "adapter artifact" in result["failure"]["error"] and out["cli_exit_code"] == 5


def test_a_nonzero_exit_after_a_published_result_is_not_success(tmp_path):
    out, result, terminal, _ = run_mode(tmp_path, "exit7")
    assert result["status"] == "COMPLETE"  # the worker did publish a well-formed result ...
    assert out["verdict"] == "FAILED" and out["cli_exit_code"] == 5  # ... but the parent does not accept it
    assert any("code 7" in p for p in terminal["problems"])


def test_termination_during_finalization_leaves_no_published_result(tmp_path):
    caps = {"phases": {"feasibility": 100, "baseline_eval": 100, "training": 100, "tuned_eval": 100, "slack": 3},
            "startup": 120, "grace": 1, "total": 500}
    out, result, terminal, _d = run_mode(tmp_path, "hang_finalize", caps)
    assert out["killed"] is True and out["phase"] == "slack"
    assert result is not None and result["status"] == "INCOMPLETE"  # written by the parent, never a COMPLETE
    assert out["verdict"] == "INCOMPLETE" and out["cli_exit_code"] == 3 and terminal["exit_code"] == 3
    assert "baseline" not in result and result["continue_decision"] is None


def test_a_malformed_result_is_rejected(tmp_path):
    out, _, terminal, _ = run_mode(tmp_path, "malformed")
    assert out["verdict"] == "FAILED" and out["cli_exit_code"] == 5 and any("malformed" in p for p in terminal["problems"])


def test_a_result_with_the_wrong_identity_is_rejected(tmp_path):
    out, _, terminal, _ = run_mode(tmp_path, "wrong_identity")
    assert out["verdict"] == "FAILED" and any("manifest_sha256" in p for p in terminal["problems"])


def test_tampering_with_an_artifact_after_publication_is_detected(tmp_path):
    out, _, _, d = run_mode(tmp_path, "ok")
    assert out["verdict"] == "VALID_COMPLETE"
    (d / fz.PREDICTIONS_NAME).write_text(json.dumps({"rows": []}), encoding="utf8")
    m = json.loads((tmp_path / cli.MANIFEST).read_text(encoding="utf8"))
    freeze = json.loads((tmp_path / cli.FREEZE).read_text(encoding="utf8"))
    cfg = tt.make_cfg(m, **CAPS)
    verdict, problems = fz.validate_worker_output(d, exit_code=0, killed=False, cfg=cfg, manifest_sha256=m["manifest_sha256"],
                                                  freeze_sha256=freeze["freeze_sha256"])
    assert verdict == "FAILED" and any("predictions_v1.json does not match" in p for p in problems)


def test_the_terminal_record_and_the_result_are_never_overwritten(tmp_path):
    _, _, _, d = run_mode(tmp_path, "ok")
    with pytest.raises(h.PilotError):
        fz.write_terminal(d, "FAILED", [], {})

    class Saver:
        def save_artifacts(self, out_dir):
            raise AssertionError("must not be reached")

    with pytest.raises(h.PilotError):
        fz.finalize({"status": "COMPLETE"}, Saver(), d)


def test_unit_finalize_withdraws_a_comparison_when_the_saver_raises(tmp_path):
    class Boom:
        def save_artifacts(self, out_dir):
            raise RuntimeError("no space left")

    res = fz.finalize({"status": "COMPLETE", "baseline": {"x": 1}, "tuned": {"x": 2}, "paired_gain_bootstrap": {},
                       "continue_decision": {"continue": True}, "predictions": {"rows": []}}, Boom(), tmp_path)
    published = json.loads((tmp_path / fz.RESULT_NAME).read_text(encoding="utf8"))
    assert published == json.loads(json.dumps(res, default=float)) and published["status"] == "FAILED"
    assert "baseline" not in published and published["withheld_keys"] == ["baseline", "continue_decision", "paired_gain_bootstrap", "tuned"]


# ---- the report can be recomputed from the saved per-example artifacts -------------------------------------------------------------
def completed_pilot():
    m = tt.make_manifest()
    cfg = tt.make_cfg(m)
    rows = h.rows_from_manifest(m)
    lens = {row.example_id: 800 for row in rows}
    freeze = r.build_freeze(m, lens, cfg)
    backend = tt.SleepyBackend({row.example_id: row.label for row in rows})
    texts = {row.example_id: ("doc", "sentence") for row in rows}
    result = r.run_pilot(cfg, m, freeze, lens, texts, backend, require_permission=False)
    return m, cfg, result


def test_the_report_is_reproducible_from_the_saved_predictions_alone():
    m, cfg, result = completed_pilot()
    pred = result["predictions"]
    assert {"rows", "thresholds", "train_sample_example_ids", "group_split", "hashes"} <= set(pred)
    assert {row["split"] for row in pred["rows"]} == {"train_select", "validation"}
    assert set(pred["group_split"].values()) == {"train_fit", "train_select", "validation"}
    assert pred["hashes"]["manifest"] == m["manifest_sha256"] and pred["hashes"]["config"] == cfg.config_hash()
    rec = fz.recompute_report(pred, m, cfg)
    for name in ("baseline", "tuned"):
        assert rec[name]["threshold"] == pred["thresholds"][name] == result[name]["threshold_selected_on_train_select"]
        for key in ("balanced_accuracy", "unsupported_recall", "kappa", "pr_auc_unsupported", "tp", "fp", "fn", "tn"):
            assert rec[name]["sentence"][key] == pytest.approx(result[name]["sentence"][key], nan_ok=True)
        assert rec[name]["response"]["n_complete"] == result[name]["response"]["n_complete"]
    assert rec["paired_gain_bootstrap"]["lower95"] == pytest.approx(result["paired_gain_bootstrap"]["lower95"])
    assert rec["continue_decision"]["continue"] == result["continue_decision"]["continue"]


def test_recomputation_exposes_a_doctored_score():
    m, cfg, result = completed_pilot()
    pred = json.loads(json.dumps(result["predictions"]))
    for row in pred["rows"]:
        if row["split"] == "validation":
            row["tuned_score"] = 1.0 - row["tuned_score"]  # invert every tuned score
    rec = fz.recompute_report(pred, m, cfg)
    assert rec["tuned"]["sentence"]["balanced_accuracy"] != pytest.approx(result["tuned"]["sentence"]["balanced_accuracy"])
    scores = np.array([row["tuned_score"] for row in result["predictions"]["rows"]])
    assert scores.min() >= 0 and scores.max() <= 1

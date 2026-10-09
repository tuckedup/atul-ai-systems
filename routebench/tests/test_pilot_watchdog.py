"""The external watchdog against REAL subprocesses, including a backend that never calls `tick`."""
import json
import os
import sys
import time
from pathlib import Path

import evalops
import pytest
from evalops.pilot import _testing as tt
from evalops.pilot.watchdog import ProgressFile, read_progress, supervise

ROOT = str(Path(evalops.__file__).parents[1])


def env():
    return {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}


def run_child(tmp_path, mode, **caps):
    m = tt.make_manifest()
    cfg = tt.make_cfg(m, **caps)
    argv = [sys.executable, "-m", "evalops.pilot._testing", mode, str(tmp_path), json.dumps(caps)]
    started = time.monotonic()
    out = supervise(argv, cfg=cfg, progress_path=tmp_path / "progress.json", env=env(), log_path=tmp_path / "child.log", poll=0.1)
    return out, time.monotonic() - started


def test_progress_file_is_atomic_and_sequenced(tmp_path):
    p = ProgressFile(tmp_path / "sub" / "progress.json")
    p.event("training", "start")
    p.event("training", "end")
    assert read_progress(tmp_path / "sub" / "progress.json") == {"seq": 2, "phase": "training", "event": "end"}
    assert read_progress(tmp_path / "missing.json") is None


def test_a_cooperating_child_finishes_and_is_not_killed(tmp_path):
    out, _ = run_child(tmp_path, "ok", startup=120, grace=1)
    assert out["killed"] is False and out["exit_code"] == 0
    result = json.loads((tmp_path / "pilot_result_v1.json").read_text(encoding="utf8"))
    assert result["status"] == "COMPLETE" and (tmp_path / "adapter" / "adapter_model.safetensors").exists()


def test_a_backend_that_never_ticks_is_killed_during_the_feasibility_phase(tmp_path):
    phases = {"feasibility": 2, "baseline_eval": 200, "training": 400, "tuned_eval": 200, "slack": 140}
    out, took = run_child(tmp_path, "hang_probe", phases=phases, startup=120, grace=1, total=500)
    assert out["killed"] is True and out["phase"] == "feasibility" and "feasibility" in out["reason"]
    assert took < 40, "the hung worker must be stopped promptly"
    assert not (tmp_path / "pilot_result_v1.json").exists()


def test_a_training_phase_that_never_ticks_is_killed_at_its_cap(tmp_path):
    phases = {"feasibility": 100, "baseline_eval": 100, "training": 2, "tuned_eval": 100, "slack": 100}
    out, took = run_child(tmp_path, "hang_train", phases=phases, startup=120, grace=1, total=500)
    assert out["killed"] is True and out["phase"] == "training" and took < 60


def test_no_phase_started_within_the_startup_cap_is_killed(tmp_path):
    out, took = run_child(tmp_path, "hang_startup", startup=2, grace=1, total=500)
    assert out["killed"] is True and out["phase"] is None and "startup" in out["reason"] and took < 30


def test_the_total_cap_kills_even_when_the_phase_cap_is_larger(tmp_path):
    phases = {"feasibility": 500, "baseline_eval": 500, "training": 500, "tuned_eval": 500, "slack": 500}
    out, took = run_child(tmp_path, "hang_probe", phases=phases, startup=120, grace=1, total=4)
    assert out["killed"] is True and "total cap" in out["reason"] and took < 40


def test_a_fast_child_is_never_reported_as_killed(tmp_path):
    out = supervise([sys.executable, "-c", "print('hi')"], cfg=tt.make_cfg(tt.make_manifest()),
                    progress_path=tmp_path / "p.json", poll=0.05)
    assert out["killed"] is False and out["exit_code"] == 0


@pytest.mark.parametrize("bad", ["hang_baseline_score", "hang_tuned_score"])
def test_hanging_during_scoring_phases_is_also_killed(tmp_path, bad):
    phases = {"feasibility": 100, "baseline_eval": 2, "training": 100, "tuned_eval": 2, "slack": 100}
    out, _ = run_child(tmp_path, bad, phases=phases, startup=120, grace=1, total=500)
    assert out["killed"] is True and out["phase"] == ("baseline_eval" if "baseline" in bad else "tuned_eval")


def test_progress_reporting_survives_a_reader_holding_the_file_and_never_crashes_the_run(tmp_path, monkeypatch):
    import evalops.pilot.watchdog as wd

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError("file in use by the watchdog (simulated)")
        return real_replace(src, dst)

    monkeypatch.setattr(wd.os, "replace", flaky_replace)
    p = ProgressFile(tmp_path / "progress.json")
    p.event("training", "start")  # retries, then succeeds
    assert read_progress(tmp_path / "progress.json")["phase"] == "training" and calls["n"] == 4

    def always_locked(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(wd.os, "replace", always_locked)
    q = ProgressFile(tmp_path / "other.json")
    q.event("training", "start")  # must not raise: reporting is best-effort
    assert q.failures == 1

"""`execute` end to end with a synthetic worker, plus input re-derivation checks. No model, no GPU."""
import json
import os
import sys
from pathlib import Path

import evalops
import pytest
from evalops.pilot import _testing as tt
from evalops.pilot import cli
from evalops.pilot import harness as h
from evalops.pilot import ingest_audit as ia
from evalops.pilot import run as r
from evalops.pilot.inputs import load_texts

ROOT = str(Path(evalops.__file__).parents[1])
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")


def child(tmp_path, mode, caps):
    return [sys.executable, "-m", "evalops.pilot._testing", mode, str(tmp_path / cli.RESULT_DIR), json.dumps(caps)]


def test_execute_refuses_without_the_confirmation_token(tmp_path):
    cfg, *_ = tt.write_inputs(tmp_path)
    with pytest.raises(SystemExit, match="--confirm"):
        cli.execute(tmp_path, tmp_path / "token_lens.json", tmp_path, python=None, confirm="yes", cfg=cfg, require_permission=False)
    assert not (tmp_path / cli.RESULT_DIR).exists()


def test_execute_refuses_without_a_recorded_permission_decision(tmp_path):
    cfg, *_ = tt.write_inputs(tmp_path)
    doc = tmp_path / "perm.md"
    doc.write_text("CNN/DM rows for a private research pilot: ______ (accept / decline)\n", encoding="utf8")
    with pytest.raises(r.PermissionNotRecorded):
        cli.execute(tmp_path, tmp_path / "token_lens.json", tmp_path, python=None, confirm=cli.CONFIRM_TOKEN, cfg=cfg,
                    permission_record=doc)
    assert not (tmp_path / cli.RESULT_DIR).exists()


def test_execute_needs_an_explicit_existing_interpreter(tmp_path):
    cfg, *_ = tt.write_inputs(tmp_path)
    for py in (None, tmp_path / "no-such-python"):
        with pytest.raises(SystemExit, match="--python"):
            cli.execute(tmp_path, tmp_path / "token_lens.json", tmp_path, python=py, confirm=cli.CONFIRM_TOKEN, cfg=cfg,
                        require_permission=False)


def test_execute_verifies_manifest_configuration_and_freeze_before_spawning(tmp_path):
    cfg, m, _freeze = tt.write_inputs(tmp_path)
    bad = tmp_path / "v5" / "training_manifest_v5.json"
    bad.write_text(json.dumps({**m, "rows": m["rows"][:-1]}), encoding="utf8")
    with pytest.raises(h.ManifestMismatch):
        cli.execute(tmp_path, tmp_path / "token_lens.json", tmp_path, python=Path(sys.executable), confirm=cli.CONFIRM_TOKEN,
                    cfg=cfg, require_permission=False)
    assert not (tmp_path / cli.RESULT_DIR).exists()


def test_a_completed_run_persists_result_progress_log_and_artifacts(tmp_path):
    caps = {"startup": 120, "grace": 1}
    cfg, *_ = tt.write_inputs(tmp_path, caps)
    out = cli.execute(tmp_path, tmp_path / "token_lens.json", tmp_path, python=None, confirm=cli.CONFIRM_TOKEN, cfg=cfg,
                      require_permission=False, child_argv=child(tmp_path, "ok", caps), poll=0.1)
    d = tmp_path / cli.RESULT_DIR
    assert out["killed"] is False
    result = json.loads((d / "pilot_result_v1.json").read_text(encoding="utf8"))
    assert result["status"] == "COMPLETE" and "continue_decision" in result
    assert (d / "progress.json").exists() and (d / "child.log").exists() and (d / "adapter" / "adapter_model.safetensors").exists()


def test_a_killed_worker_still_leaves_an_incomplete_result_and_a_second_run_is_refused(tmp_path):
    caps = {"phases": {"feasibility": 2, "baseline_eval": 100, "training": 100, "tuned_eval": 100, "slack": 100},
            "startup": 120, "grace": 1, "total": 500}
    cfg, *_ = tt.write_inputs(tmp_path, caps)
    out = cli.execute(tmp_path, tmp_path / "token_lens.json", tmp_path, python=None, confirm=cli.CONFIRM_TOKEN, cfg=cfg,
                      require_permission=False, child_argv=child(tmp_path, "hang_probe", caps), poll=0.1)
    result = json.loads((tmp_path / cli.RESULT_DIR / "pilot_result_v1.json").read_text(encoding="utf8"))
    assert out["killed"] is True
    assert result["status"] == "INCOMPLETE" and result["incomplete_phase"] == "feasibility" and result["continue_decision"] is None
    assert result["watchdog"]["killed"] is True and "baseline" not in result
    with pytest.raises(SystemExit, match="already exists"):
        cli.execute(tmp_path, tmp_path / "token_lens.json", tmp_path, python=None, confirm=cli.CONFIRM_TOKEN, cfg=cfg,
                    require_permission=False, child_argv=child(tmp_path, "ok", caps))


def test_a_worker_that_crashes_without_a_result_is_recorded_incomplete(tmp_path):
    cfg, *_ = tt.write_inputs(tmp_path, {"startup": 120, "grace": 1})
    out = cli.execute(tmp_path, tmp_path / "token_lens.json", tmp_path, python=None, confirm=cli.CONFIRM_TOKEN, cfg=cfg,
                      require_permission=False, child_argv=[sys.executable, "-c", "raise SystemExit(7)"], poll=0.05)
    result = json.loads((tmp_path / cli.RESULT_DIR / "pilot_result_v1.json").read_text(encoding="utf8"))
    assert out["exit_code"] == 7 and result["status"] == "FAILED" and "code 7" in result["incomplete_reason"]
    assert out["verdict"] == "FAILED" and out["cli_exit_code"] == 5


def test_child_main_runs_the_full_path_with_an_injected_backend(tmp_path, monkeypatch):
    cfg, m, _freeze = tt.write_inputs(tmp_path, {"startup": 120, "grace": 1})
    monkeypatch.setattr(cli, "load_texts", lambda d, man, ids: {i: ("doc", "sentence") for i in ids})
    labels = {row.example_id: row.label for row in h.rows_from_manifest(m)}
    out_dir = tmp_path / "direct"
    out_dir.mkdir()
    res = cli.child_main(tmp_path, tmp_path / "token_lens.json", tmp_path, out_dir, cfg=cfg, require_permission=False,
                         backend_factory=lambda c, d: tt.SleepyBackend(labels))
    assert res["status"] == "COMPLETE" and "python" in res["environment"]
    assert json.loads((out_dir / "pilot_result_v1.json").read_text(encoding="utf8"))["status"] == "COMPLETE"
    assert (out_dir / "adapter" / "adapter_model.safetensors").exists() and (out_dir / "progress.json").exists()


# ---- input re-derivation --------------------------------------------------------------------------------------------------------
DOC1 = ("The city council met on Monday and voted on the annual budget after a long debate about road repairs. "
        "Dr. Smith presented the figures. The library will receive new funding next year.")
DOC2 = ("A marine biologist described how coral reefs recover after bleaching events and why water temperature remains the "
        "dominant factor. Slow growing species need decades to return.")
R1 = "The council voted on the budget. Dr. Smith presented the figures. The library gets a new roof."
R2 = "Coral reefs recover slowly. Temperature is the dominant factor."


def rag_files(tmp_path):
    def src(sid, text):
        return {"source_id": sid, "task_type": "Summary", "source": "CNN/DM", "source_info": text, "prompt": "p"}

    def resp(rid, sid, text, labels=()):
        return {"id": rid, "source_id": sid, "model": "m", "temperature": 0.5, "labels": list(labels), "split": "train",
                "quality": "good", "response": text}

    span = {"start": R1.index("a new roof"), "end": R1.index("a new roof") + 10, "text": "a new roof",
            "label_type": "Evident Baseless Info", "meta": ""}
    (tmp_path / "response.jsonl").write_text("\n".join(json.dumps(x) for x in [resp("a", "s1", R1, [span]), resp("b", "s2", R2)]),
                                             encoding="utf8")
    (tmp_path / "source_info.jsonl").write_text("\n".join(json.dumps(x) for x in [src("s1", DOC1), src("s2", DOC2)]), encoding="utf8")
    report = ia.run_audit(tmp_path, {})
    return report["_manifest"]


def test_texts_are_rederived_and_must_match_the_manifest_exactly(tmp_path):
    m = rag_files(tmp_path)
    ids = [row[0] for row in m["rows"]]
    texts = load_texts(tmp_path, m, ids)
    assert set(texts) == set(ids) and all(doc and sent for doc, sent in texts.values())
    assert any("Dr. Smith presented the figures." == s for _, s in texts.values())


def test_changed_files_labels_or_unknown_ids_are_refused(tmp_path):
    m = rag_files(tmp_path)
    ids = [row[0] for row in m["rows"]]
    cols = m["columns"]
    flipped = json.loads(json.dumps(m))
    flipped["rows"][0][cols.index("label")] = 1 - flipped["rows"][0][cols.index("label")]
    with pytest.raises(h.ManifestMismatch, match="differs"):
        load_texts(tmp_path, flipped, ids)
    with pytest.raises(h.ManifestMismatch, match="cannot be re-derived"):
        load_texts(tmp_path, m, ids + ["zzz#0"])
    (tmp_path / "source_info.jsonl").write_text((tmp_path / "source_info.jsonl").read_text(encoding="utf8") + "\n", encoding="utf8")
    with pytest.raises(h.ManifestMismatch, match="hash"):
        load_texts(tmp_path, m, ids)


# ---- cumulative compute ledger -------------------------------------------------------------------------------------------------
def test_each_execute_appends_to_the_ledger_and_the_ceiling_is_enforced(tmp_path):
    from evalops.pilot import ledger as lg

    caps = {"startup": 120, "grace": 1}
    cfg, *_ = tt.write_inputs(tmp_path, caps)
    led = tmp_path / lg.LEDGER_NAME
    lg.append(led, kind="smoke", wall_seconds=8)
    out = cli.execute(tmp_path, tmp_path / "token_lens.json", tmp_path, python=None, confirm=cli.CONFIRM_TOKEN, cfg=cfg,
                      require_permission=False, child_argv=child(tmp_path, "ok", caps), poll=0.1)
    entries = lg.read(led)
    assert [e["kind"] for e in entries] == ["smoke", "pilot_execute"] and entries[1]["verdict"] == "VALID_COMPLETE"
    assert out["ledger_total_hours"] == pytest.approx(lg.total_seconds(led) / 3600, abs=1e-3)
    # the ceiling counts everything already spent plus the run's own total cap
    lg.append(led, kind="earlier", wall_seconds=lg.CEILING_SECONDS - lg.total_seconds(led) - cfg.total_seconds + 1)
    (tmp_path / cli.RESULT_DIR).rename(tmp_path / "earlier_run")
    with pytest.raises(lg.LedgerCeilingExceeded, match="ceiling"):
        cli.execute(tmp_path, tmp_path / "token_lens.json", tmp_path, python=None, confirm=cli.CONFIRM_TOKEN, cfg=cfg,
                    require_permission=False, child_argv=child(tmp_path, "ok", caps), poll=0.1)
    assert not (tmp_path / cli.RESULT_DIR).exists()


def test_any_recorded_api_spend_blocks_further_runs(tmp_path):
    from evalops.pilot import ledger as lg

    led = tmp_path / lg.LEDGER_NAME
    lg.append(led, kind="oops", wall_seconds=1)
    with led.open("a", encoding="utf8") as fh:
        fh.write(json.dumps({"kind": "api", "wall_seconds": 0, "api_usd": 0.5}) + "\n")
    with pytest.raises(lg.LedgerCeilingExceeded, match="API spend"):
        lg.check_budget(led, 10)

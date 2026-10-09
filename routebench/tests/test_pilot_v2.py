"""Corrected pilot (v2): stratified probe, label-blind response sampling, 80% / overhead rules, checkpoints, heartbeats, config identity."""
import dataclasses
import json
import os
import sys
from pathlib import Path

import evalops
import numpy as np
import pytest
from evalops.pilot import _testing as tt
from evalops.pilot import cli, config
from evalops.pilot import finalize as fz
from evalops.pilot import harness as h
from evalops.pilot import run as r

ROOT = str(Path(evalops.__file__).parents[1])
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
DATA = Path(__file__).parents[2] / ".local" / "ragtruth-ingest-20260930"


# ---- configuration identity -------------------------------------------------------------------------------------------------------
def test_v1_identity_is_unchanged_and_v2_is_a_distinct_registered_configuration():
    assert config.PILOT.config_hash().startswith("fe8c51a0")  # the failed first pilot stays reproducible
    v2 = config.PILOT_V2
    assert v2.config_hash() != config.PILOT.config_hash() and v2.result_dir != getattr(config.PILOT, "result_dir", "pilot_run_v1")
    assert v2.result_dir == "pilot_run_v2" and v2.freeze_file.endswith("pilot_freeze_v4.json")
    assert set(config.CONFIGS) == {"v1", "v2"}
    assert sum(v2.phase_seconds.values()) + v2.startup_seconds == v2.total_seconds == 7200  # two additional hours
    assert v2.feasibility_fraction == v2.overall_fraction == 0.8 and v2.probe_mode == "stratified"
    # everything that defines labels, scoring and success is unchanged
    for field in ("unsupported_token_id", "supported_token_id", "lora_r", "lora_alpha", "learning_rate", "epochs", "min_improvement",
                  "max_recall_drop", "threshold_grid", "max_input_tokens", "response_aggregation", "base_revision"):
        assert getattr(v2, field) == getattr(config.PILOT, field), field


def test_execute_and_child_require_an_explicit_config():
    with pytest.raises(SystemExit, match="--config"):
        cli.main(["execute", "d", "l", "m"])
    with pytest.raises(SystemExit, match="unknown configuration"):
        cli.main(["freeze", "d", "l", "--config", "v9"])


# ---- stratified probe ---------------------------------------------------------------------------------------------------------------
def _examples(n, length_of):
    return [(f"e{i:04d}", "d" * length_of(i), "s" * 20) for i in range(n)]


def test_probe_plan_spreads_over_lengths_uses_production_batches_and_includes_the_longest_inputs():
    m = tt.make_manifest()
    cfg = tt.make_cfg_v2(m)
    sc = _examples(400, lambda i: 400 + (i * 37) % 3000)
    lens = {e[0]: len(e[1]) + len(e[2]) for e in sc}
    tr = [(e[0], e[1], e[2], i % 2) for i, e in enumerate(_examples(120, lambda i: 300 + (i * 53) % 4000))]
    lens.update({e[0]: len(e[1]) + len(e[2]) for e in tr})
    plan = h.probe_plan(sc, tr, lens, cfg)
    production = h.score_batches_for(sc)
    assert plan["n_score_batches_total"] == len(production) == 100
    assert all(b in production for b in plan["score_batches"])  # exactly production batches, not ad-hoc ones
    sizes = [sum(lens[e[0]] for e in b) for b in plan["score_batches"]]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]  # spread from short to long
    heaviest = max(sum(lens[e[0]] for e in b) for b in production)
    assert sum(lens[e[0]] for e in plan["worst_score_batch"]) == heaviest
    need = cfg.micro_batch * cfg.grad_accum * cfg.probe_optimizer_updates
    assert len(plan["train_examples"]) == need == 32
    tlens = sorted(lens[e[0]] for e in plan["train_examples"])
    assert tlens[0] < 0.25 * max(lens[e[0]] for e in tr) and tlens[-1] > 0.85 * max(lens[e[0]] for e in tr)
    longest = sorted(tr, key=lambda e: lens[e[0]])[-cfg.micro_batch:]
    assert plan["worst_train_batch"] == longest
    assert plan == h.probe_plan(sc, tr, lens, cfg)  # deterministic


def test_the_old_probe_bias_is_what_the_new_plan_removes():
    """The first pilot scored the first 16 sorted ids (short inputs). The stratified mean length tracks the production mean."""
    m = tt.make_manifest()
    cfg = tt.make_cfg_v2(m)
    sc = _examples(800, lambda i: 300 + (i * 97) % 2600)
    lens = {e[0]: len(e[1]) + len(e[2]) for e in sc}
    tr = [(e[0], e[1], e[2], 1) for e in sc[:100]]
    plan = h.probe_plan(sc, tr, lens, cfg)
    production_mean = np.mean(list(lens.values()))
    new = np.mean([lens[e[0]] for b in plan["score_batches"] for e in b])
    old = np.mean([lens[e[0]] for e in sc[:16]])
    assert abs(new - production_mean) < 0.2 * production_mean
    assert abs(old - production_mean) > abs(new - production_mean)


# ---- label-blind response-level sampling -------------------------------------------------------------------------------------------------
def test_response_sampling_keeps_the_coverage_groups_and_whole_responses_and_ignores_labels():
    m = tt.make_manifest()
    cfg = tt.make_cfg_v2(m)
    rows = h.rows_from_manifest(m)
    plan = h.make_plan(rows, cfg)
    lens = {row.example_id: 800 for row in rows}
    cover = h.evaluation_sample(plan.validation, lens, cfg.validation_eval_sentences, cfg, "validation")
    a = h.evaluation_sample_by_response(plan.validation, lens, cfg.validation_eval_sentences, 1, cfg, "validation")
    assert a["groups"] == cover["groups"] and set(a["example_ids"]) < set(cover["example_ids"])
    by_resp = {}
    for row in plan.validation:
        by_resp.setdefault(row.response_id, set()).add(row.example_id)
    kept = set(a["example_ids"])
    assert all(ids <= kept or not ids & kept for ids in by_resp.values())  # a response is kept whole or not at all
    assert a["responses"] == len(a["groups"])
    flipped = [dataclasses.replace(row, label=1 - row.label) for row in plan.validation]
    assert h.evaluation_sample_by_response(flipped, lens, cfg.validation_eval_sentences, 1, cfg, "validation") == a
    full = h.evaluation_sample_by_response(plan.validation, lens, cfg.validation_eval_sentences, 99, cfg, "validation")
    assert full["example_ids"] == cover["example_ids"]  # a large cap preserves the coverage sample in full


@pytest.mark.skipif(not (DATA / "v5" / "pilot_freeze_v3.json").exists(), reason="local RAGTruth artifacts not present")
def test_v2_preserves_the_full_validation_sample_of_the_first_pilot_and_shrinks_only_selection_and_training():
    m = json.loads((DATA / "v5" / "training_manifest_v5.json").read_text(encoding="utf8"))
    lens = json.loads((Path(__file__).parents[2] / ".local" / "token_lens.json").read_text(encoding="utf8"))
    old = json.loads((DATA / "v5" / "pilot_freeze_v3.json").read_text(encoding="utf8"))
    new = r.build_freeze(m, lens, config.PILOT_V2)
    assert new["validation_eval"]["example_ids"] == old["validation_eval"]["example_ids"]  # 45 groups, every sampled sentence
    assert new["validation_eval"]["groups"] == old["validation_eval"]["groups"]
    assert set(new["select_eval"]["example_ids"]) < set(old["select_eval"]["example_ids"])
    assert new["select_eval"]["groups"] == old["select_eval"]["groups"]  # the same 24 selection documents
    assert len(new["train_sample"]["example_ids"]) < len(old["train_sample"]["example_ids"])
    assert new["config_hash"] == config.PILOT_V2.config_hash() and new["manifest_sha256"] == old["manifest_sha256"]


# ---- the 80% and overhead rules ----------------------------------------------------------------------------------------------------------
def _v2_setup(**backend_kw):
    m = tt.make_manifest()
    cfg = tt.make_cfg_v2(m, phases={"feasibility": 60, "baseline_eval": 100, "training": 100, "tuned_eval": 100, "slack": 100})
    rows = h.rows_from_manifest(m)
    lens = {row.example_id: 800 for row in rows}
    freeze = r.build_freeze(m, lens, cfg)
    texts = {row.example_id: ("doc", "sentence") for row in rows}

    class Clock:
        t = 0.0

        def __call__(self):
            return self.t

    clock = Clock()

    class Backend(tt.SleepyBackend):
        def __init__(self):
            super().__init__({row.example_id: row.label for row in rows})
            self.kw = {"score_s_per_example": 0.001, "train_s_per_example": 0.001, "save_verify_seconds": 0.01, **backend_kw}
            self.plan = None

        def probe_v2(self, plan):
            self.plan = plan
            clock.t += self.kw.pop("probe_clock", 0)
            return {"optimizer_updates_done": 2, "peak_memory_gb": 1.0, "worst_case_peak_memory_gb": 1.2, **self.kw}

    return m, cfg, lens, freeze, texts, clock, Backend()


def _go(m, cfg, lens, freeze, texts, clock, backend, **kw):
    return r.run_pilot(cfg, m, freeze, lens, texts, backend, clock=clock, require_permission=False, **kw)


def test_a_projection_under_100_but_over_80_percent_is_now_not_feasible():
    n_eval = None
    m, cfg, lens, freeze, texts, clock, backend = _v2_setup()
    n_eval = len(freeze["select_eval"]["example_ids"]) + len(freeze["validation_eval"]["example_ids"])
    cap = cfg.phase_seconds["baseline_eval"]
    per_ex_90 = 0.9 * cap / n_eval  # 90% of the cap: fits the OLD rule, violates the 80% rule
    m, cfg, lens, freeze, texts, clock, backend = _v2_setup(score_s_per_example=per_ex_90)
    res = _go(m, cfg, lens, freeze, texts, clock, backend)
    assert res["status"] == "NOT_FEASIBLE" and res["feasibility"]["fits_fraction"]["baseline_eval"] is False
    assert res["feasibility"]["projected_seconds"]["baseline_eval"] < cap  # would have passed under a 100% rule
    per_ex_70 = 0.7 * cap / n_eval
    m, cfg, lens, freeze, texts, clock, backend = _v2_setup(score_s_per_example=per_ex_70, train_s_per_example=0.001)
    assert _go(m, cfg, lens, freeze, texts, clock, backend)["status"] == "COMPLETE"


def test_measured_loading_probe_and_saving_overhead_count_against_the_total():
    m, cfg, lens, freeze, texts, clock, backend = _v2_setup()
    ok = _go(m, cfg, lens, freeze, texts, clock, backend, startup_seconds_measured=10)
    assert ok["status"] == "COMPLETE" and ok["feasibility"]["overhead"]["startup_seconds_measured"] == 10
    limit = 0.8 * cfg.total_seconds
    m, cfg, lens, freeze, texts, clock, backend = _v2_setup()
    bad = _go(m, cfg, lens, freeze, texts, clock, backend, startup_seconds_measured=limit)  # loading alone eats the allowance
    assert bad["status"] == "NOT_FEASIBLE" and bad["feasibility"]["projected_total_seconds"] > limit
    m, cfg, lens, freeze, texts, clock, backend = _v2_setup(save_verify_seconds=limit)  # a slow save is tripled into the estimate
    assert _go(m, cfg, lens, freeze, texts, clock, backend)["status"] == "NOT_FEASIBLE"


def test_the_probe_is_handed_the_stratified_plan_not_the_first_examples():
    m, cfg, lens, freeze, texts, clock, backend = _v2_setup()
    _go(m, cfg, lens, freeze, texts, clock, backend)
    assert backend.plan is not None and set(backend.plan) >= {"score_batches", "worst_score_batch", "train_examples", "worst_train_batch"}
    assert len(backend.plan["train_examples"]) == 32


# ---- progress heartbeat and durable partial scores ---------------------------------------------------------------------------------------------
def test_progress_log_records_counts_rate_ram_and_vram_durably(tmp_path):
    t = {"now": 0.0}
    log = h.ProgressLog(tmp_path / "p.jsonl", clock=lambda: t["now"], vram_fn=lambda: {"vram_peak_gb": 3.5})
    t["now"] = 10.0
    log.log("baseline_eval", 40, 400)
    rec = json.loads((tmp_path / "p.jsonl").read_text(encoding="utf8").splitlines()[0])
    assert rec["phase"] == "baseline_eval" and rec["done"] == 40 and rec["total"] == 400 and rec["rate_per_s"] == 4.0
    assert rec["t"] == 10.0 and rec["vram_peak_gb"] == 3.5 and {"ram_used_gb", "ram_available_gb", "process_rss_gb"} <= set(rec)
    boom = h.ProgressLog(tmp_path / "q.jsonl", vram_fn=lambda: 1 / 0)  # monitoring must never crash a run
    boom.log("x", 1, 2)


def test_checkpointer_appends_and_counts(tmp_path):
    ck = h.Checkpointer(tmp_path, "baseline_eval")
    ck.write_many({"a": 0.1, "b": 0.9})
    ck.write_many({"c": 0.5})
    lines = [json.loads(x) for x in ck.path.read_text(encoding="utf8").splitlines()]
    assert ck.count == 3 and [x["example_id"] for x in lines] == ["a", "b", "c"]


def run_mode_v2(tmp_path, mode, caps):
    caps = {**caps, "v2": True}
    cfg, *_ = tt.write_inputs_v2(tmp_path, {k: v for k, v in caps.items() if k != "v2"})
    argv = [sys.executable, "-m", "evalops.pilot._testing", mode, str(tmp_path / cfg.result_dir), json.dumps(caps)]
    out = cli.execute(tmp_path, tmp_path / "token_lens.json", tmp_path, python=None, confirm=cli.CONFIRM_TOKEN, cfg=cfg,
                      require_permission=False, child_argv=argv, poll=0.1)
    return out, tmp_path / cfg.result_dir


def test_a_hang_mid_scoring_keeps_durable_partial_scores_but_stays_incomplete(tmp_path):
    caps = {"phases": {"feasibility": 100, "baseline_eval": 4, "training": 100, "tuned_eval": 100, "slack": 100}, "startup": 120,
            "grace": 1, "total": 500}
    out, d = run_mode_v2(tmp_path, "hang_midscore", caps)
    assert out["killed"] is True and out["phase"] == "baseline_eval" and out["cli_exit_code"] == 3
    partial = d / "partial_scores_baseline_eval.jsonl"
    rows = [json.loads(x) for x in partial.read_text(encoding="utf8").splitlines()]
    assert len(rows) >= 10 and all(0 <= x["score"] <= 1 for x in rows)
    result = json.loads((d / fz.RESULT_NAME).read_text(encoding="utf8"))
    assert result["status"] == "INCOMPLETE" and result["continue_decision"] is None and "baseline" not in result
    assert (d / "progress.json").exists()


def test_a_v2_run_completes_and_writes_no_continue_decision_from_partial_data(tmp_path):
    out, d = run_mode_v2(tmp_path, "ok", {"startup": 120, "grace": 1})
    assert out["verdict"] == "VALID_COMPLETE" and (d / "partial_scores_baseline_eval.jsonl").exists()
    result = json.loads((d / fz.RESULT_NAME).read_text(encoding="utf8"))
    assert result["config_hash"] == tt.make_cfg_v2(tt.make_manifest(), startup=120, grace=1).config_hash()
    assert result["feasibility"]["fraction_rule"] == 0.8 and "overhead" in result["feasibility"]

"""Tests for reconstructing a judge from the experiment record rather than from `GRID`.

The defect these pin was live on the committed tree. `make calibrate-freeze` reported

    calibration error: threshold selection needs a usable dev set; got 0 paired items.

with 2,968 usable judgments in the cache. `cmd_freeze` looked the chosen variant up in `GRID` and
used the reconstructed `config_hash` as the cache key. `config_hash` covers every `JudgeConfig`
field AND the prompt scaffolding, so adding a field or editing a template after a run -- which is
what declaring the v8-v11 variants did -- renames every judge in the record. Every cache lookup
missed, `pooled` came back empty, and the error blamed the dev set.

`cmd_test` already had the right rule ("reconstruct the judge the BUNDLE names, not whatever GRID
currently defines"). These tests hold `freeze` to the same rule one step earlier, and require the
failure to name the drift instead of blaming the data.
"""
from __future__ import annotations

import pytest
from evalops.calibrate import CalibrationError
from evalops.judge import JudgeConfig, prompt_template_hash
from evalops.run_calibration import (
    ENSEMBLE_CANDIDATES,
    GRID,
    MATCHED_PAIRS,
    _recorded_config,
    verify_record,
)


def _record(variant="v-x", *, config=None, in_summaries=False, **row_extra):
    cfg = config or JudgeConfig(variant_id=variant, model="gpt-4o-mini", mode="rubric")
    saved = cfg.as_dict()
    row = {"variant_id": variant, "config_hash": saved["config_hash"],
           "best_dev_kappa": 0.5, "completion": 1.0}
    row.update(row_extra)
    experiments = {"variants": [row], "run_summaries": []}
    if in_summaries:
        experiments["run_summaries"] = [{"config": saved}]
    else:
        row["judge_config"] = saved
    return row, experiments


def test_recorded_config_rebuilds_from_the_row():
    row, experiments = _record()
    cfg = _recorded_config(row, experiments)
    assert cfg.variant_id == "v-x"
    assert cfg.config_hash == row["config_hash"]


def test_recorded_config_falls_back_to_run_summaries_for_older_records():
    # Records written before `judge_config` was added to the variant rows still carry the full
    # config under `run_summaries`.
    row, experiments = _record(in_summaries=True)
    assert "judge_config" not in row
    assert _recorded_config(row, experiments).variant_id == "v-x"


def test_recorded_config_refuses_rather_than_falling_back_to_grid():
    # A silent fallback to GRID is exactly how the drift went unnoticed.
    live = GRID[0]
    row = {"variant_id": live.variant_id, "config_hash": live.config_hash}
    with pytest.raises(CalibrationError, match="carries no judge_config"):
        _recorded_config(row, {"variants": [row], "run_summaries": []})


def test_a_hash_that_does_not_reproduce_names_the_prompt_scaffolding():
    # Simulates the real failure: the stored fields round-trip exactly, but the recorded hash was
    # produced under different prompt templates.
    row, experiments = _record()
    row["judge_config"] = dict(row["judge_config"], config_hash="a-hash-from-older-prompts")
    row["config_hash"] = "a-hash-from-older-prompts"
    row["prompt_template_hash"] = "older-templates"
    with pytest.raises(CalibrationError) as e:
        _recorded_config(row, experiments)
    message = str(e.value)
    assert "cannot be reconstructed" in message
    assert "prompt scaffolding" in message
    assert "older-templates" in message
    assert prompt_template_hash("rubric") in message
    # It must not suggest a workaround that would freeze an unreproducible judge.
    assert "has to be re-judged" in message


def test_grid_drift_is_reported_but_does_not_block(capsys):
    # The record is self-consistent; GRID has simply moved on. Freezing the RECORDED judge is
    # correct, and the divergence is worth a note rather than a failure.
    live = GRID[0]
    drifted = JudgeConfig(
        variant_id=live.variant_id, model=live.model, mode=live.mode,
        include_reference=not live.include_reference,
    )
    row, experiments = _record(live.variant_id, config=drifted)
    cfg = _recorded_config(row, experiments)
    assert cfg.config_hash == drifted.config_hash
    assert "GRID" in capsys.readouterr().out


# ---------------------------------------------------------------- verify_record


def test_verify_record_marks_a_good_record_reproducible():
    row, experiments = _record()
    (result,) = verify_record(experiments)
    assert result["reproducible"] is True
    assert result["rebuilt_config_hash"] == row["config_hash"]


def test_verify_record_marks_a_drifted_record_unreproducible():
    row, experiments = _record()
    row["judge_config"] = dict(row["judge_config"], config_hash="stale")
    row["config_hash"] = "stale"
    (result,) = verify_record(experiments)
    assert result["reproducible"] is False
    assert "cannot be reconstructed" in result["reason"]
    assert result["recorded_dev_kappa"] == 0.5


def test_verify_record_passes_skipped_variants_through():
    experiments = {"variants": [{"variant_id": "v-gpt5", "skipped": "HTTP 404"}],
                   "run_summaries": []}
    (result,) = verify_record(experiments)
    assert result["skipped"] == "HTTP 404"
    assert "reproducible" not in result


def test_verify_record_flags_grid_divergence_on_a_reproducible_row():
    live = GRID[0]
    drifted = JudgeConfig(variant_id=live.variant_id, model=live.model, mode=live.mode,
                          max_tokens=live.max_tokens + 100)
    _, experiments = _record(live.variant_id, config=drifted)
    (result,) = verify_record(experiments)
    assert result["reproducible"] is True
    assert result["grid_matches_record"] is False


def test_the_committed_dev_record_is_checked_by_this_command():
    # A live assertion about this checkout: the recorded dev experiment cannot be reproduced from
    # the current prompts, so the kappas in `dev_experiments.json` describe judges that no longer
    # exist and the dev split has to be re-judged. If a future re-run fixes that, this assertion
    # should be inverted deliberately rather than silently stop meaning anything.
    import json
    from pathlib import Path

    path = Path(__file__).parent.parent / "evalops" / "data" / "dev_experiments.json"
    if not path.exists():
        pytest.skip("no dev record in this checkout")
    rows = verify_record(json.loads(path.read_text(encoding="utf-8")))
    checked = [r for r in rows if "reproducible" in r]
    assert checked, "the record should contain at least one non-skipped variant"
    assert all(r["reproducible"] is False for r in checked), (
        "the committed record was measured under prompts that are no longer in the tree; if this "
        "now passes, the dev split has been re-judged and this test should be updated"
    )


# ---------------------------------------------------------------- declared experiment design


def test_matched_pairs_reference_declared_variants():
    ids = {c.variant_id for c in GRID}
    for label, treatment, control, _ in MATCHED_PAIRS:
        assert treatment in ids, f"{label}: {treatment}"
        assert control in ids, f"{label}: {control}"


def test_matched_pairs_differ_in_exactly_one_respect():
    by_id = {c.variant_id: c for c in GRID}
    for label, treatment, control, difference in MATCHED_PAIRS:
        a, b = by_id[treatment], by_id[control]
        # `concurrency` is excluded for the same reason `config_hash` excludes it: it is an
        # operational knob for staying inside a provider's token-per-minute limit and cannot
        # change a verdict. Counting it as a confound would forbid giving a stronger model a
        # lower request rate, which is the only way to measure it at all.
        differing = {
            field for field in ("model", "mode", "restrict_tasks", "max_tokens", "samples",
                                "temperature", "include_reference", "include_boundary_examples",
                                "exemplars_path")
            if getattr(a, field) != getattr(b, field)
        }
        assert len(differing) == 1, (
            f"{label}: claims to differ in '{difference}' but differs in {sorted(differing)}; "
            "an uncontrolled comparison cannot attribute a kappa change to a mechanism"
        )


def test_ensemble_candidates_reference_declared_variants():
    ids = {c.variant_id for c in GRID}
    for label, components in ENSEMBLE_CANDIDATES:
        assert set(components) <= ids, f"{label}: {set(components) - ids}"
        assert len(set(components)) == len(components), f"{label} repeats a component"


def test_the_ensemble_grid_includes_a_single_mechanism_negative_control():
    # The probe predicts one mechanism ensembles to nothing. An ensemble study without its own
    # null case is not evidence either way.
    by_id = {c.variant_id: c for c in GRID}
    controls = [
        label for label, comps in ENSEMBLE_CANDIDATES
        if len({by_id[v].mode for v in comps}) == 1
    ]
    assert controls, "no single-mechanism control declared among the ensemble candidates"


def test_every_grid_variant_has_a_distinct_config_hash():
    # Two variants sharing a hash would share cached judgments and report each other's numbers.
    hashes = [c.config_hash for c in GRID]
    assert len(set(hashes)) == len(hashes)


def test_every_grid_variant_id_is_unique():
    ids = [c.variant_id for c in GRID]
    assert len(set(ids)) == len(ids)

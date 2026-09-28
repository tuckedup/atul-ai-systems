"""Tests for the combiner probe driver.

The probe's job is to answer a spend decision honestly, so the things worth pinning are the
honesty properties rather than the kappa it happens to report: that it makes no provider calls,
that it folds on groups and not on cases, that it refuses to reuse stale cache entries, and that
it labels itself a deviation from the acceptance protocol.
"""
from __future__ import annotations

import json

import pytest
from evalops.probe_combiners import group_folds, load_cached, oof_threshold_only, to_markdown
from evalops.probe_combiners import Row


def test_group_folds_keeps_a_group_whole():
    groups = ["a", "a", "a", "b", "b", "c", "d", "e", "f", "g"]
    folds = group_folds(groups, 3)
    assert sorted(i for f in folds for i in f) == list(range(len(groups)))
    for name in set(groups):
        rows = {i for i, g in enumerate(groups) if g == name}
        holders = [k for k, f in enumerate(folds) if rows & set(f)]
        assert len(holders) == 1, f"group {name} was split across folds {holders}"


def test_group_folds_is_deterministic():
    groups = [f"g{i % 7}" for i in range(40)]
    assert group_folds(groups, 5) == group_folds(groups, 5)


def test_group_folds_balances_uneven_group_sizes():
    # One source document contributing a dozen claims must not land a fold with 12 of 20 rows.
    groups = ["big"] * 12 + [f"s{i}" for i in range(8)]
    sizes = sorted(len(f) for f in group_folds(groups, 4))
    assert sizes[-1] <= 12 + 2  # the oversized group dominates its own fold and nothing else


def test_group_folds_covers_every_row_with_more_folds_than_groups():
    folds = group_folds(["a", "b"], 5)
    assert sorted(i for f in folds for i in f) == [0, 1]


def _cache_line(case_id, variant, config_hash, score=0.8, status="ok"):
    return json.dumps({"judgment": {
        "case_id": case_id, "variant_id": variant, "model_requested": "m",
        "config_hash": config_hash, "raw_score": score, "status": status,
    }})


def test_load_cached_drops_stale_config_hashes(tmp_path):
    # The cache accumulates across corpus and rubric revisions. Mixing two config hashes under one
    # variant id would blend two different judges under one name.
    path = tmp_path / "judgment_cache.jsonl"
    path.write_text("\n".join([
        _cache_line("c1", "v1", "current"),
        _cache_line("c2", "v1", "an-older-rubric"),
        _cache_line("c3", "v-not-in-the-run-record", "whatever"),
    ]), encoding="utf-8")
    kept, skipped = load_cached(path, variant_config={"v1": "current"})
    assert [j.case_id for j in kept] == ["c1"]
    assert skipped == {"stale config_hash": 1, "variant not in the run record": 1}


def test_load_cached_keeps_errored_judgments_for_the_caller_to_exclude(tmp_path):
    # `align` is what drops unusable judgments; `load_cached` must not hide them, or the coverage
    # denominator would lose the cases that failed.
    path = tmp_path / "judgment_cache.jsonl"
    path.write_text(_cache_line("c1", "v1", "h", score=None, status="provider_error"),
                    encoding="utf-8")
    kept, skipped = load_cached(path, variant_config={"v1": "h"})
    assert len(kept) == 1 and kept[0].usable is False and skipped == {}


def test_load_cached_tolerates_blank_lines(tmp_path):
    path = tmp_path / "judgment_cache.jsonl"
    path.write_text(_cache_line("c1", "v1", "h") + "\n\n", encoding="utf-8")
    kept, _ = load_cached(path, variant_config={"v1": "h"})
    assert len(kept) == 1


def _rows(n=40):
    return [
        Row(case_id=f"c{i}", human=i % 2, group=f"g{i // 2}", task="code",
            provenance="human_expert_annotation")
        for i in range(n)
    ]


def test_oof_threshold_only_never_tunes_on_the_held_out_fold():
    # A separable set: a threshold learned on any fold generalises, so predictions are perfect.
    rows = _rows()
    scores = [0.9 if r.human else 0.1 for r in rows]
    folds = group_folds([r.group for r in rows], 4)
    assert oof_threshold_only(rows, scores, folds) == [r.human for r in rows]


def test_oof_threshold_only_does_not_recover_a_label_it_cannot_see():
    # Scores carry no information about the label, so out-of-fold tuning cannot manufacture
    # agreement. An in-sample threshold search on a noisy set often can, which is the point.
    rows = _rows(60)
    scores = [0.5] * len(rows)
    folds = group_folds([r.group for r in rows], 5)
    predicted = oof_threshold_only(rows, scores, folds)
    assert len(set(predicted)) == 1, "a constant score must produce a constant prediction"


def test_markdown_states_that_the_probe_is_not_a_result():
    report = {
        "protocol_deviation": "fits inside dev",
        "components": ["v1"],
        "provider_calls": 0,
        "coverage": {"n_complete": 10, "n_offered": 20, "coverage": 0.5,
                     "missing_by_component": {"v1": 10}},
        "evidence_features": {"note": "0/0 carry evidence_location."},
        "tracks": [{"track": "t", "n": 3, "insufficient": True, "note": "too few"}],
    }
    md = to_markdown(report)
    assert "**Not a result.**" in md
    assert "fits inside dev" in md
    assert "Provider calls: 0" in md
    assert "50.0%" in md


def test_markdown_renders_a_full_track():
    report = {
        "protocol_deviation": "d", "components": ["v1"], "provider_calls": 0,
        "coverage": {"n_complete": 1, "n_offered": 1, "coverage": 1.0,
                     "missing_by_component": {}},
        "evidence_features": {"note": "n/a"},
        "tracks": [{
            "track": "headline", "n": 100, "n_groups": 40, "human_prevalence": 0.5,
            "task_mix": {"summarize": 100}, "folds": 5,
            "results": [
                {"name": "v1", "kind": "single", "kappa": 0.6, "defined": True,
                 "observed_agreement": 0.8, "n": 100,
                 "confusion": {"tp": 40, "fp": 10, "fn": 10, "tn": 40, "n": 100}},
                {"name": "broken", "kind": "learned", "kappa": None, "note": "not fittable"},
            ],
            "comparison": {
                "best_single": {"name": "v1", "kappa": 0.6},
                "best_overall": {"name": "v1", "kappa": 0.6},
                "margin_over_best_single": 0.0, "combiner_helps": False,
            },
        }],
    }
    md = to_markdown(report)
    assert "| `v1` | single | 0.6000 |" in md
    assert "not fittable" in md
    assert "Combiner helps: **False**" in md


def test_markdown_reports_a_comparison_error_rather_than_omitting_it():
    report = {
        "protocol_deviation": "d", "components": ["v1"], "provider_calls": 0,
        "coverage": {"n_complete": 1, "n_offered": 1, "coverage": 1.0,
                     "missing_by_component": {}},
        "evidence_features": {"note": "n/a"},
        "tracks": [{
            "track": "t", "n": 50, "n_groups": 20, "human_prevalence": 0.5,
            "task_mix": {}, "folds": 5, "results": [],
            "comparison": {"error": "no single-component contender was supplied"},
        }],
    }
    assert "Comparison unavailable" in to_markdown(report)


# ---------------------------------------------------------------- the committed artifact


def test_the_committed_probe_artifact_declares_its_protocol_deviation():
    from pathlib import Path

    path = Path(__file__).parent.parent / "evalops" / "data" / "combiner_probe.json"
    if not path.exists():
        pytest.skip("probe artifact not generated in this checkout")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["provider_calls"] == 0
    assert "NOT a held-out measurement" in report["probe"]
    assert "TRAIN" in report["protocol_deviation"]
    # The finding this artifact records: on the headline track no combiner beat the strongest
    # single judge. If a future run changes that, this assertion should be updated deliberately
    # rather than drift unnoticed.
    headline = next(t for t in report["tracks"]
                    if t["track"] == "human_judgment_of_response")
    assert headline["comparison"]["combiner_helps"] is False

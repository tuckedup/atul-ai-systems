"""Tests for the full-gate and screening simulations (no data, no model)."""
from __future__ import annotations

import json

import numpy as np
from evalops import judge_improvement_gate_sim as gs


def mean(key, n=1500, gain=0.05, d=0.09, reps=60, seed=2, **kw):
    rng = np.random.default_rng(seed)
    return float(np.mean([gs.simulate_gate(n, 0.85, d, gain, 0.70, rng, draws=120, **kw)[key] for _ in range(reps)]))


def test_full_gate_is_never_more_likely_than_detection_or_the_binary_gate():
    rng = np.random.default_rng(5)
    for _ in range(80):
        r = gs.simulate_gate(1500, 0.85, 0.09, 0.04, 0.70, rng, draws=100)
        assert r["gate_full"] <= r["gate_binary"] <= r["c1"] and r["gate_binary"] <= r["c2"]


def test_a_useless_candidate_essentially_never_passes_the_full_gate():
    assert mean("gate_full", gain=0.0, reps=100) <= 0.03


def test_at_a_true_gain_equal_to_the_point_threshold_the_gate_is_roughly_a_coin_flip():
    p = mean("gate_binary", gain=0.03, reps=150)
    assert 0.25 <= p <= 0.75


def test_detection_power_exceeds_full_gate_probability_at_the_threshold_gain():
    assert mean("c1", gain=0.03, reps=100) > mean("gate_binary", gain=0.03, reps=100) + 0.2


def test_a_clearly_larger_gain_passes_the_full_gate_most_of_the_time():
    assert mean("gate_full", gain=0.06, reps=80) >= 0.9


def test_the_label_cap_branch_engages_when_disagreements_are_many_and_bounds_labels():
    rng = np.random.default_rng(1)
    rs = [gs.simulate_gate(1500, 0.85, 0.30, 0.05, 0.70, rng, draws=60, pi_conc=0.15, label_cap=100) for _ in range(20)]
    assert all(r["capped"] == 1.0 for r in rs) and all(r["labelled"] < 100 * 1.6 + 0.15 * 1500 for r in rs)


def test_without_a_cap_every_disagreement_is_labelled():
    rng = np.random.default_rng(1)
    r = gs.simulate_gate(1500, 0.85, 0.09, 0.03, 0.70, rng, draws=60, label_cap=None)
    assert r["capped"] == 0.0 and r["labelled"] >= r["disagreements"]


def test_screen_rarely_passes_a_useless_candidate_and_usually_a_good_one():
    rng = np.random.default_rng(3)
    null = np.mean([gs.simulate_screen(0.85, 0.09, 0.0, 0.70, rng)["screen"] for _ in range(400)])
    good = np.mean([gs.simulate_screen(0.85, 0.09, 0.05, 0.70, rng)["screen"] for _ in range(400)])
    assert null <= 0.25 and good >= 0.8


def test_cli_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "gate_table", lambda *a, **k: [{"N": 1}])
    monkeypatch.setattr(gs, "screen_table", lambda *a, **k: [{"s": 1}])
    out = tmp_path / "o.json"
    assert gs.main(["--out", str(out)]) == 0 and "evaluation_N1500" in json.loads(out.read_text(encoding="utf8"))
    assert gs.main(["--out", str(out)]) == 2

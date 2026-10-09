"""Tests for the offline judge-improvement power simulation (no model, no data)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from evalops import judge_improvement_power as jp


def rate(n, gain, disagree, reps=60, key="both_ok", seed=1):
    rng = np.random.default_rng(seed)
    return float(np.mean([jp.simulate_once(n, 0.85, disagree, gain, 0.70, rng, draws=150)[key] for _ in range(reps)]))


def test_kappa_matches_the_textbook_value():
    k = jp._kappa(np.array([40.0]), np.array([10.0]), np.array([10.0]), np.array([40.0]))
    assert k[0] == pytest.approx(0.6)


def test_kappa_is_nan_when_chance_agreement_is_degenerate():
    k = jp._kappa(np.array([10.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert np.isnan(k[0])


def test_a_useless_candidate_is_rarely_declared_better():
    assert rate(2000, 0.0, 0.09, reps=80) <= 0.10


def test_a_clearly_better_candidate_is_usually_detected():
    assert rate(2000, 0.05, 0.09, reps=40) >= 0.9


def test_power_grows_with_sample_size():
    assert rate(1000, 0.02, 0.09, reps=80) < rate(4000, 0.02, 0.09, reps=80)


def test_disagreements_are_always_labelled_and_agreements_sampled():
    rng = np.random.default_rng(3)
    out = jp.simulate_once(2000, 0.85, 0.09, 0.03, 0.7, rng, draws=50)
    assert out["labelled"] >= out["disagreements"] and out["labelled"] < 2000 * 0.5


def test_the_cli_refuses_to_overwrite_and_writes_the_assumptions(tmp_path, monkeypatch):
    monkeypatch.setattr(jp, "power_table", lambda **kw: [{"N": 1}])
    out = tmp_path / "p.json"
    assert jp.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf8"))["assumptions"]["pi_concordant"] == jp.PI_CONCORDANT
    assert jp.main(["--out", str(out)]) == 2

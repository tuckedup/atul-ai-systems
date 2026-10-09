"""Sanity tests for the typed-veto validation power simulation. Pure simulation: no data, no model, no calls."""
import numpy as np
from evalops import typed_veto_power as pw


def test_simulation_is_deterministic_for_a_seed():
    a = pw.power_table(scenarios=pw.SCENARIOS[:1], sizes=(300,), reps=20, draws=50, seed=1)
    b = pw.power_table(scenarios=pw.SCENARIOS[:1], sizes=(300,), reps=20, draws=50, seed=1)
    assert a == b


def test_sampling_design_is_the_declared_one():
    assert (pw.PI_PASS, pw.PI_REJECT, pw.DOC_SIZE) == (0.15, 0.15, 10)
    rng = np.random.default_rng(3)
    res = [pw.simulate_once(pw.SCENARIOS[0], 1000, rng, 50) for _ in range(30)]
    flags = np.mean([r["flags"] for r in res])
    labelled = np.mean([r["labelled"] for r in res])
    assert 20 < flags < 45 and 0.1 < (labelled - flags) / (1000 - flags) < 0.2  # flags always labelled; the others at ~0.15


def test_the_null_scenario_has_zero_accuracy_gain_and_a_controlled_false_positive_rate():
    row = pw.power_table(scenarios=[pw.SCENARIOS[3]], sizes=(1000,), reps=300, draws=100, seed=5)[0]
    assert abs(row["mean_accuracy_gain"]) < 0.004 and row["power_accuracy_gain"] <= 0.10 and row["power_precision_wilson"] <= 0.10


def test_power_rises_with_sample_size_under_the_dev_profile_and_a_harmful_veto_is_never_declared_useful():
    rows = pw.power_table(scenarios=[pw.SCENARIOS[0], pw.SCENARIOS[4]], sizes=(300, 2500), reps=150, draws=100, seed=7)
    good = {r["N"]: r for r in rows if r["scenario"].startswith("S1")}
    bad = [r for r in rows if r["scenario"].startswith("S5")]
    assert good[2500]["power_accuracy_gain"] > good[300]["power_accuracy_gain"] + 0.3 and good[2500]["power_accuracy_gain"] > 0.9
    assert all(r["power_accuracy_gain"] <= 0.05 and r["power_kappa_gain"] <= 0.1 for r in bad)


def test_kappa_gain_is_not_the_same_thing_as_accuracy_gain():
    row = pw.power_table(scenarios=[pw.SCENARIOS[3]], sizes=(2500,), reps=200, draws=100, seed=11)[0]
    assert row["mean_kappa_gain"] > 0.003 and abs(row["mean_accuracy_gain"]) < 0.004  # zero net accuracy change, yet kappa moves

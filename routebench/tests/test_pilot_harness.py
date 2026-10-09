"""Synthetic tests for the pilot guards. No model, no GPU, no network."""
import dataclasses
import sys

import numpy as np
import pytest
from evalops.pilot import harness as h
from evalops.pilot.config import PILOT, PilotConfig


def manifest(n_groups_train=30, n_groups_val=6, per_group=6, unsupported_every=7):
    rows, k = [], 0
    for part, groups in (("train", n_groups_train), ("validation", n_groups_val), ("test", 3)):
        for g in range(groups):
            for j in range(per_group):
                label = 0 if (k % unsupported_every == 0) else 1
                rows.append([f"{part}{g}#{j}", f"{part}{g}-r{j // 3}", "s", part, "fp", f"g-{part}{g}", label, 0, 5, False, False])
                k += 1
    m = {"version": PILOT.manifest_version, "columns": ["example_id", "response_id", "source_id", "partition",
                                                        "document_fp16", "group_id", "label", "start", "end",
                                                        "implicit_true", "crosses_boundary"], "rows": rows,
         "counts": {}, "meta": {}}
    m["manifest_sha256"] = h.content_hash(m)
    return m


def cfg_for(m):
    return dataclasses.replace(PILOT, manifest_sha256=m["manifest_sha256"])


def test_the_frozen_configuration_names_its_manifest_and_directions():
    assert PILOT.manifest_sha256, "the shipped configuration must carry the frozen manifest hash"
    assert (PILOT.unsupported_token_id, PILOT.supported_token_id) == (3, 209)
    assert "UNSUPPORTED = positive" in PILOT.threshold_objective
    assert sum(PILOT.phase_seconds.values()) + PILOT.startup_seconds == PILOT.total_seconds == 7200
    assert PILOT.api_spend_usd == 0.0 and PILOT.probe_optimizer_updates >= 1
    assert PILOT.config_hash() == PilotConfig().config_hash()


def test_changed_or_missing_manifest_hash_is_rejected():
    m = manifest()
    cfg = cfg_for(m)
    h.verify_manifest(m, cfg)
    tampered = {**m, "rows": m["rows"][:-1]}
    with pytest.raises(h.ManifestMismatch):
        h.verify_manifest(tampered, cfg)
    other = manifest(n_groups_train=31)
    with pytest.raises(h.ManifestMismatch):
        h.verify_manifest(other, cfg)
    with pytest.raises(h.ManifestMismatch):
        h.verify_manifest(m, dataclasses.replace(PILOT, manifest_sha256=""))
    with pytest.raises(h.ManifestMismatch):
        h.verify_manifest({**m, "version": "other"}, cfg)


def test_config_change_after_freezing_is_detected():
    frozen = PILOT.config_hash()
    h.verify_config(PILOT, frozen)
    with pytest.raises(h.ConfigMismatch):
        h.verify_config(dataclasses.replace(PILOT, learning_rate=1e-3), frozen)


def test_plan_is_group_disjoint_and_never_uses_the_ragtruth_test_partition():
    m = manifest()
    rows = h.rows_from_manifest(m)
    plan = h.make_plan(rows, cfg_for(m))
    groups = {name: {r.group_id for r in part} for name, part in
              (("fit", plan.train_fit), ("sel", plan.train_select), ("val", plan.validation))}
    assert not groups["fit"] & groups["sel"] and not groups["fit"] & groups["val"] and not groups["sel"] & groups["val"]
    assert all(r.partition == "train" for r in plan.train_fit + plan.train_select)
    assert all(r.partition == "validation" for r in plan.validation)
    assert not any(r.partition == "test" for r in plan.train_fit + plan.train_select + plan.validation)
    assert plan == h.make_plan(rows, cfg_for(m))  # deterministic


def test_training_inputs_from_held_out_partitions_are_rejected():
    m = manifest()
    rows = h.rows_from_manifest(m)
    plan = h.make_plan(rows, cfg_for(m))
    ok = [r.example_id for r in plan.train_fit[:5]]
    h.assert_training_inputs(ok, plan, rows)
    for bad in (plan.train_select[0], plan.validation[0], next(r for r in rows if r.partition == "test")):
        with pytest.raises(h.PartitionError):
            h.assert_training_inputs(ok + [bad.example_id], plan, rows)
    with pytest.raises(h.PartitionError):
        h.assert_training_inputs(["nonexistent"], plan, rows)


def test_a_group_shared_between_plan_partitions_is_refused():
    m = manifest()
    rows = h.rows_from_manifest(m)
    tr = [r for r in rows if r.partition == "train"]
    clash = dataclasses.replace(next(r for r in rows if r.partition == "validation"), group_id=tr[0].group_id)
    with pytest.raises(h.PartitionError):
        h.make_plan(rows + [clash], cfg_for(m))


def test_training_sample_is_fixed_length_capped_and_train_fit_only():
    m = manifest(n_groups_train=200, per_group=10)
    rows = h.rows_from_manifest(m)
    cfg = dataclasses.replace(cfg_for(m), train_unsupported_max=20, train_supported_per_unsupported=2)
    plan = h.make_plan(rows, cfg)
    lens = {r.example_id: 800 for r in rows}
    long_ids = [r.example_id for r in plan.train_fit[:7]]
    for i in long_ids:
        lens[i] = cfg.max_input_tokens + 1
    a = h.training_sample(plan, lens, cfg)
    assert a == h.training_sample(plan, lens, cfg)
    assert a["n_unsupported"] == 20 and a["n_supported"] == 40 and a["excluded_over_length"] == 7
    assert not set(long_ids) & set(a["example_ids"])
    h.assert_training_inputs(a["example_ids"], plan, rows)
    assert cfg.weight_unsupported == cfg.weight_supported == 1.0
    with pytest.raises(h.PartitionError):
        h.training_sample(plan, {r.example_id: 9999 for r in rows}, cfg)


def test_evaluation_sample_takes_whole_groups_and_ignores_labels():
    m = manifest()
    rows = h.rows_from_manifest(m)
    cfg = cfg_for(m)
    plan = h.make_plan(rows, cfg)
    lens = {r.example_id: 900 for r in rows}
    a = h.evaluation_sample(plan.validation, lens, 15, cfg, "val")
    by_group = {}
    for r in plan.validation:
        by_group.setdefault(r.group_id, set()).add(r.example_id)
    assert set(a["example_ids"]) == set().union(*(by_group[g] for g in a["groups"]))
    flipped = [dataclasses.replace(r, label=1 - r.label) for r in plan.validation]
    assert h.evaluation_sample(flipped, lens, 15, cfg, "val") == a  # label-blind
    assert len(a["example_ids"]) >= 15 and len(a["groups"]) < len(by_group)


def test_incomplete_or_invalid_scores_never_evaluate():
    ids = ["a", "b", "c"]
    assert list(h.check_scores(ids, {"a": 0.1, "b": 0.5, "c": 0.9})) == [0.1, 0.5, 0.9]
    for bad in ({"a": 0.1, "b": 0.5}, {"a": 0.1, "b": 0.5, "c": 0.9, "d": 0.2}, {"a": 0.1, "b": float("nan"), "c": 0.9},
                {"a": 0.1, "b": 1.5, "c": 0.9}, {"a": 0.1, "b": float("inf"), "c": 0.9}):
        with pytest.raises(h.IncompleteEvaluation):
            h.check_scores(ids, bad)


def test_metrics_treat_unsupported_as_the_positive_class():
    y = np.array([1, 1, 1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.6])
    c = h.confusion(y, s, 0.5)
    assert c == {"tp": 1, "fp": 1, "fn": 1, "tn": 3}
    m = h.metrics(y, s, 0.5)
    assert m["unsupported_recall"] == 0.5 and m["unsupported_precision"] == 0.5
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 0.75) / 2)
    assert 0 < m["pr_auc_unsupported"] <= 1
    always_supported = h.metrics(np.array([1] * 9 + [0]), np.ones(10), 0.5)
    assert always_supported["unsupported_recall"] == 0.0 and always_supported["balanced_accuracy"] == 0.5


def test_threshold_uses_only_the_selection_data_and_prefers_the_middle_on_ties():
    y = np.array([1, 1, 1, 0, 0])
    s = np.array([0.95, 0.9, 0.85, 0.1, 0.05])
    t = h.select_threshold(y, s, PILOT)
    assert t == 0.5 and h.metrics(y, s, t)["balanced_accuracy"] == 1.0


def test_stop_rule_requires_gain_ci_recall_and_complete_coverage():
    base = {"balanced_accuracy": 0.70, "unsupported_recall": 0.50}
    good = {"balanced_accuracy": 0.75, "unsupported_recall": 0.55}
    assert h.decide_continue(base, good, 0.01, PILOT, True)["continue"] is True
    assert h.decide_continue(base, good, -0.01, PILOT, True)["continue"] is False
    assert h.decide_continue(base, {"balanced_accuracy": 0.72, "unsupported_recall": 0.6}, 0.01, PILOT, True)["continue"] is False
    assert h.decide_continue(base, {"balanced_accuracy": 0.8, "unsupported_recall": 0.40}, 0.05, PILOT, True)["continue"] is False
    assert h.decide_continue(base, good, 0.01, PILOT, False)["continue"] is False


def test_paired_bootstrap_resamples_documents_not_sentences():
    m = manifest()
    rows = [r for r in h.rows_from_manifest(m) if r.partition == "validation"]
    y = np.array([r.label for r in rows])
    rng = np.random.default_rng(0)
    base = np.where(y == 1, 0.6, 0.4) + rng.normal(0, 0.05, len(y))
    tuned = np.where(y == 1, 0.9, 0.1) + rng.normal(0, 0.05, len(y))
    out = h.paired_bootstrap_lower(rows, y, np.clip(base, 0, 1), np.clip(tuned, 0, 1), 0.5, 0.5, PILOT)
    assert out["draws_used"] > 0 and out["upper95"] >= out["lower95"]


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_phase_and_total_caps_raise_and_are_not_absorbed():
    clock = FakeClock()
    b = h.ResourceBudget(PILOT, clock)
    with b.phase("baseline_eval") as tick:
        clock.t += 600
        tick()
    assert b.spent["baseline_eval"] == 600
    with pytest.raises(h.BudgetExceeded), b.phase("baseline_eval") as tick:
        clock.t += 700  # 600 + 700 > 1200
        tick()
    with pytest.raises(h.BudgetExceeded), b.phase("training"):
        clock.t += 3601
    with pytest.raises(h.BudgetExceeded), b.phase("undeclared"):
        pass


def test_total_cap_stops_before_the_two_hour_limit_is_passed():
    clock = FakeClock()
    cfg = dataclasses.replace(PILOT, total_seconds=1000, phase_seconds={"a": 800, "b": 800})
    b = h.ResourceBudget(cfg, clock)
    with b.phase("a"):
        clock.t += 700
    with pytest.raises(h.BudgetExceeded), b.phase("b") as tick:
        clock.t += 400
        tick()
    assert b.remaining("b") == 0.0


def test_real_wall_clock_timeout_kills_the_command():
    ok = h.run_with_timeout([sys.executable, "-c", "print('done')"], 20)
    assert ok.returncode == 0 and "done" in ok.stdout
    with pytest.raises(h.BudgetExceeded):
        h.run_with_timeout([sys.executable, "-c", "import time; time.sleep(30)"], 1)


# ---- readiness pass 2: tie-invariant AP, direction reporting, whole-response scope -----------------------------------------
def test_average_precision_is_invariant_to_row_order_and_matches_the_reference():
    from sklearn.metrics import average_precision_score

    y = np.array([1, 0, 1, 0])
    s = np.array([0.5, 0.5, 0.5, 0.5])  # every score tied
    ref = average_precision_score(1 - y, 1 - s)
    assert h.average_precision_unsupported(y, s) == pytest.approx(ref) == pytest.approx(0.5)
    assert h.average_precision_unsupported(y[::-1], s[::-1]) == h.average_precision_unsupported(y, s)
    a = h.average_precision_unsupported(np.array([0, 1]), np.array([0.5, 0.5]))
    b = h.average_precision_unsupported(np.array([1, 0]), np.array([0.5, 0.5]))
    assert a == b
    rng = np.random.default_rng(3)
    for _ in range(25):
        y = (rng.random(60) > 0.2).astype(int)
        if y.sum() in (0, 60):
            continue
        s = np.round(rng.random(60), 1)  # many ties
        perm = rng.permutation(60)
        got = h.average_precision_unsupported(y, s)
        assert got == pytest.approx(average_precision_score(1 - y, 1 - s))
        assert h.average_precision_unsupported(y[perm], s[perm]) == pytest.approx(got)


def test_below_chance_results_are_reported_not_rejected():
    y = np.array([1, 1, 1, 0, 0, 1, 0, 1])
    good = np.array([0.9, 0.8, 0.95, 0.1, 0.2, 0.7, 0.15, 0.85])
    assert h.direction_report(y, good)["below_chance"] is False
    bad = h.direction_report(y, 1 - good)  # a correctly wired but poor model must be preserved, not raised
    assert bad["below_chance"] is True and bad["auc_supported_higher"] < 0.5
    assert h.metrics(y, 1 - good, 0.5)["balanced_accuracy"] < 0.5


def test_scoring_path_wiring_is_checked_deterministically():
    assert h.score_from_logits(-2.0, 3.0) > 0.99 and h.score_from_logits(3.0, -2.0) < 0.01
    assert h.score_from_logits(1.0, 1.0) == pytest.approx(0.5)
    h.assert_wiring(PILOT)  # token ids, label direction and score direction agree with the configuration
    with pytest.raises(h.ScoreDirectionError):
        h.assert_wiring(dataclasses.replace(PILOT, unsupported_token_id=PILOT.supported_token_id))
    assert h.build_input("DOC", "SENT", PILOT, eos="</s>") == "predict: DOC</s>SENT"


def test_response_metrics_cover_only_fully_evaluated_responses_and_use_authoritative_labels():
    rows = [h.Row("a#0", "r1", "validation", "g1", 1), h.Row("a#1", "r1", "validation", "g1", 1),
            h.Row("b#0", "r2", "validation", "g2", 1), h.Row("b#1", "r2", "validation", "g2", 0),
            h.Row("c#0", "r3", "validation", "g3", 1), h.Row("c#1", "r3", "validation", "g3", 1)]
    scores = {"a#0": 0.9, "a#1": 0.8, "b#0": 0.9, "b#1": 0.1, "c#0": 0.9}  # r3 is only partly evaluated; a#1 is complete
    # authoritative label of r3 is unsupported (for example an annotated wrapper), although its in-scope sentences are all supported
    labels = {"r1": 1, "r2": 0, "r3": 0}
    out = h.response_level(rows, scores, labels, 0.5)
    assert out["n_complete"] == 2 and out["n_partial"] == 1 and out["n_not_evaluated"] == 0
    assert out["metrics"]["tp"] == 1 and out["metrics"]["tn"] == 1 and out["metrics"]["fp"] == 0
    assert out["scope"] == "complete responses only"
    none = h.response_level(rows, {}, labels, 0.5)
    assert none["metrics"] is None and none["n_not_evaluated"] == 3

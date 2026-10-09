"""Accumulation arithmetic for the training loop, tested without torch."""
import pytest
from evalops.pilot import trainloop as t


def test_every_example_is_seen_once_per_epoch_including_the_ragged_tail():
    plan = t.update_plan(n_examples=37, micro_batch=2, grad_accum=8, epochs=2, seed=5)
    seen = [i for g in plan for mb in g.micro_batches for i in mb]
    assert sorted(seen) == sorted(list(range(37)) * 2)
    assert len(plan) == t.expected_updates(37, 2, 8, 2) == 5  # ceil(2 * 19 micro-batches / 8)
    assert plan == t.update_plan(37, 2, 8, 2, 5)  # deterministic
    assert plan != t.update_plan(37, 2, 8, 2, 6)


def test_the_last_partial_group_is_stepped_and_rescaled_not_discarded():
    plan = t.update_plan(n_examples=35, micro_batch=2, grad_accum=8, epochs=1, seed=1)  # 18 micro-batches -> groups 8, 8, 2
    assert [g.size for g in plan] == [8, 8, 2]
    micro_calls, steps = [], []
    stats = t.run_updates(plan, grad_accum=8, warmup_fraction=0.05, micro_fn=lambda idx, div: micro_calls.append((idx, div)),
                          step_fn=lambda lr, rescale: steps.append((lr, rescale)), tick=lambda: None)
    assert stats["optimizer_updates"] == 3 and stats["micro_batches"] == 18 and stats["examples_seen"] == 35
    assert stats["partial_final_group"] == 1 and stats["final_group_size"] == 2
    assert all(div == 8 for _, div in micro_calls)
    assert [round(r, 6) for _, r in steps] == [1.0, 1.0, 4.0]  # 8 / 2: mean over the two micro-batches actually accumulated


def test_a_full_last_group_needs_no_rescale_and_tick_runs_every_micro_batch():
    plan = t.update_plan(32, 2, 8, 1, 0)
    ticks, steps = [], []
    t.run_updates(plan, grad_accum=8, warmup_fraction=0.05, micro_fn=lambda *_: None,
                  step_fn=lambda lr, rescale: steps.append(rescale), tick=lambda: ticks.append(1))
    assert len(ticks) == 16 and steps == [1.0, 1.0]


def test_learning_rate_schedule_warms_up_then_decays_to_near_zero():
    total = 40
    vals = [t.lr_scale(s, total, 0.05) for s in range(1, total + 1)]
    assert vals[0] == pytest.approx(0.5) and vals[1] == pytest.approx(1.0 * (1 - 1 / 40))  # warm-up of 2 updates
    assert max(vals) <= 1.0 and vals[-1] < 0.05 and all(v >= 0 for v in vals)
    assert vals[5] > vals[20] > vals[-1]


def test_invalid_sizes_are_rejected():
    for bad in ((0, 2, 8, 1), (5, 0, 8, 1), (5, 2, 0, 1), (5, 2, 8, 0)):
        with pytest.raises(ValueError):
            t.update_plan(*bad, seed=0)

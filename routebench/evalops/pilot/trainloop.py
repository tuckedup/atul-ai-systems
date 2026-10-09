"""Torch-free training schedule: which micro-batches make up each optimizer update, and how the update is scaled.

The real backend drives this; keeping the arithmetic here lets the accumulation, the final partial batch and the learning-rate
schedule be tested offline.

Rules (frozen with the configuration):
  * every epoch visits every example exactly once, in a seeded order;
  * micro-batches of `micro_batch` examples are grouped `grad_accum` at a time into one optimizer update;
  * the LAST group of a run may hold fewer micro-batches; it is NOT discarded: its gradients are rescaled by
    `grad_accum / len(group)` so the update is the mean over the micro-batches actually accumulated;
  * the learning rate rises linearly over the warm-up updates, then decays linearly towards zero.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UpdateGroup:
    micro_batches: tuple[tuple[int, ...], ...]  # each entry: example indices of one micro-batch

    @property
    def size(self) -> int:
        return len(self.micro_batches)


def update_plan(n_examples: int, micro_batch: int, grad_accum: int, epochs: int, seed: int) -> list[UpdateGroup]:
    if min(n_examples, micro_batch, grad_accum, epochs) < 1:
        raise ValueError("all schedule sizes must be >= 1")
    micros: list[tuple[int, ...]] = []
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(n_examples).tolist()
        micros += [tuple(order[i:i + micro_batch]) for i in range(0, n_examples, micro_batch)]
    return [UpdateGroup(tuple(micros[i:i + grad_accum])) for i in range(0, len(micros), grad_accum)]


def lr_scale(step: int, total_updates: int, warmup_fraction: float) -> float:
    """Multiplier for the base learning rate at 1-based optimizer `step`."""
    warm = max(1, int(warmup_fraction * total_updates))
    return min(1.0, step / warm) * max(0.0, 1.0 - (step - 1) / max(1, total_updates))


def run_updates(plan: Sequence[UpdateGroup], *, grad_accum: int, warmup_fraction: float,
                micro_fn: Callable[[tuple[int, ...], int], None], step_fn: Callable[[float, float], None],
                tick: Callable[[], None]) -> dict[str, int]:
    """Drive a backend through the plan.

    `micro_fn(indices, divisor)` must run forward+backward for one micro-batch with the loss divided by `divisor`
    (always `grad_accum`). `step_fn(lr_multiplier, grad_rescale)` must rescale the accumulated gradients by `grad_rescale`
    (1.0 for a full group, grad_accum / len(group) for a partial one), take the optimizer step and zero the gradients.
    `tick()` is called after every micro-batch.
    """
    micro = updates = examples = 0
    total = len(plan)
    for group in plan:
        for idx in group.micro_batches:
            micro_fn(idx, grad_accum)
            micro += 1
            examples += len(idx)
            tick()
        updates += 1
        step_fn(lr_scale(updates, total, warmup_fraction), grad_accum / group.size if group.size != grad_accum else 1.0)
    return {"optimizer_updates": updates, "micro_batches": micro, "examples_seen": examples, "planned_updates": total,
            "final_group_size": plan[-1].size if plan else 0, "partial_final_group": int(bool(plan) and plan[-1].size != grad_accum)}


def expected_updates(n_examples: int, micro_batch: int, grad_accum: int, epochs: int) -> int:
    return math.ceil(epochs * math.ceil(n_examples / micro_batch) / grad_accum)

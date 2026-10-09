"""TorchBackend behaviour that can be checked WITHOUT torch: call order, synchronisation, and real optimizer updates.

Stub torch / peft objects record calls. This does not test numerical behaviour or the GPU.
"""
import dataclasses
import sys
import types

import pytest
from evalops.pilot import torch_backend as tb
from evalops.pilot.config import PILOT


class Recorder:
    def __init__(self):
        self.log = []

    def __getattr__(self, name):
        def call(*a, **k):
            self.log.append(name)
            return 0
        return call


def make_backend(monkeypatch, log):
    torch = types.SimpleNamespace(
        manual_seed=lambda s: log.append(f"manual_seed({s})"),
        cuda=types.SimpleNamespace(
            manual_seed_all=lambda s: log.append(f"cuda_manual_seed_all({s})"),
            synchronize=lambda: log.append("sync"),
            reset_peak_memory_stats=lambda: log.append("reset_peak"),
            max_memory_allocated=lambda: 4.0e9))
    peft = types.SimpleNamespace(
        LoraConfig=lambda **k: log.append("LoraConfig") or k,
        get_peft_model=lambda base, cfg: log.append("get_peft_model") or types.SimpleNamespace(
            unload=lambda: base, eval=lambda: None, parameters=list, gradient_checkpointing_enable=lambda: None,
            enable_input_require_grads=lambda: None))
    monkeypatch.setitem(sys.modules, "peft", peft)
    b = tb.TorchBackend.__new__(tb.TorchBackend)
    b.cfg, b.torch, b._base = PILOT, torch, types.SimpleNamespace(name="base")
    b.model = b._base
    return b


def test_the_seed_is_set_before_the_adapter_is_initialised(monkeypatch):
    log = []
    b = make_backend(monkeypatch, log)
    b._attach_adapter()
    assert log.index(f"manual_seed({PILOT.train_seed})") < log.index("get_peft_model")
    assert log.index(f"cuda_manual_seed_all({PILOT.train_seed})") < log.index("get_peft_model")


def test_probe_synchronises_around_every_timed_section_and_reports_real_updates(monkeypatch):
    log = []
    b = make_backend(monkeypatch, log)
    b.score = lambda examples, tick: log.append("score") or {}
    b._attach_adapter = lambda: log.append("attach")
    b._discard_adapter = lambda: log.append("discard")
    b._run_training = lambda ex, plan, tick: log.append("train") or {
        "optimizer_updates": len(plan), "micro_batches": 0, "examples_seen": len(ex)}
    need = PILOT.micro_batch * PILOT.grad_accum * PILOT.probe_optimizer_updates
    out = b.probe([("i", "d", "s", 1)] * need, [("i", "d", "s")] * 4)
    assert out["optimizer_updates_done"] == PILOT.probe_optimizer_updates and out["peak_memory_gb"] == 4.0
    assert out["score_s_per_example"] >= 0 and out["train_s_per_example"] >= 0
    # every timed section is bracketed by a GPU synchronisation
    assert log[: log.index("score") + 2].count("sync") >= 2 and log.index("sync") < log.index("score")
    assert log[log.index("train") - 1] == "sync" and log[log.index("train") + 1] == "sync"
    assert log.index("reset_peak") < log.index("attach") < log.index("train") < log.index("discard")


def test_the_probe_with_too_few_examples_yields_fewer_updates_which_the_runner_rejects(monkeypatch):
    log = []
    b = make_backend(monkeypatch, log)
    b.score = lambda examples, tick: {}
    b._attach_adapter = lambda: None
    b._discard_adapter = lambda: None
    b._run_training = lambda ex, plan, tick: {"optimizer_updates": len(plan), "micro_batches": 0, "examples_seen": len(ex)}
    out = b.probe([("i", "d", "s", 1)] * 8, [("i", "d", "s")])  # 8 examples = 4 micro-batches < one full update of 8
    assert out["optimizer_updates_done"] == 1  # one (partial) update is still stepped, never silently dropped


def test_full_training_uses_the_configured_epochs_and_syncs(monkeypatch):
    log = []
    b = make_backend(monkeypatch, log)
    b.cfg = dataclasses.replace(PILOT, epochs=2)
    seen = {}
    b._attach_adapter = lambda: log.append("attach")
    b._run_training = lambda ex, plan, tick: seen.update(groups=len(plan), n=len(ex)) or {}
    b.train([("i", "d", "s", 1)] * 40, lambda: None)
    assert seen["n"] == 40 and seen["groups"] == -(-2 * 20 // PILOT.grad_accum) == 5
    assert log[0] == "attach" and log.count("sync") == 2


def test_the_backend_refuses_a_model_directory_that_is_not_the_pinned_revision(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=None, T5ForConditionalGeneration=None))
    with pytest.raises(RuntimeError, match="pinned revision"):
        tb.TorchBackend(PILOT, tmp_path / "some-other-directory")

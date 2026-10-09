"""Real backend for the pilot (MiniCheck-Flan-T5-Large + LoRA). Its GPU behaviour is NOT exercised by the repository's tests.

Written against the frozen configuration and the upstream MiniCheck scoring path (commit b58b9fa): input
`"predict: " + document + EOS + sentence`, first decoder step from start token 0, logits of ids (3 = unsupported, 209 = supported),
score = softmax over those two logits, index 1. torch, transformers and peft are imported lazily, so importing this module in the
project environment (which has none of them) is harmless. It must run in the isolated environment described in
`docs/ROUTEBENCH_PILOT_INSTALL_PLAN.md`, after the permission record and the GPU approval exist.

What IS tested offline (with stub torch/peft objects and a torch-free schedule): the order "seed, then adapter initialisation", the
GPU synchronisation around every timed section, the accumulation schedule including the final partial group, and that the probe
performs and reports real optimizer updates. What is NOT tested: numerical behaviour, memory, and throughput on the GPU.

Nothing here downloads a model: `model_dir` must be the already-present snapshot named by the pinned revision.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import PilotConfig
from .harness import SCORE_BATCH, build_input, score_batches_for
from .trainloop import run_updates, update_plan


def _noop() -> None:
    return None


class TorchBackend:
    def __init__(self, cfg: PilotConfig, model_dir: Path) -> None:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        if model_dir.name != cfg.base_revision:
            raise RuntimeError(f"model directory {model_dir.name!r} is not the pinned revision {cfg.base_revision!r}")
        self.cfg, self.torch = cfg, torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = T5ForConditionalGeneration.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
        self._base = self.model
        self.ids = torch.tensor([cfg.unsupported_token_id, cfg.supported_token_id], device="cuda")

    # ---- timing and determinism helpers ------------------------------------------------------------------------------------------
    def _sync(self) -> None:
        """GPU work is asynchronous: synchronise before reading a clock so a timing covers the work it claims to."""
        self.torch.cuda.synchronize()

    def vram_snapshot(self) -> dict[str, float]:
        """Allocated / reserved / peak GPU memory in GB (for the progress log)."""
        c = self.torch.cuda
        return {"vram_allocated_gb": round(c.memory_allocated() / 1e9, 3), "vram_reserved_gb": round(c.memory_reserved() / 1e9, 3),
                "vram_peak_gb": round(c.max_memory_allocated() / 1e9, 3)}

    def _beat(self, phase: str, done: int, total: int, **extra: Any) -> None:
        progress = getattr(self, "progress", None)
        if progress is not None:
            progress.log(phase, done, total, **extra)

    def _seed(self) -> None:
        self.torch.manual_seed(self.cfg.train_seed)
        self.torch.cuda.manual_seed_all(self.cfg.train_seed)

    # ---- forward: two label-token logits at the first decoder step ---------------------------------------------------------------------
    def _two_logits(self, pairs: Sequence[tuple[str, str]], grad: bool) -> Any:
        torch = self.torch
        texts = [build_input(doc, sent, self.cfg, eos=self.tok.eos_token) for doc, sent in pairs]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=False).to("cuda")  # never truncate
        if enc["input_ids"].shape[1] > self.cfg.max_input_tokens:
            raise RuntimeError("an over-length example reached the backend; the frozen sample must exclude it")
        dec = torch.full((len(pairs), 1), self.cfg.decoder_start_token_id, dtype=torch.long, device="cuda")
        with torch.set_grad_enabled(grad):
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], decoder_input_ids=dec)
        return out.logits[:, 0, :].index_select(-1, self.ids).float()

    def _score_batches(self, batches: Sequence[Sequence[tuple[str, str, str]]], tick: Callable[[], None], *, phase: str = "score",
                       durable: bool = False) -> dict[str, float]:
        """Score exactly the given batches (no re-sorting). With `durable`, write each batch's scores to the checkpoint file."""
        torch = self.torch
        self.model.eval()
        checkpoint = getattr(self, "checkpoint", None) if durable else None
        every = max(1, int(getattr(self.cfg, "heartbeat_every", 10)))
        total = sum(len(b) for b in batches)
        scores: dict[str, float] = {}
        for n, batch in enumerate(batches, start=1):
            probs = torch.softmax(self._two_logits([(d, s_) for _, d, s_ in batch], grad=False), dim=-1)[:, 1]
            new = {eid: float(p) for (eid, _, _), p in zip(batch, probs.cpu().tolist(), strict=True)}
            scores.update(new)
            if checkpoint is not None:
                checkpoint.write_many(new)
            if n % every == 0 or n == len(batches):
                self._beat(phase, len(scores), total)
            tick()
        return scores

    def score(self, examples: Sequence[tuple[str, str, str]], tick: Callable[[], None]) -> dict[str, float]:
        phase = "tuned_eval" if getattr(self, "_tuned", False) else "baseline_eval"
        return self._score_batches(score_batches_for(examples, SCORE_BATCH), tick, phase=phase, durable=True)

    # ---- LoRA ------------------------------------------------------------------------------------------------------------------------------
    def _attach_adapter(self) -> None:
        from peft import LoraConfig, get_peft_model

        cfg = self.cfg
        self._seed()  # BEFORE adapter initialisation: LoRA's A matrices are randomly initialised
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=list(cfg.lora_target_modules), bias="none")
        self.model = get_peft_model(self._base, lora)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def _discard_adapter(self) -> None:
        # PeftModel forwards `unload` to the LoRA layer wrapper: it strips the adapter WITHOUT merging it into the base weights.
        self.model = self.model.unload() if hasattr(self.model, "unload") else self._base
        self._base = self.model
        self.model.eval()

    def _run_training(self, examples: Sequence[tuple[str, str, str, int]], plan: Sequence[Any],
                      tick: Callable[[], None]) -> dict[str, int]:
        torch, cfg = self.torch, self.cfg
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        weights = torch.tensor([cfg.weight_unsupported, cfg.weight_supported], device="cuda")
        self.model.train()

        def micro(idx: tuple[int, ...], divisor: int) -> None:
            batch = [examples[j] for j in idx]
            logits = self._two_logits([(b[1], b[2]) for b in batch], grad=True)
            target = torch.tensor([b[3] for b in batch], device="cuda")  # label 0 = unsupported (index 0), 1 = supported
            (torch.nn.functional.cross_entropy(logits, target, weight=weights) / divisor).backward()

        def step(lr_multiplier: float, grad_rescale: float) -> None:
            if grad_rescale != 1.0:  # final partial group: average over the micro-batches actually accumulated
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(grad_rescale)
            for group in opt.param_groups:
                group["lr"] = cfg.learning_rate * lr_multiplier
            opt.step()
            opt.zero_grad(set_to_none=True)

        total_micro = sum(g.size for g in plan)
        done = [0]
        every = max(1, int(getattr(cfg, "heartbeat_every", 10)))

        def beat_tick() -> None:
            done[0] += 1
            if done[0] % every == 0 or done[0] == total_micro:
                self._beat("training", done[0], total_micro)
            tick()

        stats = run_updates(plan, grad_accum=cfg.grad_accum, warmup_fraction=cfg.warmup_fraction, micro_fn=micro,
                            step_fn=step, tick=beat_tick)
        self.model.eval()
        return stats

    def probe(self, train_examples: Sequence[tuple[str, str, str, int]],
              score_examples: Sequence[tuple[str, str, str]]) -> dict[str, float]:
        """Timed scoring plus REAL optimizer updates on a throw-away adapter (discarded afterwards)."""
        cfg = self.cfg
        self._sync()
        t0 = time.perf_counter()
        self.score(score_examples, _noop)
        self._sync()
        score_s = (time.perf_counter() - t0) / max(1, len(score_examples))
        self.torch.cuda.reset_peak_memory_stats()
        self._attach_adapter()
        plan = update_plan(len(train_examples), cfg.micro_batch, cfg.grad_accum, 1, cfg.train_seed)
        self._sync()
        t1 = time.perf_counter()
        stats = self._run_training(train_examples, plan, _noop)
        self._sync()
        train_s = (time.perf_counter() - t1) / max(1, stats["examples_seen"])
        peak_gb = self.torch.cuda.max_memory_allocated() / 1e9
        self._discard_adapter()
        return {"score_s_per_example": score_s, "train_s_per_example": train_s,
                "optimizer_updates_done": stats["optimizer_updates"], "peak_memory_gb": peak_gb}

    def probe_v2(self, plan: Mapping[str, Any]) -> dict[str, float]:
        """Length-stratified probe: production batches at spread positions, real optimizer updates on length-mixed examples, worst-case
        memory on the longest inputs, and a measured save-and-hash of a throw-away adapter. Everything is GPU-synchronised."""
        import tempfile

        cfg = self.cfg
        t_all = time.perf_counter()
        self.torch.cuda.reset_peak_memory_stats()
        batches = plan["score_batches"]
        n_scored = sum(len(b) for b in batches)
        self._sync()
        t0 = time.perf_counter()
        self._score_batches(batches, _noop, phase="probe_score")
        self._sync()
        score_s = (time.perf_counter() - t0) / max(1, n_scored)
        self._score_batches([plan["worst_score_batch"]], _noop, phase="probe_worst_score")  # peak memory on the most token-heavy batch
        self._attach_adapter()
        train_examples = plan["train_examples"]
        upd = update_plan(len(train_examples), cfg.micro_batch, cfg.grad_accum, 1, cfg.train_seed)
        self._sync()
        t1 = time.perf_counter()
        stats = self._run_training(train_examples, upd, _noop)
        self._sync()
        train_s = (time.perf_counter() - t1) / max(1, stats["examples_seen"])
        worst = plan["worst_train_batch"]  # the longest training inputs: forward + backward only, to measure worst-case memory
        self.model.train()
        logits = self._two_logits([(b[1], b[2]) for b in worst], grad=True)
        target = self.torch.tensor([b[3] for b in worst], device="cuda")
        self.torch.nn.functional.cross_entropy(logits, target).backward()
        for prm in self.model.parameters():
            prm.grad = None
        self.model.eval()
        self._sync()
        peak_gb = self.torch.cuda.max_memory_allocated() / 1e9
        t2 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:  # the real save + hash cost, measured once on a throw-away adapter
            self.save_artifacts(Path(tmp))
            from .finalize import REQUIRED_ADAPTER_FILES, file_sha256

            for name in REQUIRED_ADAPTER_FILES:
                file_sha256(Path(tmp) / "adapter" / name)
        save_s = time.perf_counter() - t2
        self._discard_adapter()
        return {"score_s_per_example": score_s, "train_s_per_example": train_s, "optimizer_updates_done": stats["optimizer_updates"],
                "peak_memory_gb": peak_gb, "worst_case_peak_memory_gb": peak_gb, "save_verify_seconds": save_s,
                "probe_examples": {"scoring": n_scored, "training": stats["examples_seen"]},
                "score_probe_tokens": plan.get("score_probe_tokens"), "total_probe_seconds": time.perf_counter() - t_all}

    def train(self, examples: Sequence[tuple[str, str, str, int]], tick: Callable[[], None]) -> None:
        cfg = self.cfg
        self._attach_adapter()
        plan = update_plan(len(examples), cfg.micro_batch, cfg.grad_accum, cfg.epochs, cfg.train_seed)
        self._sync()
        self._run_training(examples, plan, tick)
        self._sync()
        self._tuned = True  # later scoring is the tuned model's

    def save_artifacts(self, out_dir: Path) -> None:
        """Persist the trained adapter (never the base weights) next to the result."""
        if hasattr(self.model, "save_pretrained") and self.model is not self._base:
            self.model.save_pretrained(str(out_dir / "adapter"))

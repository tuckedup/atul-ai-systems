"""Synthetic inputs and child-process entry points used ONLY by the pilot's tests.

`python -m evalops.pilot._testing MODE OUT_DIR CAPS_JSON` runs the real `run_pilot` with a synthetic manifest and a backend that may
deliberately fail to cooperate (it never calls `tick`), so the external watchdog can be tested against a real subprocess.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import finalize as fz
from . import harness as h
from . import run as r
from .config import PILOT, PILOT_V2, PilotConfig, PilotConfigV2
from .watchdog import ProgressFile

COLUMNS = ["example_id", "response_id", "source_id", "partition", "document_fp16", "group_id", "label", "start", "end",
           "implicit_true", "crosses_boundary"]


def make_manifest(train_groups: int = 60, val_groups: int = 30, per_group: int = 8, every: int = 6) -> dict[str, Any]:
    rows, resp_labels, k = [], {}, 0
    for part, n in (("train", train_groups), ("validation", val_groups), ("test", 4)):
        for g in range(n):
            for j in range(per_group):
                label = 0 if k % every == 0 else 1
                rid = f"{part}{g}-r{j // 4}"
                resp_labels[rid] = min(resp_labels.get(rid, 1), label)
                rows.append([f"{part}{g}#{j}", rid, "s", part, "fp", f"g-{part}{g}", label, 0, 5, False, False])
                k += 1
    m: dict[str, Any] = {"version": "ragtruth-training-manifest-v5", "columns": COLUMNS, "rows": rows,
                         "response_labels": resp_labels, "counts": {}, "meta": {}}
    m["manifest_sha256"] = h.content_hash(m)
    return m


def make_cfg(m: dict[str, Any], *, phases: dict[str, int] | None = None, total: int = 1000, startup: int = 400,
             grace: int = 15) -> PilotConfig:
    return dataclasses.replace(
        PILOT, manifest_version=m["version"], manifest_sha256=m["manifest_sha256"], train_unsupported_max=15,
        train_supported_per_unsupported=2, select_eval_sentences=30, validation_eval_sentences=60, bootstrap_draws=100,
        phase_seconds=phases or {"feasibility": 60, "baseline_eval": 200, "training": 400, "tuned_eval": 200, "slack": 140},
        total_seconds=total, startup_seconds=startup, watchdog_grace_seconds=grace)


def make_cfg_v2(m: dict[str, Any], *, phases: dict[str, int] | None = None, total: int = 1000, startup: int = 400,
                grace: int = 15, **overrides: Any) -> PilotConfigV2:
    """A small v2-style configuration for synthetic manifests (stratified probe, response-level sampling, 80% rules)."""
    base = {
        "manifest_version": m["version"], "manifest_sha256": m["manifest_sha256"], "train_unsupported_max": 15,
        "train_supported_per_unsupported": 2, "select_eval_sentences": 30, "validation_eval_sentences": 60,
        "select_responses_per_group": 1, "validation_responses_per_group": 2, "bootstrap_draws": 100,
        "phase_seconds": phases or {"feasibility": 60, "baseline_eval": 200, "training": 400, "tuned_eval": 200, "slack": 140},
        "total_seconds": total, "startup_seconds": startup, "watchdog_grace_seconds": grace, "finalize_allowance_seconds": 10,
    }
    base.update(overrides)
    return dataclasses.replace(PILOT_V2, **base)


def write_inputs(directory: Path, cfg_kwargs: dict[str, Any] | None = None) -> tuple[PilotConfig, dict[str, Any], dict[str, Any]]:
    """Write a synthetic manifest, token lengths and freeze under `directory` in the layout the CLI expects."""
    m = make_manifest()
    cfg = make_cfg(m, **(cfg_kwargs or {}))
    lens = {row[0]: 800 for row in m["rows"]}
    freeze = r.build_freeze(m, lens, cfg)
    (directory / "v5").mkdir(parents=True, exist_ok=True)
    (directory / "v5" / "training_manifest_v5.json").write_text(json.dumps(m), encoding="utf8")
    (directory / "v5" / "pilot_freeze_v3.json").write_text(json.dumps(freeze), encoding="utf8")
    (directory / "token_lens.json").write_text(json.dumps(lens), encoding="utf8")
    return cfg, m, freeze


def write_inputs_v2(directory: Path, cfg_kwargs: dict[str, Any] | None = None) -> tuple[PilotConfigV2, dict[str, Any], dict[str, Any]]:
    """Like `write_inputs`, for a v2-style configuration: the freeze goes where the configuration says."""
    m = make_manifest()
    cfg = make_cfg_v2(m, **(cfg_kwargs or {}))
    lens = {row[0]: 800 for row in m["rows"]}
    freeze = r.build_freeze(m, lens, cfg)
    (directory / "v5").mkdir(parents=True, exist_ok=True)
    (directory / "v5" / "training_manifest_v5.json").write_text(json.dumps(m), encoding="utf8")
    (directory / cfg.freeze_file).write_text(json.dumps(freeze), encoding="utf8")
    (directory / "token_lens.json").write_text(json.dumps(lens), encoding="utf8")
    return cfg, m, freeze


class SleepyBackend:
    """Cooperative-looking backend whose phases really sleep; `hang` names the method that blocks forever WITHOUT calling tick."""

    def __init__(self, labels: dict[str, int], hang: str | None = None) -> None:
        self.labels, self.hang, self.tuned = labels, hang, False
        self.rng = np.random.default_rng(0)

    def _maybe_hang(self, name: str) -> None:
        if self.hang == name:
            time.sleep(3600)

    def probe(self, train_examples, score_examples):
        self._maybe_hang("probe")
        return {"score_s_per_example": 0.001, "train_s_per_example": 0.001, "optimizer_updates_done": 2, "peak_memory_gb": 1.0}

    def probe_v2(self, plan):
        self._maybe_hang("probe")
        return {"score_s_per_example": 0.001, "train_s_per_example": 0.001, "optimizer_updates_done": 2, "peak_memory_gb": 1.0,
                "worst_case_peak_memory_gb": 1.5, "save_verify_seconds": 0.01}

    def score(self, examples, tick):
        if self.tuned:
            self._maybe_hang("tuned_score")
        else:
            self._maybe_hang("baseline_score")
        sep = 0.6 if self.tuned else 0.2
        out = {}
        ck = getattr(self, "checkpoint", None)
        for k, (eid, _d, _s) in enumerate(examples):
            out[eid] = float(np.clip(0.5 + (sep / 2 if self.labels[eid] else -sep / 2) + self.rng.normal(0, 0.12), 0, 1))
            if ck is not None and k % 10 == 9:
                ck.write_many({e: out[e] for e in list(out)[-10:]})
            if self.hang == "midscore" and k == len(examples) // 2:
                if ck is not None:
                    ck.write_many({e: out[e] for e in list(out)[-(len(out) % 10 or 10):]})
                time.sleep(3600)
        tick()
        return out

    def train(self, examples, tick):
        self._maybe_hang("train")
        self.tuned = True

    save_mode = "ok"

    def save_artifacts(self, out_dir: Path) -> None:
        if self.save_mode == "raise":
            raise OSError("disk full while saving the adapter (simulated)")
        if self.save_mode == "hang":
            time.sleep(3600)
        if self.save_mode == "none":
            return  # writes nothing: the required adapter files are missing
        (out_dir / "adapter").mkdir(exist_ok=True)
        for name in fz.REQUIRED_ADAPTER_FILES:
            (out_dir / "adapter" / name).write_text("synthetic adapter file", encoding="utf8")


def child_main(mode: str, out_dir: Path, caps: dict[str, Any]) -> int:
    if mode == "hang_startup":
        time.sleep(3600)
    m = make_manifest()
    cfg_caps = {k: v for k, v in caps.items() if k != "v2"}
    cfg = make_cfg_v2(m, **cfg_caps) if caps.get("v2") else make_cfg(m, **cfg_caps)
    lens = {row[0]: 800 for row in m["rows"]}
    freeze = r.build_freeze(m, lens, cfg)
    rows = h.rows_from_manifest(m)
    hang = mode.removeprefix("hang_") if mode.startswith("hang_") and mode != "hang_finalize" else None
    backend = SleepyBackend({row.example_id: row.label for row in rows}, hang=hang)
    backend.save_mode = {"adapter_fail": "raise", "hang_finalize": "hang", "adapter_missing": "none"}.get(mode, "ok")
    texts = {row.example_id: ("doc", "sentence") for row in rows}
    progress = ProgressFile(out_dir / "progress.json")
    result = r.run_pilot(cfg, m, freeze, lens, texts, backend, require_permission=False, on_event=progress.event, run_dir=out_dir)
    if mode == "malformed":
        (out_dir / fz.RESULT_NAME).write_text("{ this is not json", encoding="utf8")
        return 0
    if mode == "wrong_identity":
        result["manifest_sha256"] = "0" * 64
    fz.finalize(result, backend, out_dir, progress=progress)
    return 7 if mode == "exit7" else 0


if __name__ == "__main__":
    raise SystemExit(child_main(sys.argv[1], Path(sys.argv[2]), json.loads(sys.argv[3])))

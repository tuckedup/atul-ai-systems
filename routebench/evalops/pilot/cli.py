"""Command line for the pilot.

    python -m evalops.pilot.cli freeze  DIR TOKEN_LENS_JSON     # write DIR/v5/pilot_freeze_v3.json (refuses to overwrite)
    python -m evalops.pilot.cli dry-run DIR TOKEN_LENS_JSON     # full path on the real frozen manifest, SIMULATED backend
    python -m evalops.pilot.cli execute DIR TOKEN_LENS_JSON MODEL_DIR --python PY --confirm RUN-GPU-PILOT
                                                                # real run under the external watchdog
    python -m evalops.pilot.cli child   DIR TOKEN_LENS_JSON MODEL_DIR OUT_DIR   # the supervised worker (started by `execute`)

`execute` refuses unless: the permission document records an explicit "accept", the confirmation token is given verbatim, the manifest,
configuration and frozen samples verify, the result directory does not exist, and an isolated interpreter is named explicitly. It runs
the worker as a CHILD PROCESS that the watchdog can kill (startup, per-phase and total caps), always persists a result (a killed or
crashed worker yields status INCOMPLETE), and forces offline model loading. It does not install anything and does not start on its own.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import finalize as fz
from . import harness as h
from . import ledger as lg
from . import run as r
from .config import CONFIGS, PILOT, PilotConfig
from .inputs import load_texts
from .watchdog import ProgressFile, supervise

MANIFEST = "v5/training_manifest_v5.json"
FREEZE = "v5/pilot_freeze_v3.json"
RESULT_DIR = "pilot_run_v1"  # the first pilot's directory; other configurations name their own (`cfg.result_dir`)
CONFIRM_TOKEN = "RUN-GPU-PILOT"
PERMISSION_DOC = Path("docs/ROUTEBENCH_TRAINING_PERMISSION_STATUS.md")
PACKAGES = ("torch", "transformers", "peft", "tokenizers", "accelerate", "safetensors", "numpy")


class SimulatedBackend:
    """Deterministic stand-in with a fake clock; exercises the guards and plumbing, not any model."""

    def __init__(self, labels: dict[str, int]) -> None:
        self.labels, self.tuned, self.t = labels, False, 0.0
        self.rng = np.random.default_rng(0)

    def clock(self) -> float:
        return self.t

    def probe(self, train_examples, score_examples):
        return {"score_s_per_example": 0.4, "train_s_per_example": 1.2, "optimizer_updates_done": PILOT.probe_optimizer_updates,
                "peak_memory_gb": 0.0}

    def probe_v2(self, plan):
        return {"score_s_per_example": 0.4, "train_s_per_example": 1.2, "optimizer_updates_done": PILOT.probe_optimizer_updates,
                "peak_memory_gb": 0.0, "worst_case_peak_memory_gb": 0.0, "save_verify_seconds": 0.5}

    def score(self, examples, tick):
        sep = 0.5 if self.tuned else 0.2
        out = {}
        for eid, _d, _s in examples:
            self.t += 0.4
            tick()
            out[eid] = float(np.clip(0.5 + (sep / 2 if self.labels[eid] else -sep / 2) + self.rng.normal(0, 0.2), 0, 1))
        return out

    def train(self, examples, tick):
        for _ in examples:
            self.t += 1.2
            tick()
        self.tuned = True


def _freeze_path(directory: Path, cfg: PilotConfig) -> Path:
    return directory / getattr(cfg, "freeze_file", FREEZE)


def _result_dir(directory: Path, cfg: PilotConfig) -> Path:
    return directory / getattr(cfg, "result_dir", RESULT_DIR)


def _pop_config(args: list[str]) -> tuple[list[str], PilotConfig]:
    """Extract `--config NAME` (default v1 for read-only commands). `execute`/`child` demand it explicitly."""
    if "--config" in args:
        i = args.index("--config")
        name = args[i + 1]
        if name not in CONFIGS:
            raise SystemExit(f"unknown configuration {name!r}; known: {sorted(CONFIGS)}")
        return args[:i] + args[i + 2:], CONFIGS[name]
    return args, PILOT


def _load(directory: Path, lens_path: Path) -> tuple[dict[str, Any], dict[str, int]]:
    return json.loads((directory / MANIFEST).read_text(encoding="utf8")), json.loads(lens_path.read_text(encoding="utf8"))


def environment_info() -> dict[str, Any]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    info: dict[str, Any] = {"python": sys.version, "platform": platform.platform(), "packages": versions,
                            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE")}
    try:  # only inside the GPU environment
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - environment info must never fail the run
        info["cuda_available"] = None
    return info


# ---------------------------------------------------------------- the supervised worker
def child_main(directory: Path, lens_path: Path, model_dir: Path, out_dir: Path, *,
               backend_factory: Callable[[PilotConfig, Path], Any] | None = None, cfg: PilotConfig = PILOT,
               permission_record: Path = PERMISSION_DOC, require_permission: bool = True) -> dict[str, Any]:
    manifest, lens = _load(directory, lens_path)
    freeze = json.loads(_freeze_path(directory, cfg).read_text(encoding="utf8"))
    wanted = freeze["train_sample"]["example_ids"] + freeze["select_eval"]["example_ids"] + freeze["validation_eval"]["example_ids"]
    texts = load_texts(directory, manifest, wanted)
    if backend_factory is None:
        from .torch_backend import TorchBackend

        backend_factory = TorchBackend
    backend = backend_factory(cfg, model_dir)
    progress = ProgressFile(out_dir / "progress.json")
    try:  # time from process creation to here: interpreter start, imports, text re-derivation and model load
        import psutil

        startup = max(0.0, time.time() - psutil.Process().create_time())
    except Exception:  # noqa: BLE001
        startup = 0.0
    result = r.run_pilot(cfg, manifest, freeze, lens, texts, backend, on_event=progress.event, permission_record=permission_record,
                         require_permission=require_permission, run_dir=out_dir, startup_seconds_measured=startup)
    result["environment"] = environment_info()
    return fz.finalize(result, backend, out_dir, progress=progress)  # artifacts first, then one atomic publication


# ---------------------------------------------------------------- the parent: pre-flight, supervision, persistence
def execute(directory: Path, lens_path: Path, model_dir: Path, *, python: Path | None, confirm: str,
            cfg: PilotConfig = PILOT, permission_record: Path = PERMISSION_DOC, require_permission: bool = True,
            child_argv: Sequence[str] | None = None, poll: float = 0.25) -> dict[str, Any]:
    if confirm != CONFIRM_TOKEN:
        raise SystemExit(f"refusing: pass --confirm {CONFIRM_TOKEN} to start a real run")
    if require_permission:
        r.check_permission_record(permission_record)
    manifest, lens = _load(directory, lens_path)
    freeze = json.loads(_freeze_path(directory, cfg).read_text(encoding="utf8"))
    h.verify_config(cfg, freeze["config_hash"])
    h.verify_manifest(manifest, cfg)
    r.verify_freeze(freeze, manifest, lens, cfg)
    h.assert_wiring(cfg)
    out_dir = _result_dir(directory, cfg)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; a pilot result is never overwritten")
    ledger_path = directory / lg.LEDGER_NAME
    lg.check_budget(ledger_path, cfg.total_seconds)  # cumulative ceiling, including every earlier GPU step
    if child_argv is None:
        if python is None or not python.exists():
            raise SystemExit("name the isolated interpreter explicitly with --python (it must exist)")
        cfg_name = next((k for k, v in CONFIGS.items() if v.config_hash() == cfg.config_hash()), None)
        if cfg_name is None:
            raise SystemExit("the configuration is not a registered one; register it in config.CONFIGS")
        child_argv = [str(python), "-m", "evalops.pilot.cli", "child", str(directory), str(lens_path), str(model_dir), str(out_dir),
                      "--config", cfg_name]
    out_dir.mkdir(parents=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    outcome = supervise(list(child_argv), cfg=cfg, progress_path=out_dir / "progress.json", env=env, log_path=out_dir / "child.log",
                        poll=poll)
    result_path = out_dir / fz.RESULT_NAME
    if not result_path.exists():  # killed by the watchdog, or the worker crashed before publishing anything
        reason = outcome["reason"] or f"the worker exited with code {outcome['exit_code']} without publishing a result"
        r.write_result({"status": "INCOMPLETE" if outcome["killed"] else "FAILED", "incomplete_phase": outcome["phase"],
                        "incomplete_reason": reason, "continue_decision": None, "watchdog": outcome,
                        "config_hash": cfg.config_hash(), "manifest_sha256": manifest["manifest_sha256"],
                        "freeze_sha256": freeze["freeze_sha256"],
                        "note": "an incomplete run yields no comparison and no continue decision"}, out_dir)
    # The parent NEVER trusts the mere existence of a result file: it validates the exit status, identities and artifact hashes.
    verdict, problems = fz.validate_worker_output(out_dir, exit_code=outcome["exit_code"], killed=outcome["killed"], cfg=cfg,
                                                  manifest_sha256=manifest["manifest_sha256"], freeze_sha256=freeze["freeze_sha256"])
    fz.write_terminal(out_dir, verdict, problems, {"watchdog": outcome})
    lg.append(ledger_path, kind="pilot_execute", wall_seconds=outcome["elapsed_seconds"], verdict=verdict,
              note=f"{getattr(cfg, 'name', 'pilot')} under the watchdog")
    outcome.update(result_path=str(result_path), verdict=verdict, problems=problems, cli_exit_code=fz.EXIT_CODES[verdict],
                   ledger_total_hours=round(lg.total_seconds(ledger_path) / 3600, 4))
    return outcome


def main(argv: list[str] | None = None) -> int:
    a = sys.argv[1:] if argv is None else argv
    a, cfg = _pop_config(list(a))
    if a[:1] in (["execute"], ["child"]) and "--config" not in (sys.argv[1:] if argv is None else argv):
        raise SystemExit("`execute` and `child` need an explicit --config (v1 or v2)")
    if len(a) >= 3 and a[0] == "freeze":
        manifest, lens = _load(Path(a[1]), Path(a[2]))
        freeze = r.build_freeze(manifest, lens, cfg)
        target = _freeze_path(Path(a[1]), cfg)
        if target.exists():
            raise SystemExit(f"{target} exists; refusing to overwrite a frozen file")
        target.write_text(json.dumps(freeze), encoding="utf8")
        print(json.dumps({k: freeze[k] for k in ("config_hash", "manifest_sha256", "freeze_sha256")}, indent=1))
        return 0
    if len(a) >= 3 and a[0] == "dry-run":
        manifest, lens = _load(Path(a[1]), Path(a[2]))
        freeze = json.loads(_freeze_path(Path(a[1]), cfg).read_text(encoding="utf8"))
        rows = h.rows_from_manifest(manifest)
        backend = SimulatedBackend({row.example_id: row.label for row in rows})
        texts = {row.example_id: ("document", "sentence") for row in rows}
        res = r.run_pilot(cfg, manifest, freeze, lens, texts, backend, clock=backend.clock, require_permission=False)
        print(json.dumps({k: res[k] for k in ("status", "config_hash", "manifest_sha256", "freeze_sha256", "scope", "feasibility")
                          if k in res}, indent=1, default=float))
        print("SIMULATED backend: nothing above says anything about a real model")
        return 0
    if len(a) >= 5 and a[0] == "child":
        child_main(Path(a[1]), Path(a[2]), Path(a[3]), Path(a[4]), cfg=cfg)
        return 0
    if len(a) >= 4 and a[0] == "execute":
        flags = {a[i]: a[i + 1] for i in range(4, len(a) - 1, 2)}
        outcome = execute(Path(a[1]), Path(a[2]), Path(a[3]), python=Path(flags["--python"]) if "--python" in flags else None,
                          confirm=flags.get("--confirm", ""), cfg=cfg)
        print(json.dumps(outcome, indent=1, default=str))
        return int(outcome["cli_exit_code"])
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

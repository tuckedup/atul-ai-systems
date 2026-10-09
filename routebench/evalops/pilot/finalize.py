"""Terminal-result publication and validation.

A result is published (atomically, last) only after every artifact it names has been written and verified. The PARENT never trusts the
mere existence of a result file: it validates the worker's exit status, the result's identity and contents, and the artifact hashes, and
records its own terminal verdict. A failed finalization produces a non-success status; it can never leave a COMPLETE result behind.

Verdicts (parent side) and command exit codes:
    VALID_COMPLETE  0   a full baseline-vs-tuned comparison whose artifacts verify (the comparison itself may be positive or negative)
    INCOMPLETE      3   killed by the watchdog, a phase overran, or scores were incomplete
    NOT_FEASIBLE    4   the feasibility probe projected a cap overrun; no comparison exists
    FAILED          5   worker crashed, exited non-zero, finalization failed, or the result is malformed / does not match the frozen identities
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import harness as h
from .config import PilotConfig

RESULT_NAME = "pilot_result_v1.json"
TERMINAL_NAME = "pilot_terminal_v1.json"
PREDICTIONS_NAME = "predictions_v1.json"
ADAPTER_DIR = "adapter"
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
WORKER_STATUSES = frozenset({"COMPLETE", "INCOMPLETE", "NOT_FEASIBLE", "FAILED"})
EXIT_CODES = {"VALID_COMPLETE": 0, "INCOMPLETE": 3, "NOT_FEASIBLE": 4, "FAILED": 5}


class Saver(Protocol):
    def save_artifacts(self, out_dir: Path) -> None: ...


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(40):
        try:
            tmp.write_text(text, encoding="utf8")
            os.replace(tmp, path)
            return
        except OSError:
            if attempt == 39:
                raise
            time.sleep(0.025)


def _adapter_hashes(out_dir: Path, required: tuple[str, ...]) -> dict[str, str]:
    root = out_dir / ADAPTER_DIR
    missing = [n for n in required if not (root / n).is_file() or (root / n).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"adapter artifact(s) missing or empty: {missing}")
    return {f"{ADAPTER_DIR}/{n}": file_sha256(root / n) for n in required}


def finalize(result: dict[str, Any], backend: Saver, out_dir: Path, *, progress: Any = None,
             required_adapter: tuple[str, ...] = REQUIRED_ADAPTER_FILES) -> dict[str, Any]:
    """Write and verify artifacts, THEN publish the result atomically. Never publishes COMPLETE if anything failed."""
    if progress is not None:
        progress.event("slack", "start")  # finalization is covered by the watchdog's `slack` phase cap
    try:
        target = out_dir / RESULT_NAME
        if target.exists():
            raise h.PilotError(f"{target} exists; a result is never overwritten")
        predictions = result.pop("predictions", None)
        hashes: dict[str, str] = {}
        try:
            if predictions is not None:
                _atomic_write(out_dir / PREDICTIONS_NAME, json.dumps(predictions, sort_keys=True))
                hashes[PREDICTIONS_NAME] = file_sha256(out_dir / PREDICTIONS_NAME)
            if result.get("status") == "COMPLETE":
                if predictions is None:
                    raise RuntimeError("a COMPLETE run must carry per-example predictions")
                backend.save_artifacts(out_dir)
                hashes.update(_adapter_hashes(out_dir, required_adapter))
        except Exception as e:  # noqa: BLE001 - any finalization failure must withdraw the comparison
            withheld = {k: result.pop(k) for k in ("baseline", "tuned", "paired_gain_bootstrap", "continue_decision")
                        if k in result}
            result.update(status="FAILED", failure={"stage": "finalization", "error": repr(e)}, comparison_withheld=True,
                          withheld_keys=sorted(withheld))
        result["artifact_hashes"] = hashes
        _atomic_write(target, json.dumps(result, indent=1, default=float))
        return result
    finally:
        if progress is not None:
            progress.event("slack", "end")


# ---------------------------------------------------------------- recompute the report from the saved per-example scores
def recompute_report(pred: Mapping[str, Any], manifest: Mapping[str, Any], cfg: PilotConfig) -> dict[str, Any]:
    """Rebuild the headline numbers from `predictions_v1.json` alone (plus the manifest for response-level scope)."""
    rows = pred["rows"]
    sel = [r for r in rows if r["split"] == "train_select"]
    val = [r for r in rows if r["split"] == "validation"]
    y_sel = np.array([r["label"] for r in sel])
    y_val = np.array([r["label"] for r in val])
    out: dict[str, Any] = {}
    scores = {}
    for name in ("baseline", "tuned"):
        s_sel = np.array([r[f"{name}_score"] for r in sel])
        s_val = np.array([r[f"{name}_score"] for r in val])
        thr = h.select_threshold(y_sel, s_sel, cfg)
        scores[name] = (s_val, thr)
        out[name] = {"threshold": thr, "sentence": h.metrics(y_val, s_val, thr)}
    val_rows = [h.Row(r["example_id"], r["response_id"], "validation", r["group_id"], r["label"]) for r in val]
    ci = h.paired_bootstrap_lower(val_rows, y_val, scores["baseline"][0], scores["tuned"][0], scores["baseline"][1],
                                  scores["tuned"][1], cfg)
    out["paired_gain_bootstrap"] = ci
    out["continue_decision"] = h.decide_continue(out["baseline"]["sentence"], out["tuned"]["sentence"], ci["lower95"], cfg,
                                                 coverage_complete=True)
    all_val = [r for r in h.rows_from_manifest(manifest) if r.partition == "validation"]
    for name in ("baseline", "tuned"):
        s_map = {r["example_id"]: r[f"{name}_score"] for r in val}
        out[name]["response"] = h.response_level(all_val, s_map, manifest["response_labels"], scores[name][1])
    return out


# ---------------------------------------------------------------- the parent's validation of what the worker left behind
def validate_worker_output(out_dir: Path, *, exit_code: int | None, killed: bool, cfg: PilotConfig, manifest_sha256: str,
                           freeze_sha256: str, required_adapter: tuple[str, ...] = REQUIRED_ADAPTER_FILES) -> tuple[str, list[str]]:
    """Return (verdict, problems). Existence of the result file alone is never enough."""
    problems: list[str] = []
    path = out_dir / RESULT_NAME
    if killed:
        problems.append("the worker was killed by the watchdog")
    if exit_code not in (0, None) and not killed:
        problems.append(f"the worker exited with code {exit_code}")
    if not path.exists():
        return ("INCOMPLETE" if killed else "FAILED"), problems + ["no result was published"]
    try:
        result = json.loads(path.read_text(encoding="utf8"))
        if not isinstance(result, dict):
            raise TypeError("the result is not a JSON object")
    except (OSError, ValueError, TypeError) as e:
        return "FAILED", problems + [f"malformed result: {e}"]
    status = result.get("status")
    if status not in WORKER_STATUSES:
        problems.append(f"unknown result status {status!r}")
    for key, expected in (("config_hash", cfg.config_hash()), ("manifest_sha256", manifest_sha256), ("freeze_sha256", freeze_sha256)):
        if result.get(key) != expected:
            problems.append(f"{key} does not match the frozen value")
    if status == "COMPLETE" and not problems:
        hashes = result.get("artifact_hashes") or {}
        for need in (PREDICTIONS_NAME, *[f"{ADAPTER_DIR}/{n}" for n in required_adapter]):
            f = out_dir / need
            if need not in hashes:
                problems.append(f"artifact {need} is not listed in the result")
            elif not f.is_file():
                problems.append(f"artifact {need} is missing")
            elif file_sha256(f) != hashes[need]:
                problems.append(f"artifact {need} does not match its recorded hash")
        for key in ("baseline", "tuned", "paired_gain_bootstrap", "continue_decision"):
            if key not in result:
                problems.append(f"COMPLETE result lacks {key}")
    if problems:
        if status in ("INCOMPLETE",) and killed and len(problems) == 1:
            return "INCOMPLETE", problems
        return "FAILED", problems
    return {"COMPLETE": "VALID_COMPLETE", "INCOMPLETE": "INCOMPLETE", "NOT_FEASIBLE": "NOT_FEASIBLE",
            "FAILED": "FAILED"}[status], problems


def write_terminal(out_dir: Path, verdict: str, problems: list[str], extra: Mapping[str, Any]) -> Path:
    target = out_dir / TERMINAL_NAME
    if target.exists():
        raise h.PilotError(f"{target} exists; a terminal record is never overwritten")
    record = {"verdict": verdict, "exit_code": EXIT_CODES[verdict], "problems": problems, **extra}
    _atomic_write(target, json.dumps(record, indent=1, default=str))
    return target

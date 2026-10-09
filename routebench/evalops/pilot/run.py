"""Pilot runner: the ONE execution path that connects the frozen configuration to the guards.

    verify configuration + manifest + frozen samples  ->  feasibility probe  ->  baseline scoring  ->  training  ->  tuned scoring
    ->  metrics  ->  stop rule  ->  result file

Every guard in `harness` is on this path: manifest and configuration hashes, group-disjoint partitions, train-only inputs,
completeness of scores, wall-clock caps (a `tick` is handed to the backend so a running phase can be stopped mid-way), and the
predeclared stop rule. An overrun or an incomplete score set produces a result with status INCOMPLETE and NO metrics that could be
mistaken for a comparison. A below-chance score is reported as a result, never rejected.

The backend is injected. Tests use a fake backend; `torch_backend.TorchBackend` is the real one and is NOT exercised in this
repository's tests (it needs the GPU environment and the training-use permission record).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import harness as h
from .config import PilotConfig

PERMISSION_LINE = re.compile(r"CNN/DM rows for a private research pilot:\s*(accept)\b", re.IGNORECASE)


class PermissionNotRecorded(h.PilotError):
    pass


def check_permission_record(path: Path) -> None:
    """The user's own risk decision must be written in the permission document. It is NOT a rights clearance."""
    text = path.read_text(encoding="utf8") if path.exists() else ""
    if not PERMISSION_LINE.search(text):
        raise PermissionNotRecorded("no 'accept' decision is recorded in the training-use permission document; refusing to run")


class Backend(Protocol):
    def probe(self, train_examples: Sequence[tuple[str, str, str, int]],
              score_examples: Sequence[tuple[str, str, str]]) -> dict[str, float]:
        """Timed probe that performs REAL scoring and REAL optimizer updates on a throw-away adapter.

        Returns {'score_s_per_example', 'train_s_per_example', 'optimizer_updates_done', 'peak_memory_gb'}; GPU timings must be
        synchronised. The adapter is discarded afterwards.
        """

    def score(self, examples: Sequence[tuple[str, str, str]], tick: Callable[[], None]) -> dict[str, float]:
        """(example_id, document, sentence) -> P(supported). May call `tick()` often; it raises when a cap is exceeded."""

    def train(self, examples: Sequence[tuple[str, str, str, int]], tick: Callable[[], None]) -> None:
        """(example_id, document, sentence, label): fit the adapter, then switch the backend to the tuned model."""

    # Optional (duck-typed) for the stratified probe: `probe_v2(plan)` returns the same keys as `probe` plus `worst_case_peak_memory_gb`
    # and `save_verify_seconds`. The runner also sets `backend.progress` (a ProgressLog) and `backend.checkpoint` (a Checkpointer for the
    # current scoring phase); a backend that supports them logs heartbeats and writes durable partial scores.


def build_freeze(manifest: Mapping[str, Any], token_lens: Mapping[str, int], cfg: PilotConfig) -> dict[str, Any]:
    """The frozen samples and hashes for a manifest and per-example token lengths."""
    h.verify_manifest(manifest, cfg)
    rows = h.rows_from_manifest(manifest)
    plan = h.make_plan(rows, cfg)
    train = h.training_sample(plan, token_lens, cfg)
    if getattr(cfg, "select_responses_per_group", 0):
        sel = h.evaluation_sample_by_response(plan.train_select, token_lens, cfg.select_eval_sentences,
                                              cfg.select_responses_per_group, cfg, "select")
        val = h.evaluation_sample_by_response(plan.validation, token_lens, cfg.validation_eval_sentences,
                                              cfg.validation_responses_per_group, cfg, "validation")
    else:
        sel = h.evaluation_sample(plan.train_select, token_lens, cfg.select_eval_sentences, cfg, "select")
        val = h.evaluation_sample(plan.validation, token_lens, cfg.validation_eval_sentences, cfg, "validation")
    h.assert_training_inputs(train["example_ids"], plan, rows)
    body = {"config_hash": cfg.config_hash(), "manifest_sha256": manifest["manifest_sha256"], "train_sample": train,
            "select_eval": sel, "validation_eval": val, "max_input_tokens": cfg.max_input_tokens}
    body["freeze_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def verify_freeze(freeze: Mapping[str, Any], manifest: Mapping[str, Any], token_lens: Mapping[str, int], cfg: PilotConfig) -> None:
    rebuilt = build_freeze(manifest, token_lens, cfg)
    if rebuilt != dict(freeze):
        raise h.ConfigMismatch("the frozen samples do not match the manifest, token lengths and configuration")


def _incomplete(result: dict[str, Any], phase: str, reason: str) -> dict[str, Any]:
    result.update(status="INCOMPLETE", incomplete_phase=phase, incomplete_reason=reason, continue_decision=None,
                  note="an incomplete run yields no comparison and no continue decision; partial scores (if any) are diagnostic only")
    return result


def _attach_partial(result: dict[str, Any], checkpoints: Mapping[str, h.Checkpointer]) -> dict[str, Any]:
    """Record how many scores were durably checkpointed before a stop. Partial scores never authorise continuation."""
    if checkpoints:
        result["partial_scores"] = {phase: {"scored": c.count, "file": c.path.name} for phase, c in checkpoints.items()}
        result["partial_scores_note"] = "diagnostic only: an INCOMPLETE run can never authorise continuation"
    return result


def run_pilot(cfg: PilotConfig, manifest: Mapping[str, Any], freeze: Mapping[str, Any], token_lens: Mapping[str, int],
              texts: Mapping[str, tuple[str, str]], backend: Backend, *, clock: Callable[[], float] = time.monotonic,
              frozen_config_hash: str | None = None, permission_record: Path | None = None,
              require_permission: bool = True, on_event: Callable[[str, str], None] | None = None,
              run_dir: Path | None = None, startup_seconds_measured: float = 0.0) -> dict[str, Any]:
    if require_permission:
        check_permission_record(permission_record or Path("docs/ROUTEBENCH_TRAINING_PERMISSION_STATUS.md"))
    h.verify_config(cfg, frozen_config_hash or freeze["config_hash"])
    h.verify_manifest(manifest, cfg)
    verify_freeze(freeze, manifest, token_lens, cfg)
    h.assert_wiring(cfg)

    rows = h.rows_from_manifest(manifest)
    plan = h.make_plan(rows, cfg)
    by_id = {r.example_id: r for r in rows}
    train_ids, sel_ids, val_ids = (freeze["train_sample"]["example_ids"], freeze["select_eval"]["example_ids"],
                                   freeze["validation_eval"]["example_ids"])
    h.assert_training_inputs(train_ids, plan, rows)
    missing = [i for i in train_ids + sel_ids + val_ids if i not in texts]
    if missing:
        raise h.IncompleteEvaluation(f"{len(missing)} planned examples have no text")

    def trip(ids: Sequence[str]) -> list[tuple[str, str, str]]:
        return [(i, texts[i][0], texts[i][1]) for i in ids]

    train_ex = [(i, texts[i][0], texts[i][1], by_id[i].label) for i in train_ids]
    result: dict[str, Any] = {
        "status": "COMPLETE", "config_hash": cfg.config_hash(), "manifest_sha256": manifest["manifest_sha256"],
        "freeze_sha256": freeze["freeze_sha256"], "scope": {
            "train_examples": len(train_ids), "select_eval_examples": len(sel_ids), "validation_eval_examples": len(val_ids),
            "validation_groups": len(freeze["validation_eval"]["groups"]),
            "excluded_over_length": {"train_fit": freeze["train_sample"]["excluded_over_length"],
                                     "select": freeze["select_eval"]["excluded_over_length"],
                                     "validation": freeze["validation_eval"]["excluded_over_length"]},
            "note": ("length-filtered, label-blind sample of validation DOCUMENT GROUPS from RAGTruth news summaries; "
                     "not the RouteBench benchmark and not full coverage of the RAGTruth validation partition")},
        "permission_note": "a recorded 'accept' is the user's own risk decision, not a rights clearance"}
    budget = h.ResourceBudget(cfg, clock, on_event=on_event)

    def guarded(phase: str, fn: Callable[[Callable[[], None]], Any]) -> Any:
        with budget.phase(phase) as tick:
            return fn(tick)

    # ---- feasibility ---------------------------------------------------------------------------------------------------------------
    need = cfg.micro_batch * cfg.grad_accum * cfg.probe_optimizer_updates  # examples that yield the required optimizer updates
    if len(train_ex) < need:
        raise h.IncompleteEvaluation(f"the training sample ({len(train_ex)}) is smaller than the probe needs ({need})")
    stratified = getattr(cfg, "probe_mode", "legacy") == "stratified"
    n_eval = len(sel_ids) + len(val_ids)
    checkpoints: dict[str, h.Checkpointer] = {}
    if run_dir is not None and hasattr(backend, "probe_v2"):
        backend.progress = h.ProgressLog(run_dir / "progress_detail.jsonl", clock=clock,
                                         vram_fn=getattr(backend, "vram_snapshot", None))
    probe_started = clock()
    try:
        if stratified:
            plan_p = h.probe_plan(trip(sel_ids + val_ids), train_ex, token_lens, cfg)
            probe = guarded("feasibility", lambda tick: backend.probe_v2(plan_p))
        else:
            probe_score = trip(sel_ids[: cfg.probe_score_examples])
            probe = guarded("feasibility", lambda tick: backend.probe(train_ex[:need], probe_score))
    except h.BudgetExceeded as e:
        return _incomplete(result, "feasibility", str(e))
    probe_seconds = clock() - probe_started
    if probe.get("optimizer_updates_done", 0) < cfg.probe_optimizer_updates:
        result.update(status="NOT_FEASIBLE", continue_decision=None, feasibility={"probe": probe, "feasible": False},
                      note="the probe did not perform the required optimizer updates, so training time and memory are unmeasured")
        result["seconds_by_phase"] = dict(budget.spent)
        return result
    proj = {"training": probe["train_s_per_example"] * len(train_ids), "baseline_eval": probe["score_s_per_example"] * n_eval,
            "tuned_eval": probe["score_s_per_example"] * n_eval}
    fraction = getattr(cfg, "feasibility_fraction", 1.0)
    remaining = {k: budget.remaining(k) for k in proj}
    fits = h.fits_fraction(proj, remaining, fraction)
    overhead = {"startup_seconds_measured": startup_seconds_measured, "probe_seconds": probe_seconds,
                "finalize_allowance_seconds": getattr(cfg, "finalize_allowance_seconds", 0),
                "save_verify_seconds_measured": probe.get("save_verify_seconds", 0.0)}
    overall = (overhead["startup_seconds_measured"] + overhead["probe_seconds"] + sum(proj.values())
               + max(overhead["finalize_allowance_seconds"], 3 * overhead["save_verify_seconds_measured"]))
    overall_limit = getattr(cfg, "overall_fraction", 1.0) * cfg.total_seconds
    result["feasibility"] = {
        "probe": probe, "projected_seconds": proj, "caps": dict(remaining), "fraction_rule": fraction, "fits_fraction": fits,
        "overhead": overhead, "projected_total_seconds": overall, "overall_limit_seconds": overall_limit,
        "feasible": all(fits.values()) and overall <= overall_limit,
        "sample": {"train": len(train_ids), "select": len(sel_ids), "validation": len(val_ids)}}
    if not result["feasibility"]["feasible"]:
        result.update(status="NOT_FEASIBLE", continue_decision=None,
                      note=f"projected time exceeds {fraction:.0%} of a phase budget or {getattr(cfg, 'overall_fraction', 1.0):.0%} "
                           "of the total; stopped before training")
        result["seconds_by_phase"] = dict(budget.spent)
        return result

    # ---- baseline scoring ---------------------------------------------------------------------------------------------------------------
    def score_all(phase: str) -> tuple[np.ndarray, np.ndarray]:
        if run_dir is not None and hasattr(backend, "probe_v2"):
            checkpoints[phase] = h.Checkpointer(run_dir, phase)
            backend.checkpoint = checkpoints[phase]  # durable partial scores for this phase
        raw = guarded(phase, lambda tick: backend.score(trip(sel_ids + val_ids), tick))
        s = h.check_scores(sel_ids + val_ids, raw)
        return s[: len(sel_ids)], s[len(sel_ids):]

    y_sel = np.array([by_id[i].label for i in sel_ids])
    y_val = np.array([by_id[i].label for i in val_ids])
    try:
        base_sel, base_val = score_all("baseline_eval")
    except (h.BudgetExceeded, h.IncompleteEvaluation) as e:
        return _attach_partial(_incomplete(result, "baseline_eval", str(e)), checkpoints)
    t_base = h.select_threshold(y_sel, base_sel, cfg)

    # ---- training -------------------------------------------------------------------------------------------------------------------------
    try:
        guarded("training", lambda tick: backend.train(train_ex, tick))
    except h.BudgetExceeded as e:
        return _attach_partial(_incomplete(result, "training", str(e)), checkpoints)
    try:
        tuned_sel, tuned_val = score_all("tuned_eval")
    except (h.BudgetExceeded, h.IncompleteEvaluation) as e:
        return _attach_partial(_incomplete(result, "tuned_eval", str(e)), checkpoints)
    t_tuned = h.select_threshold(y_sel, tuned_sel, cfg)

    # ---- metrics and the predeclared decision --------------------------------------------------------------------------------------------------------
    val_rows = [by_id[i] for i in val_ids]
    all_val_rows = list(plan.validation)
    labels = manifest["response_labels"]

    def block(sel: np.ndarray, val: np.ndarray, thr: float) -> dict[str, Any]:
        scores_by_id = dict(zip(val_ids, val, strict=True))
        return {"threshold_selected_on_train_select": thr, "sentence": h.metrics(y_val, val, thr),
                "sentence_at_native_0.5": h.metrics(y_val, val, 0.5), "direction_on_validation": h.direction_report(y_val, val),
                "direction_on_train_select": h.direction_report(y_sel, sel),
                "response": h.response_level(all_val_rows, scores_by_id, labels, thr)}

    base_block, tuned_block = block(base_sel, base_val, t_base), block(tuned_sel, tuned_val, t_tuned)
    ci = h.paired_bootstrap_lower(val_rows, y_val, base_val, tuned_val, t_base, t_tuned, cfg)
    decision = h.decide_continue(base_block["sentence"], tuned_block["sentence"], ci["lower95"], cfg, coverage_complete=True)
    prediction_rows = []
    for split, ids, base_s, tuned_s in (("train_select", sel_ids, base_sel, tuned_sel), ("validation", val_ids, base_val, tuned_val)):
        for eid, b, t in zip(ids, base_s, tuned_s, strict=True):
            row = by_id[eid]
            prediction_rows.append({"example_id": eid, "response_id": row.response_id, "group_id": row.group_id, "split": split,
                                    "label": row.label, "baseline_score": float(b), "tuned_score": float(t)})
    group_split = {row.group_id: name for name, part in (("train_fit", plan.train_fit), ("train_select", plan.train_select),
                                                           ("validation", plan.validation)) for row in part}
    result.update(baseline=base_block, tuned=tuned_block, paired_gain_bootstrap=ci, continue_decision=decision,
                  seconds_by_phase=dict(budget.spent), total_seconds=budget.total(),
                  predictions={"rows": prediction_rows, "thresholds": {"baseline": t_base, "tuned": t_tuned},
                               "train_sample_example_ids": list(train_ids), "group_split": group_split,
                               "hashes": {"config": cfg.config_hash(), "manifest": manifest["manifest_sha256"],
                                          "freeze": freeze["freeze_sha256"]}})
    return result


def write_result(result: Mapping[str, Any], out_dir: Path, name: str = "pilot_result_v1.json") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    if target.exists():
        raise h.PilotError(f"{target} exists; a result is never overwritten")
    target.write_text(json.dumps(result, indent=1, default=float), encoding="utf8")
    return target

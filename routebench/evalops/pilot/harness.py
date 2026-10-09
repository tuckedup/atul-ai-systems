"""Guards for the bounded pilot: everything that can silently invalidate the experiment fails loudly here.

Pure Python plus numpy; no model, no GPU, no network. Covered by synthetic tests.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PilotConfig


class PilotError(RuntimeError):
    pass


class ManifestMismatch(PilotError):
    pass


class PartitionError(PilotError):
    pass


class IncompleteEvaluation(PilotError):
    pass


class ScoreDirectionError(PilotError):
    pass


class BudgetExceeded(PilotError):
    pass


class ConfigMismatch(PilotError):
    pass


# ---------------------------------------------------------------- manifest and configuration integrity
def content_hash(manifest: Mapping[str, Any]) -> str:
    """The hash `ingest_audit.build_manifest` embeds: SHA-256 of the manifest content without that field."""
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def verify_manifest(manifest: Mapping[str, Any], cfg: PilotConfig) -> None:
    if not cfg.manifest_sha256:
        raise ManifestMismatch("the pilot configuration has no frozen manifest hash")
    if manifest.get("version") != cfg.manifest_version:
        raise ManifestMismatch(f"manifest version {manifest.get('version')!r} != {cfg.manifest_version!r}")
    if manifest.get("manifest_sha256") != content_hash(manifest):
        raise ManifestMismatch("the manifest content does not match its embedded hash (it was modified)")
    if manifest["manifest_sha256"] != cfg.manifest_sha256:
        raise ManifestMismatch("the manifest hash differs from the hash frozen in the pilot configuration")


def verify_config(cfg: PilotConfig, recorded_hash: str) -> None:
    if cfg.config_hash() != recorded_hash:
        raise ConfigMismatch("the pilot configuration changed after it was frozen")


# ---------------------------------------------------------------- partition plan
@dataclass(frozen=True)
class Row:
    example_id: str
    response_id: str
    partition: str
    group_id: str
    label: int


@dataclass(frozen=True)
class Plan:
    train_fit: tuple[Row, ...]
    train_select: tuple[Row, ...]
    validation: tuple[Row, ...]

    def fit_ids(self) -> frozenset[str]:
        return frozenset(r.example_id for r in self.train_fit)


def rows_from_manifest(manifest: Mapping[str, Any]) -> list[Row]:
    c = manifest["columns"]
    i = {k: c.index(k) for k in ("example_id", "response_id", "partition", "group_id", "label")}
    return [Row(r[i["example_id"]], r[i["response_id"]], r[i["partition"]], r[i["group_id"]], int(r[i["label"]]))
            for r in manifest["rows"]]


def _bucket(group_id: str, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def make_plan(rows: Sequence[Row], cfg: PilotConfig) -> Plan:
    """TRAIN is split by near-duplicate GROUP into fit / select; the RAGTruth test partition is never touched."""
    fit, sel, val = [], [], []
    for r in rows:
        if r.partition == "train":
            (sel if _bucket(r.group_id, cfg.select_seed) < cfg.select_fraction else fit).append(r)
        elif r.partition == "validation":
            val.append(r)
    plan = Plan(tuple(fit), tuple(sel), tuple(val))
    seen: dict[str, str] = {}
    for name, part in (("train_fit", plan.train_fit), ("train_select", plan.train_select), ("validation", plan.validation)):
        for r in part:
            if seen.setdefault(r.group_id, name) != name:
                raise PartitionError(f"group {r.group_id} appears in both {seen[r.group_id]} and {name}")
    if not plan.train_fit or not plan.train_select or not plan.validation:
        raise PartitionError("a partition of the plan is empty")
    return plan


def assert_training_inputs(example_ids: Sequence[str], plan: Plan, rows: Sequence[Row]) -> None:
    """Reject anything that is not in `train_fit`: held-out (select / validation / test) and unknown ids alike."""
    by_id = {r.example_id: r for r in rows}
    fit = plan.fit_ids()
    for eid in example_ids:
        if eid not in fit:
            where = by_id[eid].partition if eid in by_id else "unknown"
            raise PartitionError(f"example {eid} is not in train_fit (partition: {where}); refusing to train on it")


def _shuffled(items: Sequence[Row], seed: int, tag: str) -> list[Row]:
    return sorted(items, key=lambda r: hashlib.sha256(f"{seed}:{tag}:{r.example_id}".encode()).hexdigest())


def training_sample(plan: Plan, token_lens: Mapping[str, int], cfg: PilotConfig) -> dict[str, Any]:
    """The fixed training sample: over-length examples excluded (counted, never truncated), all unsupported up to the cap, and
    `train_supported_per_unsupported` supported examples per unsupported one, by seeded shuffle from train_fit only."""
    usable = [r for r in plan.train_fit if token_lens.get(r.example_id, cfg.max_input_tokens + 1) <= cfg.max_input_tokens]
    unsup = _shuffled([r for r in usable if r.label == 0], cfg.sample_seed, "u")[: cfg.train_unsupported_max]
    if not unsup:
        raise PartitionError("train_fit has no usable unsupported examples")
    sup = _shuffled([r for r in usable if r.label == 1], cfg.sample_seed, "s")[: cfg.train_supported_per_unsupported * len(unsup)]
    ids = sorted(r.example_id for r in unsup + sup)
    return {"example_ids": ids, "n_unsupported": len(unsup), "n_supported": len(sup),
            "excluded_over_length": len(plan.train_fit) - len(usable)}


def evaluation_sample(part: Sequence[Row], token_lens: Mapping[str, int], target_sentences: int, cfg: PilotConfig,
                      tag: str) -> dict[str, Any]:
    """Label-blind sample of whole document groups, in seeded random order, until `target_sentences` in-length sentences are
    reached. Groups (never single sentences) are the sampling unit, so the sample keeps the document structure."""
    by_group: dict[str, list[Row]] = defaultdict(list)
    excluded = 0
    for r in part:
        if token_lens.get(r.example_id, cfg.max_input_tokens + 1) <= cfg.max_input_tokens:
            by_group[r.group_id].append(r)
        else:
            excluded += 1
    order = sorted(by_group, key=lambda g: hashlib.sha256(f"{cfg.sample_seed}:{tag}:{g}".encode()).hexdigest())
    chosen: list[str] = []
    ids: list[str] = []
    for g in order:
        if len(ids) >= target_sentences:
            break
        chosen.append(g)
        ids.extend(r.example_id for r in by_group[g])
    return {"example_ids": sorted(ids), "groups": sorted(chosen), "excluded_over_length": excluded}


# ---------------------------------------------------------------- scores and metrics (unsupported = positive)
def check_scores(expected_ids: Sequence[str], scores: Mapping[str, float]) -> np.ndarray:
    missing = [i for i in expected_ids if i not in scores]
    extra = set(scores) - set(expected_ids)
    if missing or extra:
        raise IncompleteEvaluation(f"{len(missing)} expected ids missing, {len(extra)} unexpected ids present")
    arr = np.array([scores[i] for i in expected_ids], dtype=float)
    if not np.all(np.isfinite(arr)) or arr.min() < 0.0 or arr.max() > 1.0:
        raise IncompleteEvaluation("scores must be finite probabilities in [0, 1]")
    return arr


def _auc_supported(y_supported: np.ndarray, score: np.ndarray) -> float:
    """Probability that a random supported example outscores a random unsupported one (ties count half)."""
    pos, neg = score[y_supported == 1], score[y_supported == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def direction_report(y_supported: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    """Empirical direction of a score. Below-chance is a RESULT to report (a wired-correctly model can be bad), never an error."""
    auc = _auc_supported(y_supported, score)
    return {"auc_supported_higher": auc, "below_chance": bool(auc < 0.5)}


def score_from_logits(unsupported_logit: float, supported_logit: float) -> float:
    """P(supported): softmax over the two label-token logits, index 1 (the declared scoring path)."""
    m = max(unsupported_logit, supported_logit)
    eu, es = math.exp(unsupported_logit - m), math.exp(supported_logit - m)
    return es / (eu + es)


def build_input(document: str, sentence: str, cfg: PilotConfig, *, eos: str) -> str:
    """The declared model input: prefix + whole document + EOS + sentence. No truncation happens here or in the caller."""
    return f"{cfg.prompt_prefix}{document}{eos}{sentence}"


def assert_wiring(cfg: PilotConfig) -> None:
    """Deterministic checks of the scoring path itself (not of any model's quality)."""
    if cfg.unsupported_token_id == cfg.supported_token_id:
        raise ScoreDirectionError("the unsupported and supported label tokens are the same id")
    if not score_from_logits(-3.0, 3.0) > 0.5 > score_from_logits(3.0, -3.0):
        raise ScoreDirectionError("score_from_logits does not return P(supported)")
    if abs(score_from_logits(1.0, 1.0) - 0.5) > 1e-12:
        raise ScoreDirectionError("equal logits must score exactly 0.5")


def confusion(y_supported: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, int]:
    pred_unsup = score < threshold
    true_unsup = y_supported == 0
    return {"tp": int((pred_unsup & true_unsup).sum()), "fp": int((pred_unsup & ~true_unsup).sum()),
            "fn": int((~pred_unsup & true_unsup).sum()), "tn": int((~pred_unsup & ~true_unsup).sum())}


def balanced_accuracy(c: Mapping[str, int]) -> float:
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
    spec = c["tn"] / (c["tn"] + c["fp"]) if c["tn"] + c["fp"] else float("nan")
    return (rec + spec) / 2


def average_precision_unsupported(y_supported: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC with UNSUPPORTED as the positive class and a LOW score as the detection signal.

    Tied scores form one threshold step (precision and recall are read after the whole tie group), so the value does not depend on
    the order of rows. Equals scikit-learn's average precision computed on (1 - y_supported, 1 - score).
    """
    pos = (y_supported == 0).astype(int)
    total = int(pos.sum())
    if total == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")  # ascending: lowest score = most likely unsupported
    sc, hits = score[order], pos[order]
    tp = np.cumsum(hits)
    k = np.arange(1, len(hits) + 1)
    last = np.r_[np.flatnonzero(np.diff(sc) != 0), len(sc) - 1]  # last row of each distinct score
    precision, recall = tp[last] / k[last], tp[last] / total
    prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))


def kappa(c: Mapping[str, int]) -> float:
    n = sum(c.values())
    po = (c["tp"] + c["tn"]) / n
    pe = ((c["tp"] + c["fp"]) * (c["tp"] + c["fn"]) + (c["fn"] + c["tn"]) * (c["fp"] + c["tn"])) / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def metrics(y_supported: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    c = confusion(y_supported, score, threshold)
    return {"threshold": threshold, "balanced_accuracy": balanced_accuracy(c),
            "unsupported_recall": c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan"),
            "unsupported_precision": c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan"),
            "pr_auc_unsupported": average_precision_unsupported(y_supported, score), "kappa": kappa(c), **c}


def select_threshold(y_supported: np.ndarray, score: np.ndarray, cfg: PilotConfig) -> float:
    """Balanced-accuracy-optimal threshold on the SELECTION data only; ties resolve toward 0.5."""
    best = max(cfg.threshold_grid,
               key=lambda t: (round(balanced_accuracy(confusion(y_supported, score, t)), 12), -abs(t - 0.5)))
    return float(best)


def response_level(rows: Sequence[Row], scores: Mapping[str, float], response_labels: Mapping[str, int], threshold: float) -> dict[str, Any]:
    """Whole-response metrics, only for responses whose EVERY in-scope sentence was evaluated.

    `rows` are all in-scope rows of the partition (before any sampling or length exclusion); `response_labels` are the AUTHORITATIVE
    response labels carried by the manifest (they include annotated wrapper segments). Response score = minimum sentence score.
    Responses that were only partly evaluated are counted and reported, never scored as if complete.
    """
    by_response: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_response[r.response_id].append(r)
    complete, partial, none_seen = [], 0, 0
    for rid, rs in by_response.items():
        seen = sum(1 for r in rs if r.example_id in scores)
        if seen == len(rs):
            complete.append(rid)
        elif seen:
            partial += 1
        else:
            none_seen += 1
    out: dict[str, Any] = {"scope": "complete responses only", "n_complete": len(complete), "n_partial": partial,
                           "n_not_evaluated": none_seen, "metrics": None}
    if complete:
        ids = sorted(complete)
        resp_score = np.array([min(scores[r.example_id] for r in by_response[i]) for i in ids])
        labels = np.array([int(response_labels[i]) for i in ids])
        out["metrics"] = metrics(labels, resp_score, threshold)
    return out


def paired_bootstrap_lower(rows: Sequence[Row], y: np.ndarray, base: np.ndarray, tuned: np.ndarray, t_base: float,
                           t_tuned: float, cfg: PilotConfig) -> dict[str, float]:
    """Document-group bootstrap of (tuned - baseline) balanced accuracy; groups, not sentences, are resampled."""
    groups = sorted({r.group_id for r in rows})
    idx: dict[str, list[int]] = defaultdict(list)
    for k, r in enumerate(rows):
        idx[r.group_id].append(k)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    diffs = []
    for _ in range(cfg.bootstrap_draws):
        ks = np.array([k for g in rng.choice(groups, len(groups)) for k in idx[g]])
        d = balanced_accuracy(confusion(y[ks], tuned[ks], t_tuned)) - balanced_accuracy(confusion(y[ks], base[ks], t_base))
        if not math.isnan(d):
            diffs.append(d)
    return {"lower95": float(np.percentile(diffs, 2.5)), "upper95": float(np.percentile(diffs, 97.5)),
            "mean": float(np.mean(diffs)), "draws_used": len(diffs)}


def decide_continue(base: Mapping[str, float], tuned: Mapping[str, float], ci_lower: float, cfg: PilotConfig,
                    coverage_complete: bool) -> dict[str, Any]:
    """The predeclared stop rule. Anything incomplete is a stop, never a pass."""
    gain = tuned["balanced_accuracy"] - base["balanced_accuracy"]
    recall_ok = tuned["unsupported_recall"] >= base["unsupported_recall"] - cfg.max_recall_drop
    checks = {"complete_coverage": coverage_complete, "gain_at_least_min": gain >= cfg.min_improvement,
              "ci_lower_above_zero": ci_lower > 0.0, "recall_not_worse": recall_ok}
    return {"continue": all(checks.values()), "gain": gain, "checks": checks}


# ---------------------------------------------------------------- resource caps
class ResourceBudget:
    """Wall-clock accounting with a hard total cap. An overrun raises; it is never absorbed."""

    def __init__(self, cfg: PilotConfig, clock: Callable[[], float] = time.monotonic,
                 on_event: Callable[[str, str], None] | None = None) -> None:
        self.cfg, self._clock, self._on_event = cfg, clock, on_event
        self.spent: dict[str, float] = defaultdict(float)

    def total(self) -> float:
        return sum(self.spent.values())

    def remaining(self, phase: str) -> float:
        cap = self.cfg.phase_seconds[phase]
        return max(0.0, min(cap - self.spent[phase], self.cfg.total_seconds - self.total()))

    @contextmanager
    def phase(self, name: str):
        if name not in self.cfg.phase_seconds:
            raise BudgetExceeded(f"unknown phase {name!r}")
        if self.remaining(name) <= 0:
            raise BudgetExceeded(f"no budget left for phase {name!r}")
        start = self._clock()
        if self._on_event:
            self._on_event(name, "start")  # lets an EXTERNAL watchdog enforce this phase's cap
        try:
            yield lambda: self.check(name, start)
        finally:
            self.spent[name] += self._clock() - start
            if self._on_event:
                self._on_event(name, "end")
        if self.spent[name] > self.cfg.phase_seconds[name] or self.total() > self.cfg.total_seconds:
            raise BudgetExceeded(f"phase {name!r} or the total cap was exceeded")

    def check(self, name: str, start: float) -> None:
        """Call inside loops: raises as soon as the running phase or the total would exceed its cap."""
        elapsed = self._clock() - start
        if self.spent[name] + elapsed > self.cfg.phase_seconds[name]:
            raise BudgetExceeded(f"phase {name!r} exceeded {self.cfg.phase_seconds[name]} s")
        if self.total() + elapsed > self.cfg.total_seconds:
            raise BudgetExceeded(f"total cap of {self.cfg.total_seconds} s exceeded")


def run_with_timeout(cmd: Sequence[str], seconds: float) -> subprocess.CompletedProcess[str]:
    """Run a subprocess under a real wall-clock timeout; on expiry the process is killed and BudgetExceeded is raised."""
    try:
        return subprocess.run(list(cmd), capture_output=True, text=True, timeout=seconds, check=False)
    except subprocess.TimeoutExpired as e:
        raise BudgetExceeded(f"command exceeded {seconds} s and was killed") from e


# ---------------------------------------------------------------- pilot v2: stratified probe, response-level sampling, monitoring
SCORE_BATCH = 4


def score_batches_for(examples: Sequence[tuple[str, str, str]], batch_size: int = SCORE_BATCH) -> list[list[tuple[str, str, str]]]:
    """The production scoring order: sorted by input length (document + sentence characters), cut into fixed-size batches.

    Used by the real backend AND by the probe planner, so the probe exercises exactly the batching the run will use.
    """
    order = sorted(range(len(examples)), key=lambda i: (len(examples[i][1]) + len(examples[i][2]), examples[i][0]))
    return [[examples[i] for i in order[s:s + batch_size]] for s in range(0, len(order), batch_size)]


def systematic_positions(n_items: int, k: int) -> list[int]:
    """k deterministic, evenly spread positions in [0, n_items): an unbiased systematic sample of a sorted sequence."""
    k = min(k, n_items)
    return sorted({min(n_items - 1, int((i + 0.5) * n_items / k)) for i in range(k)})


def probe_plan(score_examples: Sequence[tuple[str, str, str]], train_examples: Sequence[tuple[str, str, str, int]],
               token_lens: Mapping[str, int], cfg: Any) -> dict[str, Any]:
    """A deterministic, length-stratified feasibility probe with real batching and worst-case inputs.

    Scoring: `probe_score_batches` production batches at evenly spaced positions of the length-sorted batch sequence (so their mean time is
    an unbiased estimate of the mean batch time), plus the single most token-heavy batch for peak memory. Training: `n` examples at evenly
    spaced positions of the training sample sorted by token length, shuffled by a seeded permutation so micro-batches have the
    mixed lengths production has, plus the longest training examples as a worst-case micro-batch for peak memory.
    """
    batches = score_batches_for(score_examples)
    pos = systematic_positions(len(batches), getattr(cfg, "probe_score_batches", 8))
    heavy = max(range(len(batches)), key=lambda i: sum(token_lens[e[0]] for e in batches[i]))
    need = cfg.micro_batch * cfg.grad_accum * cfg.probe_optimizer_updates
    by_len = sorted(train_examples, key=lambda e: (token_lens[e[0]], e[0]))
    chosen = [by_len[p] for p in systematic_positions(len(by_len), need)]
    order = np.random.default_rng(cfg.train_seed).permutation(len(chosen)).tolist()
    return {
        "score_batches": [batches[p] for p in pos], "worst_score_batch": batches[heavy],
        "train_examples": [chosen[i] for i in order], "worst_train_batch": by_len[-cfg.micro_batch:],
        "score_positions": pos, "n_score_batches_total": len(batches),
        "score_probe_tokens": [sum(token_lens[e[0]] for e in batches[p]) for p in pos],
    }


def evaluation_sample_by_response(part: Sequence[Row], token_lens: Mapping[str, int], coverage_sentences: int,
                                  responses_per_group: int, cfg: PilotConfig, tag: str) -> dict[str, Any]:
    """Keep the documents (groups) that `evaluation_sample` would keep, but only `responses_per_group` WHOLE responses of each.

    Predetermined and label/prediction-blind: the groups come from the same seeded group order as the coverage sample; within a group,
    responses are taken in seeded hash order. A response is either kept with all of its in-length sentences or dropped, so
    response-level metrics stay meaningful and no difficult example is ever removed by looking at labels or scores.
    """
    base = evaluation_sample(part, token_lens, coverage_sentences, cfg, tag)
    keep_groups = set(base["groups"])
    by_group: dict[str, dict[str, list[Row]]] = defaultdict(lambda: defaultdict(list))
    excluded = 0
    for r in part:
        if r.group_id not in keep_groups:
            continue
        if token_lens.get(r.example_id, cfg.max_input_tokens + 1) <= cfg.max_input_tokens:
            by_group[r.group_id][r.response_id].append(r)
        else:
            excluded += 1
    ids: list[str] = []
    n_resp = 0
    for g in sorted(by_group):
        order = sorted(by_group[g], key=lambda rid: hashlib.sha256(f"{cfg.sample_seed}:{tag}:resp:{rid}".encode()).hexdigest())
        for rid in order[:responses_per_group]:
            ids.extend(r.example_id for r in by_group[g][rid])
            n_resp += 1
    return {"example_ids": sorted(ids), "groups": sorted(by_group), "responses": n_resp, "excluded_over_length": excluded,
            "coverage_rule": f"groups of the {coverage_sentences}-sentence coverage sample; {responses_per_group} whole responses per group"}


def fits_fraction(projected: Mapping[str, float], remaining: Mapping[str, float], fraction: float) -> dict[str, bool]:
    """Each projected phase time must be <= `fraction` of that phase's remaining budget."""
    return {k: projected[k] <= fraction * remaining[k] for k in projected}


def system_snapshot() -> dict[str, float | None]:
    """RAM of the machine and of this process (GB); None when psutil is unavailable. VRAM is added by the backend."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"ram_used_gb": round(vm.used / 1e9, 3), "ram_available_gb": round(vm.available / 1e9, 3),
                "process_rss_gb": round(psutil.Process().memory_info().rss / 1e9, 3)}
    except Exception:  # noqa: BLE001 - monitoring must never crash a run
        return {"ram_used_gb": None, "ram_available_gb": None, "process_rss_gb": None}


class ProgressLog:
    """Append-only JSON-lines heartbeat: counts, elapsed time, rate, RAM and VRAM. Flushed and fsynced so a kill leaves it intact."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.monotonic,
                 vram_fn: Callable[[], Mapping[str, float]] | None = None) -> None:
        self.path, self._clock, self.vram_fn = path, clock, vram_fn
        self._t0 = clock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, phase: str, done: int, total: int, **extra: Any) -> None:
        now = self._clock() - self._t0
        rec = {"t": round(now, 2), "phase": phase, "done": done, "total": total,
               "rate_per_s": round(done / now, 4) if now > 0 and done else 0.0, **system_snapshot(), **extra}
        if self.vram_fn is not None:
            with contextlib.suppress(Exception):  # monitoring must never crash a run
                rec.update(self.vram_fn())
        with self.path.open("a", encoding="utf8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


class Checkpointer:
    """Durable partial scores: one JSON line per batch, flushed and fsynced. Partial scores never authorise continuation."""

    def __init__(self, directory: Path, phase: str) -> None:
        self.path = directory / f"partial_scores_{phase}.jsonl"
        directory.mkdir(parents=True, exist_ok=True)
        self.phase, self.count = phase, 0

    def write_many(self, scores: Mapping[str, float]) -> None:
        with self.path.open("a", encoding="utf8") as fh:
            for eid, s in scores.items():
                fh.write(json.dumps({"example_id": eid, "score": float(s)}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.count += len(scores)

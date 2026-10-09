"""Simulation of the COMPLETE improvement gate and of the Phase 1 screening gate (docs/ROUTEBENCH_JUDGE_IMPROVEMENT_PROPOSAL_V2_20261005.md). No data, model or provider call.

`judge_improvement_power.py` (kept unchanged) simulated only two conditions (lower confidence bounds of the accuracy and kappa differences). This module applies EVERY condition of the
evaluation gate, the exact disagreement-subsampling branch, and the Phase 1 screening rule, and reports detection power separately from full-gate success probability.

Evaluation gate (N sentences in documents of DOC_SIZE; both judges run on everything; humans label every disagreement - subsampled at a fixed probability when more than LABEL_CAP - and a uniform
PI_CONCORDANT of the agreements; inverse-probability weights; documents are the resampling unit):
  c1 detection: lower 95% bounds of the accuracy difference and of the kappa difference are > 0
  c2 weighted point accuracy gain >= 0.03
  c3 neither the false-pass nor the false-rejection rate (share of ALL sentences) rises by more than 0.01, and at least one falls (point estimates)
  c4 flip precision (candidate-right share of labelled disagreements) > 0.5
  c5 the uncertain band's error rate exceeds the confident band's (ASSUMED process: P(uncertain | candidate wrong) = U_WRONG, P(uncertain | candidate right) = U_RIGHT)
  detection = c1; binary gate = c1..c4; full gate = c1..c5. Assumptions that are NOT simulated or are simply asserted are listed in the proposal; results are rates under assumptions, not predictions.
Screening gate (all items labelled, N = 257 split TRAIN 129 / dev 128 by document, v18 vs candidate; exposed data, a futility screen only):
  s1 pooled kappa difference >= +0.04; s2 TRAIN and dev kappa differences are each > 0; s3 error-class rule as c3; s4 flip precision > 0.5.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from evalops.judge_improvement_power import _kappa

DOC_SIZE, PI_CONCORDANT, LABEL_CAP = 10, 0.15, 250
DOC_DISAGREE_SHAPE, DOC_PREV_CONC = 2.0, 5.0
U_WRONG, U_RIGHT = 0.30, 0.10
MIN_GAIN, MAX_CLASS_RISE = 0.03, 0.01


def _world(n, accuracy, disagree, gain, prevalence, rng, doc_size, disagree_prev):
    docs = max(2, n // doc_size)
    doc = np.arange(n) % docs
    prev_d = rng.beta(prevalence * DOC_PREV_CONC, (1 - prevalence) * DOC_PREV_CONC, docs)
    mult = rng.gamma(DOC_DISAGREE_SHAPE, 1 / DOC_DISAGREE_SHAPE, docs)
    y = (rng.random(n) < prev_d[doc]).astype(int)
    disc = rng.random(n) < np.clip(disagree * mult[doc], 0, 0.9)
    if disagree_prev is not None:  # asymmetric scenario: the share of supported sentences among the disagreements differs from the base rate
        y = np.where(disc, (rng.random(n) < disagree_prev).astype(int), y)
    q = float(np.clip((1 + gain / disagree) / 2, 0, 1))  # P(candidate right | disagreement)
    r = float(np.clip((accuracy - disagree * (1 - q)) / (1 - disagree), 0, 1))  # P(both right | agreement) so that v18's accuracy is `accuracy`
    new_right = np.where(disc, rng.random(n) < q, rng.random(n) < r)
    old_right = np.where(disc, ~new_right, new_right)
    return docs, doc, y, disc, np.where(old_right, y, 1 - y), np.where(new_right, y, 1 - y), new_right


def _cells(pred, y, w, doc, docs):
    def f(v):
        return np.bincount(doc, weights=v, minlength=docs)
    return np.stack([f(w * ((pred == 1) & (y == 1))), f(w * ((pred == 1) & (y == 0))), f(w * ((pred == 0) & (y == 1))), f(w * ((pred == 0) & (y == 0)))], axis=1)


def simulate_gate(n: int, accuracy: float, disagree: float, gain: float, prevalence: float, rng: np.random.Generator, draws: int = 300, doc_size: int = DOC_SIZE,
                  pi_conc: float = PI_CONCORDANT, label_cap: int | None = LABEL_CAP, disagree_prev: float | None = None) -> dict[str, float]:
    docs, doc, y, disc, old, new, new_right = _world(n, accuracy, disagree, gain, prevalence, rng, doc_size, disagree_prev)
    n_disc = int(disc.sum())
    pi_disc = 1.0 if not label_cap or n_disc <= label_cap else label_cap / n_disc
    pi = np.where(disc, pi_disc, pi_conc)
    labelled = rng.random(n) < pi
    w = np.where(labelled, 1.0 / pi, 0.0)
    unc = rng.random(n) < np.where(new_right, U_RIGHT, U_WRONG)

    def f(v):
        return np.bincount(doc, weights=v, minlength=docs)

    a, b = _cells(old, y, w, doc, docs), _cells(new, y, w, doc, docs)
    better, worse, tot = f(w * (disc & new_right)), f(w * (disc & ~new_right)), f(w)
    unc_n, unc_e, conf_n, conf_e = f(w * unc), f(w * (unc & ~new_right)), f(w * ~unc), f(w * (~unc & ~new_right))
    counts = rng.multinomial(docs, np.full(docs, 1 / docs), size=draws).astype(float)
    dacc = (counts @ (better - worse)) / (counts @ tot)
    dk = _kappa(*(counts @ b).T) - _kappa(*(counts @ a).T)
    dk = dk[np.isfinite(dk)]
    c1 = float(np.percentile(dacc, 2.5) > 0 and len(dk) > 20 and np.percentile(dk, 2.5) > 0)
    nw = float(tot.sum())
    acc_gain = float((better.sum() - worse.sum()) / nw)
    d_fp = float((b[:, 1].sum() - a[:, 1].sum()) / nw)  # false passes: judge pass, humans unsupported
    d_fr = float((b[:, 2].sum() - a[:, 2].sum()) / nw)  # false rejections: judge reject, humans supported
    c2 = float(acc_gain >= MIN_GAIN)
    c3 = float(d_fp <= MAX_CLASS_RISE and d_fr <= MAX_CLASS_RISE and min(d_fp, d_fr) < 0)
    fl = better.sum() + worse.sum()
    c4 = float(fl > 0 and better.sum() / fl > 0.5)
    c5 = float(unc_n.sum() > 0 and conf_n.sum() > 0 and unc_e.sum() / unc_n.sum() > conf_e.sum() / conf_n.sum())
    b14 = c1 * c2 * c3 * c4
    return {"labelled": float(labelled.sum()), "disagreements": float(n_disc), "capped": float(pi_disc < 1), "acc_gain": acc_gain, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "gate_binary": b14, "gate_full": b14 * c5}


def simulate_screen(accuracy: float, disagree: float, gain: float, prevalence: float, rng: np.random.Generator, n_train: int = 129, n_dev: int = 128, doc_size: int = 3) -> dict[str, float]:
    n = n_train + n_dev
    docs, doc, y, disc, old, new, new_right = _world(n, accuracy, disagree, gain, prevalence, rng, doc_size, None)
    one = np.ones(n)
    in_train = (doc * 7919 % docs) < docs * n_train / n  # deterministic document split, roughly proportional

    def kd(mask):
        a, b = _cells(old[mask], y[mask], one[mask], doc[mask], docs), _cells(new[mask], y[mask], one[mask], doc[mask], docs)
        return float(_kappa(*b.sum(0)) - _kappa(*a.sum(0)))

    k_all, k_tr, k_dv = kd(np.ones(n, bool)), kd(in_train), kd(~in_train)
    d_fp = float(((new == 1) & (y == 0)).sum() - ((old == 1) & (y == 0)).sum()) / n
    d_fr = float(((new == 0) & (y == 1)).sum() - ((old == 0) & (y == 1)).sum()) / n
    better, worse = int((disc & new_right).sum()), int((disc & ~new_right).sum())
    s1, s2 = float(k_all >= 0.04), float(k_tr > 0 and k_dv > 0)
    s3 = float(d_fp <= MAX_CLASS_RISE and d_fr <= MAX_CLASS_RISE and min(d_fp, d_fr) < 0)
    s4 = float(better + worse > 0 and better / (better + worse) > 0.5)
    return {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "screen": s1 * s2 * s3 * s4, "kappa_diff": k_all}


def _mean(res, k):
    return round(float(np.nanmean([r[k] for r in res])), 3)


def gate_table(n: int = 1500, gains: Sequence[float] = (0.0, 0.03, 0.04, 0.05, 0.06), disagreements: Sequence[float] = (0.06, 0.09, 0.12), accuracy: float = 0.85,
               prevalence: float = 0.70, reps: int = 1000, draws: int = 300, seed: int = 20261105, disagree_prev: float | None = None, label_cap: int | None = LABEL_CAP) -> list[dict]:
    rows = []
    for d in disagreements:
        for g in gains:
            rng = np.random.default_rng([seed, n, zlib.crc32(f"{d}:{g}:{disagree_prev}:{label_cap}".encode())])
            res = [simulate_gate(n, accuracy, d, g, prevalence, rng, draws, disagree_prev=disagree_prev, label_cap=label_cap) for _ in range(reps)]
            rows.append({"N": n, "disagreement_rate": d, "true_accuracy_gain": g, "disagree_prev": disagree_prev, "label_cap": label_cap, "reps": reps, "mean_labels": round(float(np.mean([r["labelled"] for r in res])), 1),
                         "share_capped": _mean(res, "capped"), "p_detection_c1": _mean(res, "c1"), "p_c2_point_gain": _mean(res, "c2"), "p_c3_class_rule": _mean(res, "c3"), "p_c4_flip": _mean(res, "c4"),
                         "p_c5_uncertain": _mean(res, "c5"), "p_binary_gate": _mean(res, "gate_binary"), "p_full_gate": _mean(res, "gate_full")})
    return rows


def screen_table(gains: Sequence[float] = (0.0, 0.02, 0.03, 0.05), disagreements: Sequence[float] = (0.06, 0.09, 0.12), settings: Sequence[tuple[float, float]] = ((0.85, 0.70), (0.78, 0.55)),
                 reps: int = 3000, seed: int = 20261105) -> list[dict]:
    rows = []
    for acc, prev in settings:
        for d in disagreements:
            for g in gains:
                rng = np.random.default_rng([seed, 257, zlib.crc32(f"{acc}:{prev}:{d}:{g}".encode())])
                res = [simulate_screen(acc, d, g, prev, rng) for _ in range(reps)]
                rows.append({"v18_accuracy": acc, "prevalence": prev, "disagreement_rate": d, "true_accuracy_gain": g, "reps": reps, "p_s1": _mean(res, "s1"), "p_s2": _mean(res, "s2"), "p_s3": _mean(res, "s3"),
                             "p_s4": _mean(res, "s4"), "p_screen_pass": _mean(res, "screen"), "mean_kappa_diff": _mean(res, "kappa_diff")})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--screen-reps", type=int, default=3000)
    a = p.parse_args(argv)
    if a.out.exists():
        print("REFUSED: output exists", file=sys.stderr)
        return 2
    out = {"assumptions": {"doc_size": DOC_SIZE, "pi_concordant": PI_CONCORDANT, "label_cap": LABEL_CAP, "u_wrong": U_WRONG, "u_right": U_RIGHT, "min_gain": MIN_GAIN, "max_class_rise": MAX_CLASS_RISE,
                           "v18_accuracy": 0.85, "prevalence": 0.70, "bootstrap_draws_in_simulation": 300}}
    out["evaluation_N1500"] = gate_table(1500, reps=a.reps)
    out["evaluation_N1500_asymmetric_disagreement_prev0.4"] = gate_table(1500, reps=a.reps, disagree_prev=0.4)
    out["evaluation_N3000"] = gate_table(3000, reps=a.reps)
    out["screening_N257"] = screen_table(reps=a.screen_reps)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf8")
    print("wrote", {k: len(v) for k, v in out.items() if isinstance(v, list)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

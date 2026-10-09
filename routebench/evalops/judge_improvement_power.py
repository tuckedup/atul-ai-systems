"""Simulation-based POWER ANALYSIS for a paired comparison of a candidate judge against frozen v18 on NEW human-labelled documents (no data, no model, no calls).

Design simulated (docs/ROUTEBENCH_JUDGE_IMPROVEMENT_PROPOSAL_20261005.md): N sentences in documents of DOC_SIZE. Both judges are run on everything. Humans label (a) EVERY sentence on which the
two judges disagree (probability 1) and (b) a uniform random sample of the sentences on which they agree (probability PI_CONCORDANT), with inverse-probability weights. Agreements
cancel in an accuracy difference, so the accuracy difference is estimated exactly from the labelled disagreements; kappa needs the weighted margins. Documents are the resampling unit.

Parameters: v18 accuracy A, disagreement rate d, true accuracy gain delta of the candidate (P(candidate right | disagreement) = (1 + delta / d) / 2), prevalence of supported claims,
per-document heterogeneity in disagreement rate and prevalence. Tests: document-bootstrap 95% lower bound of the accuracy difference > 0, of the kappa difference > 0, and both together
(the proposal's improvement criterion). All results are simulated rates under stated assumptions, not claims about the real process.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

DOC_SIZE, PI_CONCORDANT = 10, 0.15
DOC_DISAGREE_SHAPE, DOC_PREV_CONC = 2.0, 5.0


def _kappa(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray) -> np.ndarray:
    n = tp + fp + fn + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        po = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
        k = (po - pe) / (1 - pe)
    return np.where((n > 0) & (pe < 1 - 1e-12), k, np.nan)


def simulate_once(n: int, accuracy: float, disagree: float, gain: float, prevalence: float, rng: np.random.Generator, draws: int = 300) -> dict[str, float]:
    docs = max(2, n // DOC_SIZE)
    doc = np.arange(n) % docs
    prev_d = rng.beta(prevalence * DOC_PREV_CONC, (1 - prevalence) * DOC_PREV_CONC, docs)
    mult = rng.gamma(DOC_DISAGREE_SHAPE, 1 / DOC_DISAGREE_SHAPE, docs)
    y = (rng.random(n) < prev_d[doc]).astype(int)
    d_i = np.clip(disagree * mult[doc], 0, 0.9)
    disc = rng.random(n) < d_i
    q = float(np.clip((1 + gain / disagree) / 2, 0, 1))  # P(candidate right | disagreement)
    r = float(np.clip((accuracy - disagree * (1 - q)) / (1 - disagree), 0, 1))  # P(both right | agreement), so that v18's overall accuracy is `accuracy`
    new_right = np.where(disc, rng.random(n) < q, rng.random(n) < r)
    old_right = np.where(disc, ~new_right, new_right)  # on disagreements exactly one of the two judges is right
    old = np.where(old_right, y, 1 - y)
    new = np.where(new_right, y, 1 - y)
    pi = np.where(disc, 1.0, PI_CONCORDANT)
    labelled = rng.random(n) < pi
    w = np.where(labelled, 1.0 / pi, 0.0)

    def per_doc(v: np.ndarray) -> np.ndarray:
        return np.bincount(doc, weights=v, minlength=docs)

    cells = {}
    for name, pred in (("a", old), ("b", new)):
        cells[name] = np.stack([per_doc(w * ((pred == 1) & (y == 1))), per_doc(w * ((pred == 1) & (y == 0))), per_doc(w * ((pred == 0) & (y == 1))), per_doc(w * ((pred == 0) & (y == 0)))], axis=1)
    better, worse = per_doc(disc & new_right), per_doc(disc & ~new_right)
    nd = per_doc(np.ones(n))
    counts = rng.multinomial(docs, np.full(docs, 1 / docs), size=draws).astype(float)
    dacc = (counts @ (better - worse)) / (counts @ nd)
    dk = _kappa(*(counts @ cells["b"]).T) - _kappa(*(counts @ cells["a"]).T)
    dk = dk[np.isfinite(dk)]
    acc_ok = float(np.percentile(dacc, 2.5) > 0)
    kap_ok = float(len(dk) > 20 and np.percentile(dk, 2.5) > 0)
    return {"labelled": float(labelled.sum()), "disagreements": float(disc.sum()), "acc_gain": float((better.sum() - worse.sum()) / n), "acc_ok": acc_ok, "kappa_ok": kap_ok, "both_ok": acc_ok * kap_ok}


def power_table(sizes: Sequence[int] = (1000, 2000, 3000, 4000), gains: Sequence[float] = (0.0, 0.02, 0.03, 0.05), disagreements: Sequence[float] = (0.06, 0.09, 0.12),
                accuracy: float = 0.85, prevalence: float = 0.70, reps: int = 400, draws: int = 300, seed: int = 20261106) -> list[dict]:
    rows = []
    for d in disagreements:
        for g in gains:
            for n in sizes:
                rng = np.random.default_rng([seed, n, zlib.crc32(f"{d}:{g}".encode())])
                res = [simulate_once(n, accuracy, d, g, prevalence, rng, draws) for _ in range(reps)]
                m = lambda k, res=res: float(np.nanmean([r[k] for r in res]))
                rows.append({"disagreement_rate": d, "true_accuracy_gain": g, "N": n, "documents": n // DOC_SIZE, "mean_disagreements": round(m("disagreements"), 1), "mean_human_labels": round(m("labelled"), 1),
                             "mean_accuracy_gain": round(m("acc_gain"), 4), "power_accuracy": round(m("acc_ok"), 3), "power_kappa": round(m("kappa_ok"), 3), "power_both": round(m("both_ok"), 3), "reps": reps})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=400)
    a = p.parse_args(argv)
    if a.out.exists():
        print("REFUSED: output exists", file=sys.stderr)
        return 2
    rows = power_table(reps=a.reps)
    a.out.write_text(json.dumps({"assumptions": {"pi_concordant": PI_CONCORDANT, "doc_size": DOC_SIZE, "v18_accuracy": 0.85, "prevalence": 0.70, "doc_disagreement_gamma_shape": DOC_DISAGREE_SHAPE,
                                                  "doc_prevalence_concentration": DOC_PREV_CONC}, "rows": rows}, indent=1), encoding="utf8")
    print(f"wrote {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

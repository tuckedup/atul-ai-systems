"""Simulation-based POWER ANALYSIS for the flag-then-adjudicate validation of typed-veto-v1.1 (no data, no model; assumptions are explicit parameters).

Design simulated (the one in docs/ROUTEBENCH_TYPED_VETO_V1_20261004.md, amended): N new candidate sentences in D = N / m documents. v18 judges all of them; the veto flags a subset of the
v18 passes; humans label (a) EVERY flagged case (sampling probability 1), (b) a uniform random sample of the other v18 passes with probability PI_PASS, (c) a uniform random sample of
the v18 rejections with probability PI_REJECT. Inverse-probability weights (1 / sampling probability) are kept for every labelled case. Documents are the resampling unit.

Quantities, kept apart:
  accuracy improvement  = (flags that are truly unsupported - flags that are truly supported) / N      (only flagged cases change, and all of them are labelled)
  kappa improvement     = weighted Cohen's kappa of v18+veto minus that of v18, from the IPW confusion matrices
  flip precision        = truly-unsupported flags / flags;  the veto helps accuracy iff precision > 0.5
Each is tested by a document-level bootstrap lower bound > 0 (precision: > 0.5); the naive Wilson bound for precision is also shown. All results are simulated rates under stated assumptions,
not claims about the real process.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import zlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PI_PASS, PI_REJECT, DOC_SIZE = 0.15, 0.15, 10
DOC_FLAG_SHAPE = 2.0  # gamma shape of the per-document multiplier on flag probabilities (CV ~ 0.71): wrong flips cluster in few documents
DOC_PREV_CONC = 5.0  # beta concentration of per-document supported prevalence


@dataclass(frozen=True)
class Scenario:
    name: str
    p_supported: float
    v18_sens: float  # P(v18 pass | supported)
    v18_spec: float  # P(v18 reject | unsupported)
    flag_on_false_pass: float  # P(flag | v18 pass, unsupported)
    flag_on_true_pass: float  # P(flag | v18 pass, supported)

    @property
    def expected_precision(self) -> float:
        fp = (1 - self.p_supported) * (1 - self.v18_spec)
        tp = self.p_supported * self.v18_sens
        good, bad = fp * self.flag_on_false_pass, tp * self.flag_on_true_pass
        return good / (good + bad) if good + bad else float("nan")


SCENARIOS = (
    Scenario("S1 dev-observed rates (dev profile: many false passes)", 0.62, 0.94, 0.71, 3 / 14, 1 / 74),
    Scenario("S2 same flag rates, fresh-document profile (few false passes)", 0.80, 0.82, 0.86, 3 / 14, 1 / 74),
    Scenario("S3 better veto on the dev profile", 0.62, 0.94, 0.71, 0.30, 0.010),
    Scenario("S4 null: precision 0.5, no net gain", 0.62, 0.94, 0.71, 3 / 14, 0.0405),
    Scenario("S5 harmful veto", 0.62, 0.94, 0.71, 0.15, 0.040),
)


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def _kappa_w(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray) -> np.ndarray:
    n = tp + fp + fn + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        po = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
        k = (po - pe) / (1 - pe)
    return np.where((n > 0) & (pe < 1 - 1e-12), k, np.nan)


def simulate_once(sc: Scenario, n: int, rng: np.random.Generator, draws: int, doc_size: int = DOC_SIZE, pi_pass: float = PI_PASS, pi_reject: float = PI_REJECT) -> dict[str, float]:
    docs = max(2, n // doc_size)
    doc = np.repeat(np.arange(docs), doc_size)[:n] if docs * doc_size >= n else np.arange(n) % docs
    n = len(doc)
    prev_d = rng.beta(sc.p_supported * DOC_PREV_CONC, (1 - sc.p_supported) * DOC_PREV_CONC, docs)
    mult = rng.gamma(DOC_FLAG_SHAPE, 1 / DOC_FLAG_SHAPE, docs)
    y = (rng.random(n) < prev_d[doc]).astype(int)  # 1 = supported
    v18 = np.where(y == 1, rng.random(n) < sc.v18_sens, rng.random(n) >= sc.v18_spec).astype(int)  # 1 = pass
    p_flag = np.where(y == 1, sc.flag_on_true_pass, sc.flag_on_false_pass) * mult[doc]
    flag = ((v18 == 1) & (rng.random(n) < np.clip(p_flag, 0, 1))).astype(int)
    final = np.where(flag == 1, 0, v18)
    pi = np.where(flag == 1, 1.0, np.where(v18 == 1, pi_pass, pi_reject))  # fixed, pre-declared sampling probabilities
    labelled = rng.random(n) < pi
    w = np.where(labelled, 1.0 / pi, 0.0)  # inverse-probability weight, zero if not labelled

    def per_doc(values: np.ndarray) -> np.ndarray:
        return np.bincount(doc, weights=values, minlength=docs)

    cells = {}
    for name, pred in (("a", v18), ("b", final)):
        cells[name] = np.stack([per_doc(w * ((pred == 1) & (y == 1))), per_doc(w * ((pred == 1) & (y == 0))), per_doc(w * ((pred == 0) & (y == 1))), per_doc(w * ((pred == 0) & (y == 0)))], axis=1)
    good = per_doc((flag == 1) & (y == 0))
    bad = per_doc((flag == 1) & (y == 1))  # flags are always labelled
    nd = per_doc(np.ones(n))
    point_k = float(_kappa_w(*cells["b"].sum(0)) - _kappa_w(*cells["a"].sum(0)))
    counts = rng.multinomial(docs, np.full(docs, 1 / docs), size=draws).astype(float)  # document bootstrap
    ka = _kappa_w(*(counts @ cells["a"]).T)
    kb = _kappa_w(*(counts @ cells["b"]).T)
    dk = (kb - ka)[np.isfinite(kb - ka)]
    dacc = (counts @ (good - bad)) / (counts @ nd)
    gb, bb = counts @ good, counts @ bad
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_b = np.where(gb + bb > 0, gb / (gb + bb), np.nan)
    prec_b = prec_b[np.isfinite(prec_b)]
    flags_total = int(flag.sum())
    c, wr = int(good.sum()), int(bad.sum())
    return {"flags": flags_total, "labelled": int(labelled.sum()), "acc_gain": (c - wr) / n, "kappa_gain": point_k, "precision": (c / flags_total) if flags_total else float("nan"),
            "wilson_ok": float(flags_total > 0 and wilson_lower(c, flags_total) > 0.5), "boot_prec_ok": float(len(prec_b) > 20 and np.percentile(prec_b, 2.5) > 0.5),
            "acc_ok": float(np.percentile(dacc, 2.5) > 0), "kappa_ok": float(len(dk) > 20 and np.percentile(dk, 2.5) > 0)}


def _mean(res: list[dict[str, float]], k: str) -> float:
    return float(np.nanmean([r[k] for r in res]))


def power_table(scenarios: Sequence[Scenario] = SCENARIOS, sizes: Sequence[int] = (300, 600, 1000, 1500, 2500, 4000), reps: int = 300, draws: int = 300, seed: int = 20261105) -> list[dict]:
    rows = []
    for sc in scenarios:
        for n in sizes:
            rng = np.random.default_rng([seed, n, zlib.crc32(sc.name.encode())])
            res = [simulate_once(sc, n, rng, draws) for _ in range(reps)]
            m = functools.partial(_mean, res)
            rows.append({"scenario": sc.name, "N": n, "documents": n // DOC_SIZE, "expected_precision": round(sc.expected_precision, 3), "mean_flags": round(m("flags"), 1), "mean_human_labels": round(m("labelled"), 1),
                         "mean_accuracy_gain": round(m("acc_gain"), 4), "mean_kappa_gain": round(m("kappa_gain"), 4), "power_precision_wilson": round(m("wilson_ok"), 3),
                         "power_precision_doc_bootstrap": round(m("boot_prec_ok"), 3), "power_accuracy_gain": round(m("acc_ok"), 3), "power_kappa_gain": round(m("kappa_ok"), 3), "reps": reps})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=300)
    a = p.parse_args(argv)
    if a.out.exists():
        print("REFUSED: output exists", file=sys.stderr)
        return 2
    rows = power_table(reps=a.reps)
    a.out.write_text(json.dumps({"assumptions": {"pi_pass": PI_PASS, "pi_reject": PI_REJECT, "doc_size": DOC_SIZE, "doc_flag_gamma_shape": DOC_FLAG_SHAPE, "doc_prevalence_concentration": DOC_PREV_CONC,
                                                  "scenarios": [asdict(s) | {"expected_precision": s.expected_precision} for s in SCENARIOS]}, "rows": rows}, indent=1), encoding="utf8")
    print(f"wrote {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

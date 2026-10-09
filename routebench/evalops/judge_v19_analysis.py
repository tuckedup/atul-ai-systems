"""Pure analysis definitions for the judge-v19 Phase 1 screening (docs/ROUTEBENCH_JUDGE_IMPROVEMENT_PROPOSAL_V2_20261005.md). No provider call, no file I/O, no labels read here.

Arms (frozen; nothing is added after a result is seen):
  A0  the cached frozen v18 verdict: pass iff the original all-criteria-pass score >= 1.0.
  A1  MAJORITY VOTE over the five v18 calls' ORIGINAL pass/fail verdicts (each call passes iff its score >= 1.0; the vote passes iff at least 3 of 5 pass). It isolates voting and does not
      loosen the rubric: five calls that each score 0.8 are five rejections, and A1 rejects them. Averaging raw scores and testing the mean at 0.5 is NOT A1 and is not used anywhere.
  A2  the primary candidate: class-balanced L2 logistic regression on the nine-feature schema S, which keeps the RAW scores (mean, all-pass fraction, standard deviation) as features.
Nested CV, thresholds and the screening gate are specified once, here, and hashed with this file at freeze time.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .audit2_v2 import kappa

PASS_THRESHOLD = 1.0
K = 5
MAJORITY_MIN_PASSES = 3
RULE = "phase1-v1"
SEED = 20261105
C_GRID = (0.01, 0.1, 1.0, 10.0)
OUTER_FOLDS, INNER_FOLDS, FINAL_INNER_FOLDS = 5, 4, 5
ALPHA_PASS, ALPHA_REJECT = 0.10, 0.10
MIN_CONFIDENT_COVERAGE = 0.30
CLIP = (1e-3, 1 - 1e-3)
FEATURES = ("v18_mean_raw_score", "v18_all_pass_fraction", "v18_raw_score_sd", "minicheck_full_logit", "minicheck_weakest_sentence_logit", "log_sentence_chars", "log_source_chars",
            "sentence_digit_count", "negation_or_modal")
GATE = {"min_pooled_kappa_gain": 0.04, "each_subset_kappa_gain_above": 0.0, "max_class_count_rise_share": 0.01, "flip_precision_above": 0.5}
_NEG_MODAL = re.compile(r"\b(?:not|no|never|none|neither|nor|cannot|may|might|could|would|should|can|will|must)\b|n't", re.IGNORECASE)


class AnalysisError(ValueError):
    pass


def verdict(score: float) -> int:
    """One v18 call's original all-criteria-pass verdict."""
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise AnalysisError(f"invalid v18 score {score!r}")
    return int(score >= PASS_THRESHOLD)


def a0_verdict(cached_score: float) -> int:
    return verdict(cached_score)


def a1_majority(scores: Sequence[float]) -> int:
    if len(scores) != K:
        raise AnalysisError(f"A1 needs exactly {K} v18 scores, got {len(scores)}")
    return int(sum(verdict(s) for s in scores) >= MAJORITY_MIN_PASSES)


def _logit(p: float) -> float:
    p = min(max(float(p), CLIP[0]), CLIP[1])
    return math.log(p / (1 - p))


def feature_row(scores: Sequence[float], mc_full: float, mc_sentences: Sequence[float], sentence: str, source: str) -> list[float]:
    if len(scores) != K or not mc_sentences:
        raise AnalysisError("a feature row needs five v18 scores and at least one MiniCheck sentence score")
    for s in scores:
        verdict(s)
    mean = sum(scores) / K
    sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / K)  # population sd (ddof = 0)
    return [mean, sum(verdict(s) for s in scores) / K, sd, _logit(mc_full), _logit(min(mc_sentences)), math.log1p(len(sentence)), math.log1p(len(source)),
            float(sum(ch.isdigit() for ch in sentence)), float(bool(_NEG_MODAL.search(sentence)))]


def fold_of(tag: str, group: str, mod: int) -> int:
    return int(hashlib.sha256(f"{RULE}|{tag}|{SEED}|{group}".encode()).hexdigest(), 16) % mod


def _fit(X: np.ndarray, y: np.ndarray, c: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(X)  # fitted on the training fold only
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=1000).fit(scaler.transform(X), y)
    return scaler, model


def _balanced_logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(0.5 * (np.mean(-np.log(1 - p[y == 0])) + np.mean(-np.log(p[y == 1]))))


def choose_c(X: np.ndarray, y: np.ndarray, groups: Sequence[str], tag: str, folds: int) -> float:
    fold = np.array([fold_of(tag, g, folds) for g in groups])
    best: tuple[float, float] | None = None
    for c in C_GRID:  # ascending, so a tie keeps the smaller C
        p = np.full(len(y), np.nan)
        for f in range(folds):
            tr, te = fold != f, fold == f
            if not te.any():
                continue
            if len(set(y[tr].tolist())) < 2:
                raise AnalysisError("an inner training fold lacks one class; the sample cannot support this procedure")
            sc, m = _fit(X[tr], y[tr], c)
            p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        loss = _balanced_logloss(y, p)
        if best is None or loss < best[0] - 1e-12:
            best = (loss, c)
    assert best is not None
    return best[1]


def nested_oof(X: np.ndarray, y: np.ndarray, groups: Sequence[str]) -> dict[str, Any]:
    """Outer 5-fold document-grouped out-of-fold probabilities of 'supported'; C is chosen by inner 4-fold CV inside each outer training set. Labels never leave their own fold."""
    outer = np.array([fold_of("outer", g, OUTER_FOLDS) for g in groups])
    oof, chosen = np.full(len(y), np.nan), {}
    for f in range(OUTER_FOLDS):
        tr, te = outer != f, outer == f
        if not te.any():
            continue
        c = choose_c(X[tr], y[tr], [g for g, t in zip(groups, tr, strict=True) if t], f"inner|{f}", INNER_FOLDS)
        sc, m = _fit(X[tr], y[tr], c)
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        chosen[f] = c
    if np.isnan(oof).any():
        raise AnalysisError("some rows have no out-of-fold prediction")
    return {"oof": oof, "outer_fold_sizes": [int((outer == f).sum()) for f in range(OUTER_FOLDS)], "chosen_c_per_outer_fold": chosen}


def bands(oof: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Class-conditional bands. t_hi: smallest t >= 0.5 with P(p >= t | unsupported) <= ALPHA_PASS. t_lo: largest t <= 0.5 with P(p <= t | supported) <= ALPHA_REJECT.
    +inf / -inf (an empty band) always qualify, so the rule is always feasible; coverage tells how useful it is."""
    uns, sup = oof[y == 0], oof[y == 1]
    t_hi = next((t for t in sorted({0.5, *oof[oof >= 0.5].tolist()}) if float(np.mean(uns >= t)) <= ALPHA_PASS), math.inf)
    t_lo = next((t for t in sorted({0.5, *oof[oof <= 0.5].tolist()}, reverse=True) if float(np.mean(sup <= t)) <= ALPHA_REJECT), -math.inf)
    confident = float(np.mean((oof >= t_hi) | (oof <= t_lo)))
    return {"t_hi": t_hi, "t_lo": t_lo, "confident_coverage": confident, "abstain_claim_allowed": confident >= MIN_CONFIDENT_COVERAGE,
            "false_pass_rate_among_unsupported": float(np.mean(uns >= t_hi)), "false_reject_rate_among_supported": float(np.mean(sup <= t_lo))}


def final_model(X: np.ndarray, y: np.ndarray, groups: Sequence[str], oof: np.ndarray) -> dict[str, Any]:
    c = choose_c(X, y, groups, "final", FINAL_INNER_FOLDS)
    sc, m = _fit(X, y, c)
    b = bands(oof, y)
    body = {"features": list(FEATURES), "C": c, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
            "t_lo": b["t_lo"] if math.isfinite(b["t_lo"]) else None, "t_hi": b["t_hi"] if math.isfinite(b["t_hi"]) else None, "abstain_claim_allowed": b["abstain_claim_allowed"]}
    body["model_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def predict(model: dict[str, Any], X: np.ndarray) -> dict[str, np.ndarray]:
    z = (np.asarray(X, float) - np.array(model["mean"])) / np.array(model["scale"])
    p = 1 / (1 + np.exp(-(z @ np.array(model["coef"]) + model["intercept"])))
    hi, lo = model["t_hi"], model["t_lo"]
    confident = np.zeros(len(p), bool)
    if hi is not None:
        confident |= p >= hi
    if lo is not None:
        confident |= p <= lo
    return {"p": p, "binary": (p >= 0.5).astype(int), "uncertain": ~confident}  # uncertain items keep their best-guess verdict; they are never rejected by default


def _k(y: Sequence[int], p: Sequence[int]) -> float | None:
    return kappa(list(y), list(p)) if len(set(y)) > 1 else None


def screen_gate(pops: Sequence[str], y: Sequence[int], a0: Sequence[int], a2: Sequence[int]) -> dict[str, Any]:
    """Phase 1 screening gate on the 257 evaluation-only cases. All four conditions are required."""
    n = len(y)
    if not (len(pops) == len(a0) == len(a2) == n) or n == 0:
        raise AnalysisError("screen inputs differ in length")

    def gain(mask: list[bool]) -> float | None:
        yy, p0, p2 = ([v for v, m in zip(s, mask, strict=True) if m] for s in (y, a0, a2))
        k0, k2 = _k(yy, p0), _k(yy, p2)
        return None if k0 is None or k2 is None else k2 - k0

    pooled, tr, dv = gain([True] * n), gain([p == "train" for p in pops]), gain([p == "dev" for p in pops])
    fp = lambda p: sum(1 for a, b in zip(y, p, strict=True) if a == 0 and b == 1)
    fr = lambda p: sum(1 for a, b in zip(y, p, strict=True) if a == 1 and b == 0)
    d_fp, d_fr = fp(a2) - fp(a0), fr(a2) - fr(a0)
    limit = GATE["max_class_count_rise_share"] * n
    fix = sum(1 for a, u, v in zip(y, a0, a2, strict=True) if u != a and v == a)
    brk = sum(1 for a, u, v in zip(y, a0, a2, strict=True) if u == a and v != a)
    s = {"s1_pooled_kappa_gain": pooled is not None and pooled >= GATE["min_pooled_kappa_gain"],
         "s2_each_subset_positive": tr is not None and dv is not None and tr > GATE["each_subset_kappa_gain_above"] and dv > GATE["each_subset_kappa_gain_above"],
         "s3_error_classes": d_fp <= limit and d_fr <= limit and min(d_fp, d_fr) < 0,
         "s4_flip_precision": fix + brk > 0 and fix / (fix + brk) > GATE["flip_precision_above"]}
    return {**s, "pass": all(s.values()), "pooled_kappa_gain": pooled, "train_kappa_gain": tr, "dev_kappa_gain": dv, "false_pass_change": d_fp, "false_rejection_change": d_fr,
            "class_rise_limit_count": limit, "flips_fixing_error": fix, "flips_creating_error": brk}


def paired_bootstrap(groups: Sequence[str], y: Sequence[int], a0: Sequence[int], other: Sequence[int], draws: int = 2000) -> list[float] | None:
    """95% document-grouped paired bootstrap interval of kappa(other) - kappa(a0); REPORTED, never gating. None if too few usable draws."""
    rng = np.random.default_rng(SEED)
    members: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    keys = sorted(members)
    diffs = []
    for _ in range(draws):
        idx = [i for g in rng.choice(keys, size=len(keys)) for i in members[g]]
        yy = [y[i] for i in idx]
        k0, k1 = _k(yy, [a0[i] for i in idx]), _k(yy, [other[i] for i in idx])
        if k0 is not None and k1 is not None:
            diffs.append(k1 - k0)
    if len(diffs) < draws // 2:
        return None
    diffs.sort()
    return [diffs[int(0.025 * len(diffs))], diffs[max(0, int(0.975 * len(diffs)) - 1)]]

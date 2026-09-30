"""Frozen protocol: regularized MiniCheck + v18 combiner (docs/ROUTEBENCH_COMBINER_PROTOCOL.md).

Stages (each refuses to overwrite its output):
    judge-train  v18 on the 129 headline TRAIN cases, hard USD cap, 129/129 coverage required
    fit          document-grouped CV on TRAIN -> choose (C, threshold) -> refit on all TRAIN
    dev          ONE evaluation on the headline dev cases with the frozen model; no re-tuning

Nothing here touches the test split. Every constant below is part of the frozen protocol; editing
one after the protocol hash was recorded makes every stage refuse to run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

from .dataset import HEADLINE_PROVENANCE
from .experiments import JudgmentCache, SpendMeter, _cache_key, run_variant
from .run_calibration import DATA, GRID, _dev_cases, _load
from .splits import select

FOLD_SEED = 20260929
N_FOLDS = 5
C_GRID = (0.01, 0.1, 1.0, 10.0)
THRESHOLD_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 .. 0.95
CLIP = (1e-3, 1 - 1e-3)  # MiniCheck probabilities are clipped before the logit
FEATURES = ("mc_full_logit", "mc_minsent_logit", "v18_score")
V18 = "v18-reasoning-o4mini"
V18_HASH = "12fc8b550d954359"
OUT = DATA / "stack_minicheck"
PROTOCOL_DOC = Path(__file__).parents[2] / "docs" / "ROUTEBENCH_COMBINER_PROTOCOL.md"
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_N = 2000


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, *CLIP)
    return np.log(p / (1 - p))


def fold_assignment(groups: list[str]) -> dict[str, int]:
    """Deterministic document->fold map: sorted ids, seeded shuffle, round-robin."""
    docs = sorted(set(groups))
    order = np.random.default_rng(FOLD_SEED).permutation(len(docs))
    return {docs[i]: rank % N_FOLDS for rank, i in enumerate(order)}


def _headline(corpus: Any, sp: Any, split: str) -> list[Any]:
    labels = corpus.resolved_labels()
    return [c for c in select(corpus.cases, sp, split)
            if c.case_id in labels and labels[c.case_id].provenance in HEADLINE_PROVENANCE]


def _features(cases: list[Any], mc: dict[str, Any], v18: dict[str, float]) -> np.ndarray:
    full = np.array([mc[c.case_id]["full"] for c in cases])
    mn = np.array([min(mc[c.case_id]["sent"]) for c in cases])
    return np.column_stack([_logit(full), _logit(mn), np.array([v18[c.case_id] for c in cases])])


def _kappa(y: np.ndarray, p: np.ndarray) -> float:
    if len(set(p.tolist())) < 2 and len(set(y.tolist())) < 2:
        return float("nan")
    return float(cohen_kappa_score(y, p))


def _fit(X: np.ndarray, y: np.ndarray, c: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(X)  # preprocessing fitted on the training rows only
    model = LogisticRegression(C=c, max_iter=1000).fit(scaler.transform(X), y)
    return scaler, model


def protocol_hash() -> str:
    consts = json.dumps([FOLD_SEED, N_FOLDS, C_GRID, THRESHOLD_GRID, CLIP, FEATURES, V18_HASH,
                         BOOTSTRAP_SEED, BOOTSTRAP_N])
    blob = Path(__file__).read_text(encoding="utf8") + consts
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _check_frozen() -> None:
    text = PROTOCOL_DOC.read_text(encoding="utf8") if PROTOCOL_DOC.exists() else ""
    if f"protocol_hash: {protocol_hash()}" not in text:
        raise SystemExit("protocol hash not recorded in the protocol doc (or code changed); refusing")


def _load_mc(name: str) -> dict[str, Any]:
    return {r["id"]: r for r in json.loads((DATA / name).read_text(encoding="utf8"))}


def _v18_scores(cases: list[Any], cfg: Any, cache: JudgmentCache) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in cases:
        j = cache.get(_cache_key(c, cfg))
        if j is None or not j.usable:
            raise SystemExit(f"missing/unusable v18 judgment for {c.case_id}; incomplete, stopping")
        out[c.case_id] = float(j.raw_score)
    return out


def cmd_judge_train(cap: float) -> int:
    corpus, sp, _ = _load()
    cfg = next(c for c in GRID if c.variant_id == V18)
    assert cfg.config_hash == V18_HASH, "v18 config changed"
    cases = _headline(corpus, sp, "train")
    cache = JudgmentCache(DATA / "judgment_cache.jsonl")
    meter = SpendMeter(cap_usd=cap, reserve_usd=0.02)
    _, summary = run_variant(cases, cfg, cache=cache, meter=meter, concurrency=cfg.concurrency or 4)
    print(json.dumps({"n_cases": summary.n_cases, "n_ok": summary.n_ok, "errors": summary.errors,
                      "spend": summary.spend}, indent=1))
    OUT.mkdir(exist_ok=True)
    (OUT / "judge_train_summary.json").write_text(json.dumps(summary.as_dict(), indent=1),
                                                  encoding="utf8")
    return 0 if summary.n_ok == len(cases) and not summary.errors else 3


def cmd_fit() -> int:
    _check_frozen()
    target = OUT / "fit_result.json"
    if target.exists():
        raise SystemExit("fit_result.json exists; refusing to overwrite")
    corpus, sp, _ = _load()
    labels = corpus.resolved_labels()
    cfg = next(c for c in GRID if c.variant_id == V18)
    cases = _headline(corpus, sp, "train")
    v18 = _v18_scores(cases, cfg, JudgmentCache(DATA / "judgment_cache.jsonl"))
    X = _features(cases, _load_mc("minicheck_flan_t5_large_train_scores.json"), v18)
    y = np.array([labels[c.case_id].label for c in cases])
    fold_of = fold_assignment([c.group_id for c in cases])
    fold = np.array([fold_of[c.group_id] for c in cases])
    rows = []
    for c in C_GRID:
        oof = np.zeros(len(y))
        for k in range(N_FOLDS):
            tr, te = fold != k, fold == k
            sc, m = _fit(X[tr], y[tr], c)  # scaler refit inside every training fold
            oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        for t in THRESHOLD_GRID:
            rows.append({"C": c, "threshold": t, "oof_kappa": _kappa(y, (oof >= t).astype(int))})
    ok = [r for r in rows if r["oof_kappa"] == r["oof_kappa"]]
    best = max(ok, key=lambda r: (round(r["oof_kappa"], 12), -r["C"], -abs(r["threshold"] - 0.5)))
    sc, m = _fit(X, y, best["C"])
    baselines = {
        "v18_alone_train_kappa": _kappa(y, (X[:, 2] >= 1.0).astype(int)),
        "minicheck_full_alone_train_kappa_at_0.5": _kappa(y, (X[:, 0] >= 0).astype(int)),
    }
    result = {"protocol_hash": protocol_hash(), "n_train": len(y), "n_docs": len(set(fold_of)),
              "fold_sizes": [int((fold == k).sum()) for k in range(N_FOLDS)], "selected": best,
              "grid": rows, "baselines_on_train": baselines,
              "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
              "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}
    OUT.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=1), encoding="utf8")
    print(json.dumps({k: result[k] for k in ("selected", "coef", "intercept", "baselines_on_train",
                                             "fold_sizes")}, indent=1))
    return 0


def _ci(a: list[float]) -> list[float]:
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


def cmd_dev() -> int:
    _check_frozen()
    target = OUT / "dev_result.json"
    if target.exists():
        raise SystemExit("dev_result.json exists: dev is evaluated ONCE under this protocol")
    fit = json.loads((OUT / "fit_result.json").read_text(encoding="utf8"))
    if fit["protocol_hash"] != protocol_hash():
        raise SystemExit("fit was produced under a different protocol hash")
    corpus, sp, _ = _load()
    labels = corpus.resolved_labels()
    cfg = next(c for c in GRID if c.variant_id == V18)
    cases = _dev_cases(corpus, sp, "headline")
    v18 = _v18_scores(cases, cfg, JudgmentCache(DATA / "judgment_cache.jsonl"))
    X = _features(cases, _load_mc("minicheck_flan_t5_large_dev_scores.json"), v18)
    y = np.array([labels[c.case_id].label for c in cases])
    z = (X - np.array(fit["scaler_mean"])) / np.array(fit["scaler_scale"])
    prob = 1 / (1 + np.exp(-(z @ np.array(fit["coef"]) + fit["intercept"])))
    t = fit["selected"]["threshold"]
    pred = (prob >= t).astype(int)
    base_v18 = (X[:, 2] >= 1.0).astype(int)  # v18 at its frozen (all-criteria-pass) threshold
    base_mc = (X[:, 0] >= 0).astype(int)     # MiniCheck at its own 0.5
    groups = np.array([c.group_id for c in cases])
    ug = sorted(set(groups.tolist()))
    idx = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    d_v18: list[float] = []
    d_mc: list[float] = []
    for _ in range(BOOTSTRAP_N):
        ks = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        k0 = _kappa(y[ks], pred[ks])
        d_v18.append(k0 - _kappa(y[ks], base_v18[ks]))
        d_mc.append(k0 - _kappa(y[ks], base_mc[ks]))
    fixed = [c.case_id for c, p, b, yy in zip(cases, pred, base_v18, y) if b != yy and p == yy]
    new = [c.case_id for c, p, b, yy in zip(cases, pred, base_v18, y) if b == yy and p != yy]
    result = {"protocol_hash": protocol_hash(), "n": len(y), "coverage": f"{len(y)}/{len(cases)}",
              "threshold": t, "combiner_dev_kappa": _kappa(y, pred),
              "v18_dev_kappa_frozen_threshold": _kappa(y, base_v18),
              "minicheck_dev_kappa_at_0.5": _kappa(y, base_mc),
              "combiner_errors": int((pred != y).sum()), "v18_errors": int((base_v18 != y).sum()),
              "corrected_vs_v18": fixed, "new_vs_v18": new,
              "paired_ci_vs_v18": _ci(d_v18), "paired_ci_vs_minicheck": _ci(d_mc),
              "note": "dev is NOT independent: inspected and tuned on repeatedly in earlier rounds"}
    target.write_text(json.dumps(result, indent=1), encoding="utf8")
    print(json.dumps(result, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if a[:1] == ["judge-train"]:
        _check_frozen()
        return cmd_judge_train(float(a[1]) if len(a) > 1 else 1.5)
    if a[:1] == ["fit"]:
        return cmd_fit()
    if a[:1] == ["dev"]:
        return cmd_dev()
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen HHEM-2.1-Open screening report (docs/ROUTEBENCH_HHEM_SCREEN_PROTOCOL.md).

    export  write .local/hhem/pairs.json for the 128 dev cases and the 320 already-inspected comparison rows
    report  compare the frozen HHEM decision (P(consistent) >= 0.5) with cached v18 (all-criteria-pass); refuses to overwrite

Screening only: no threshold search, no ensemble, no fitting. Never touches the 30 reserve documents or the RouteBench test split.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from . import upstream_validation as uv
from .experiments import JudgmentCache, _cache_key
from .run_calibration import DATA, GRID, _dev_cases, _load

WORK = Path(__file__).parents[2] / ".local" / "hhem"
PROTOCOL_DOC = Path(__file__).parents[2] / "docs" / "ROUTEBENCH_HHEM_SCREEN_PROTOCOL.md"
HHEM_REVISION = "8e4a2e6e96c708cc76c2344f7e4757df2515292c"
FLAN_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
DECISION_THRESHOLD = 0.5
BOOT_SEED, BOOT_N = 20261005, 2000
V18 = "v18-reasoning-o4mini"


def protocol_hash() -> str:
    here = Path(__file__).read_text(encoding="utf8")
    scorer = (Path(__file__).parent / "hhem_score.py").read_text(encoding="utf8")
    consts = json.dumps([HHEM_REVISION, FLAN_REVISION, DECISION_THRESHOLD, BOOT_SEED, BOOT_N])
    return hashlib.sha256((here + scorer + consts).encode()).hexdigest()[:16]


def _check_frozen() -> None:
    text = PROTOCOL_DOC.read_text(encoding="utf8") if PROTOCOL_DOC.exists() else ""
    if f"protocol_hash: {protocol_hash()}" not in text:
        raise SystemExit("protocol hash not recorded in the protocol doc (or code changed); refusing")


def _sets() -> tuple[list[Any], list[str], list[Any], list[dict[str, Any]]]:
    corpus, sp, _ = _load()
    dev = _dev_cases(corpus, sp, "headline")
    dev_labels = corpus.resolved_labels()
    rows = uv.load_rows()
    fresh = [uv.to_case(r, i) for i, r in enumerate(rows)]
    return dev, [str(dev_labels[c.case_id].label) for c in dev], fresh, rows


def cmd_export() -> int:
    _check_frozen()
    dev, _, fresh, _ = _sets()
    WORK.mkdir(parents=True, exist_ok=True)
    pairs = [{"id": c.case_id, "premise": c.context, "hypothesis": c.candidate_output.strip()} for c in dev + fresh]
    assert len({p["id"] for p in pairs}) == len(pairs) == 128 + 320
    (WORK / "pairs.json").write_text(json.dumps(pairs), encoding="utf8")
    print(f"exported {len(pairs)} pairs")
    return 0


def _k(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    return uv._kappa_w(y, p, np.ones(len(y)) if w is None else w)


def cmd_report() -> int:
    _check_frozen()
    target = WORK / "report.json"
    if target.exists():
        raise SystemExit("report.json exists; refusing to overwrite")
    raw = json.loads((WORK / "scores.json").read_text(encoding="utf8"))
    scores = raw["scores"]
    dev, dev_y, fresh, rows = _sets()
    cfg = next(c for c in GRID if c.variant_id == V18)
    cache_dev = JudgmentCache(DATA / "judgment_cache.jsonl")
    cache_fresh = JudgmentCache(uv.CACHE)

    def v18_pred(cases: list[Any], cache: JudgmentCache) -> np.ndarray:
        out = []
        for c in cases:
            j = cache.get(_cache_key(c, cfg))
            if j is None or not j.usable:
                raise SystemExit(f"missing v18 judgment for {c.case_id}")
            out.append(int(float(j.raw_score) >= uv.PASS_THRESHOLD))
        return np.array(out)

    result: dict[str, Any] = {"protocol_hash": protocol_hash(), "hhem_revision": HHEM_REVISION,
                              "flan_t5_base_revision": FLAN_REVISION, "seconds": raw["seconds"],
                              "max_prompt_tokens": raw["max_tokens"], "n_prompts_over_512_tokens": raw["n_over_512"],
                              "truncation": "none applied by the frozen procedure", "device": raw["device"],
                              "contamination_warning": ("HHEM-2.1-Open's training data are not documented; overlap with "
                                                        "AggreFact/TofuEval/RAGTruth cannot be excluded. Screening only.")}
    # ---- dev (128) -------------------------------------------------------------------------------------------
    miss = [c.case_id for c in dev + fresh if c.case_id not in scores]
    if miss:
        result["status"] = "INCOMPLETE"
        result["missing"] = miss
        target.write_text(json.dumps(result, indent=1), encoding="utf8")
        print(json.dumps(result, indent=1))
        return 3
    result["coverage"] = f"{len(dev) + len(fresh)}/{len(dev) + len(fresh)}"
    y = np.array([int(v) for v in dev_y])
    h = np.array([int(scores[c.case_id] >= DECISION_THRESHOLD) for c in dev])
    b = v18_pred(dev, cache_dev)
    groups = np.array([c.group_id for c in dev])
    ug = sorted(set(groups.tolist()))
    idx = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(BOOT_SEED)
    d = []
    for _ in range(BOOT_N):
        ks = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        d.append(_k(y[ks], h[ks]) - _k(y[ks], b[ks]))
    result["dev"] = {
        "n": len(y), "hhem_kappa": _k(y, h), "v18_kappa": _k(y, b),
        "hhem_balanced_accuracy": float(balanced_accuracy_score(y, h)), "v18_balanced_accuracy": float(balanced_accuracy_score(y, b)),
        "hhem_errors": int((h != y).sum()), "v18_errors": int((b != y).sum()),
        "corrected_vs_v18": int(((b != y) & (h == y)).sum()), "new_vs_v18": int(((b == y) & (h != y)).sum()),
        "hhem_confusion": uv._conf(y, h), "paired_kappa_diff_vs_v18_ci95":
            [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
        "note": "dev is not independent (inspected repeatedly); v18's dev kappa was dev-selected",
    }
    # ---- fresh comparison set (320) --------------------------------------------------------------------------
    yf = np.array([r["label"] for r in rows])
    hf = np.array([int(scores[c.case_id] >= DECISION_THRESHOLD) for c in fresh])
    bf = v18_pred(fresh, cache_fresh)
    w = np.array([r["weight"] for r in rows])
    src = np.array([r["source"] for r in rows])
    doc = np.array([r["doc_id"] for r in rows])
    by_src = {s: sorted(set(doc[src == s].tolist())) for s in sorted(set(src.tolist()))}
    idxf = {x: np.where(doc == x)[0] for x in doc}
    rng = np.random.default_rng(BOOT_SEED)
    dw, dh_, db_ = [], [], []
    for _ in range(BOOT_N):
        ks = np.concatenate([idxf[x] for s, ds in by_src.items() for x in rng.choice(ds, len(ds))])
        if len(set(yf[ks].tolist())) < 2:
            continue
        dw.append(_k(yf[ks], hf[ks], w[ks]) - _k(yf[ks], bf[ks], w[ks]))
        dh_.append(_k(yf[ks], hf[ks], w[ks]))
        db_.append(_k(yf[ks], bf[ks], w[ks]))
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    per_source = {s: {"hhem_weighted_kappa": _k(yf[src == s], hf[src == s], w[src == s]),
                      "v18_weighted_kappa": _k(yf[src == s], bf[src == s], w[src == s]),
                      "hhem_confusion": uv._conf(yf[src == s], hf[src == s])} for s in by_src}
    result["fresh_320"] = {
        "n": len(yf), "hhem_weighted_kappa": _k(yf, hf, w), "hhem_weighted_kappa_ci95": ci(dh_),
        "v18_weighted_kappa": _k(yf, bf, w), "v18_weighted_kappa_ci95": ci(db_),
        "paired_weighted_kappa_diff_vs_v18_ci95": ci(dw),
        "hhem_unweighted_kappa": _k(yf, hf), "v18_unweighted_kappa": _k(yf, bf),
        "hhem_balanced_accuracy_unweighted": float(balanced_accuracy_score(yf, hf)),
        "v18_balanced_accuracy_unweighted": float(balanced_accuracy_score(yf, bf)),
        "hhem_weighted_rates": uv._rates(yf, hf, w), "v18_weighted_rates": uv._rates(yf, bf, w),
        "hhem_confusion": uv._conf(yf, hf), "v18_confusion": uv._conf(yf, bf),
        "hhem_errors": int((hf != yf).sum()), "v18_errors": int((bf != yf).sum()),
        "corrected_vs_v18": int(((bf != yf) & (hf == yf)).sum()), "new_vs_v18": int(((bf == yf) & (hf != yf)).sum()),
        "per_source": per_source,
        "note": "already-inspected comparison set (v18 was judged on it); documents are evaluation-only",
    }
    result["status"] = "COMPLETE"
    result["note"] = ("Screening result only. Even a strong number does not certify the original RouteBench target: "
                      "training-data overlap is unknown and these sets are not the sealed test split.")
    target.write_text(json.dumps(result, indent=1), encoding="utf8")
    print(json.dumps(result, indent=1))
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a[:1] == ["export"]:
        return cmd_export()
    if a[:1] == ["report"]:
        return cmd_report()
    if a[:1] == ["hash"]:
        print(protocol_hash())
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

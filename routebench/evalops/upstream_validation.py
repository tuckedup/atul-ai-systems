"""Frozen v18 measurement on the TofuEval upstream-DEV sample (docs/ROUTEBENCH_UPSTREAM_REPORTING_RULE.md).

Separate from the original RouteBench result: own cases, own judgment cache, own output. Evaluation only:
nothing here fits, tunes, selects a threshold, or reads a RouteBench split.

    judge   frozen v18 on the 320 PRIMARY manifest rows (hard USD cap, complete coverage or 'incomplete')
    report  one report; refuses to overwrite

Reporting rule (frozen before judging):
  PRIMARY  sentence-population kappa from inverse-inclusion-probability-weighted confusion counts
           (w = 1 / inclusion_prob from the manifest), pooled over both sources, with a document bootstrap:
           within each source resample its 20 sampled documents with replacement (seed 20261004, 2000 draws),
           keeping each row's weight, and recompute the weighted kappa. 95% percentile interval.
  SECONDARY unweighted sample kappa (all 320 rows equal), sensitivity, specificity, balanced accuracy,
           per-source results, confusion counts, false-pass / false-rejection split.
Decision: v18 'supported' iff raw_score >= 1.0 (its all-criteria-pass rule, unchanged).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import CaseRecord
from .experiments import JudgmentCache, SpendMeter, _cache_key, run_variant
from .run_calibration import DATA, GRID

BOOT_SEED, BOOT_N = 20261004, 2000
V18, V18_HASH, PASS_THRESHOLD = "v18-reasoning-o4mini", "12fc8b550d954359", 1.0
MANIFEST = DATA / "tofueval_upstream_dev_sample_manifest.json"
DOCS = DATA / "raw" / "tofueval_upstream" / "docs.jsonl"
OUT = DATA / "upstream_validation"
CACHE = OUT / "judgments.jsonl"
RULE_DOC = Path(__file__).parents[2] / "docs" / "ROUTEBENCH_UPSTREAM_REPORTING_RULE.md"
CSV_FILES = {"MediaSum": "tofueval_mediasum_factual_eval_dev.csv",
             "MeetingBank": "tofueval_meetingbank_factual_eval_dev.csv"}
TASK_INPUT = ("Summarise the source document faithfully. Every statement in your summary must be "
              "supported by the document.")


def protocol_hash() -> str:
    consts = json.dumps([BOOT_SEED, BOOT_N, V18_HASH, PASS_THRESHOLD])
    manifest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    return hashlib.sha256((Path(__file__).read_text(encoding="utf8") + consts + manifest).encode()).hexdigest()[:16]


def _check_frozen() -> None:
    text = RULE_DOC.read_text(encoding="utf8") if RULE_DOC.exists() else ""
    if f"protocol_hash: {protocol_hash()}" not in text:
        raise SystemExit("protocol hash not recorded in the reporting-rule doc (or code/manifest changed); refusing")


def load_rows() -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    docs = {(d["source"], d["doc_id"]): d for d in
            (json.loads(line) for line in DOCS.read_text(encoding="utf8").splitlines() if line.strip())}
    rows: list[dict[str, Any]] = []
    for src, fn in CSV_FILES.items():
        with (DATA / "raw" / "authoritative" / fn).open(encoding="utf8", newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        for p in manifest["sources"][src]["primary"]:
            r = csv_rows[p["row_index"]]
            assert r["doc_id"] == p["doc_id"] and r["sent_idx"] == p["sent_idx"]
            doc = docs[(src, p["doc_id"])]
            assert doc["upstream_part"] == "dev"
            rows.append({"source": src, "doc_id": p["doc_id"], "label": {"yes": 1, "no": 0}[r["sent_label"].strip().lower()],
                         "weight": 1.0 / p["inclusion_prob"], "text": doc["text"], "claim": r["summ_sent"].strip(),
                         "row_index": p["row_index"]})
    return rows


def to_case(r: dict[str, Any], i: int) -> CaseRecord:
    return CaseRecord(
        case_id=f"upv-{r['source'][:5].lower()}-{r['row_index']}", group_id=f"upv-{r['doc_id']}",
        task_class="summarize", task_input=TASK_INPUT, context=r["text"], candidate_output=r["claim"],
        generator_id=f"published-summary-system:TofuEval-{r['source']}", source=f"TofuEval-upstream-dev/{r['source']}",
        source_license="MIT-0 annotations; evaluation-only; transcripts CC-BY-NC-SA-4.0 (local, not redistributed)",
        meta={"source_row_index": r["row_index"], "subset": f"TofuEval-{r['source']}-upstream-dev"})


def cmd_judge(cap: float) -> int:
    _check_frozen()
    OUT.mkdir(exist_ok=True)
    cfg = next(c for c in GRID if c.variant_id == V18)
    assert cfg.config_hash == V18_HASH, "v18 config changed"
    rows = load_rows()
    cases = [to_case(r, i) for i, r in enumerate(rows)]
    assert len(cases) == 320 and len({c.case_id for c in cases}) == 320
    meter = SpendMeter(cap_usd=cap, reserve_usd=0.02)
    _, summary = run_variant(cases, cfg, cache=JudgmentCache(CACHE), meter=meter, concurrency=cfg.concurrency or 4)
    s = summary.as_dict()
    (OUT / "judge_summary.json").write_text(json.dumps(s, indent=1), encoding="utf8")
    print(json.dumps({"n_cases": s["n_cases"], "n_ok": s["n_ok"], "errors": s["errors"], "spend": s["spend"]}, indent=1))
    return 0 if s["n_ok"] == len(cases) and not s["errors"] else 3


def _kappa_w(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    tp = w[(p == 1) & (y == 1)].sum()
    fp = w[(p == 1) & (y == 0)].sum()
    fn = w[(p == 0) & (y == 1)].sum()
    tn = w[(p == 0) & (y == 0)].sum()
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def _rates(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    tp = w[(p == 1) & (y == 1)].sum()
    fn = w[(p == 0) & (y == 1)].sum()
    tn = w[(p == 0) & (y == 0)].sum()
    fp = w[(p == 1) & (y == 0)].sum()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return {"sensitivity": float(sens), "specificity": float(spec), "balanced_accuracy": float((sens + spec) / 2)}


def _conf(y: np.ndarray, p: np.ndarray) -> dict[str, int]:
    return {"tp": int(((p == 1) & (y == 1)).sum()), "fp": int(((p == 1) & (y == 0)).sum()),
            "fn": int(((p == 0) & (y == 1)).sum()), "tn": int(((p == 0) & (y == 0)).sum())}


def cmd_report() -> int:
    _check_frozen()
    target = OUT / "report.json"
    if target.exists():
        raise SystemExit("report.json exists; refusing to overwrite")
    rows = load_rows()
    cases = [to_case(r, i) for i, r in enumerate(rows)]
    cfg = next(c for c in GRID if c.variant_id == V18)
    cache = JudgmentCache(CACHE)
    scores, missing = [], []
    for c in cases:
        j = cache.get(_cache_key(c, cfg))
        if j is None or not j.usable:
            missing.append(c.case_id)
        else:
            scores.append(float(j.raw_score))
    if missing:
        result = {"status": "INCOMPLETE", "coverage": f"{len(scores)}/{len(cases)}", "missing": missing,
                  "note": "no kappa is reported for an incomplete run"}
        target.write_text(json.dumps(result, indent=1), encoding="utf8")
        print(json.dumps(result, indent=1))
        return 3
    y = np.array([r["label"] for r in rows])
    p = (np.array(scores) >= PASS_THRESHOLD).astype(int)
    w = np.array([r["weight"] for r in rows])
    ones = np.ones(len(y))
    src = np.array([r["source"] for r in rows])
    doc = np.array([r["doc_id"] for r in rows])
    rng = np.random.default_rng(BOOT_SEED)
    by_src = {s: sorted(set(doc[src == s].tolist())) for s in sorted(set(src.tolist()))}
    idx = {d: np.where(doc == d)[0] for d in doc}
    boot_w, boot_u = [], []
    for _ in range(BOOT_N):
        ks = np.concatenate([idx[d] for s, ds in by_src.items() for d in rng.choice(ds, len(ds))])
        if len(set(y[ks].tolist())) < 2:
            continue
        boot_w.append(_kappa_w(y[ks], p[ks], w[ks]))
        boot_u.append(_kappa_w(y[ks], p[ks], ones[ks]))
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    per_source = {}
    for s in by_src:
        m = src == s
        per_source[s] = {"n": int(m.sum()), "unweighted_kappa": _kappa_w(y[m], p[m], ones[m]),
                         "weighted_kappa": _kappa_w(y[m], p[m], w[m]), "confusion": _conf(y[m], p[m]),
                         **{k: v for k, v in _rates(y[m], p[m], w[m]).items()},
                         "supported_rate_labels_weighted": float((w[m] * y[m]).sum() / w[m].sum())}
    result = {
        "status": "COMPLETE", "protocol_hash": protocol_hash(), "coverage": f"{len(scores)}/{len(cases)}",
        "primary_weighted_kappa": _kappa_w(y, p, w), "primary_weighted_kappa_ci95": ci(boot_w),
        "unweighted_kappa": _kappa_w(y, p, ones), "unweighted_kappa_ci95": ci(boot_u),
        "weighted_rates": _rates(y, p, w), "unweighted_rates": _rates(y, p, ones),
        "confusion_unweighted": _conf(y, p), "per_source": per_source,
        "supported_rate_labels_weighted": float((w * y).sum() / w.sum()),
        "effective_sample_size_weighted": float(w.sum() ** 2 / (w**2).sum()),
        "bootstrap": {"seed": BOOT_SEED, "draws_used": len(boot_w)},
        "note": ("separate from the original RouteBench result; evaluation-only on fresh upstream-dev documents; "
                 "not the original benchmark's target")}
    target.write_text(json.dumps(result, indent=1), encoding="utf8")
    print(json.dumps(result, indent=1))
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a[:1] == ["judge"]:
        return cmd_judge(float(a[1]) if len(a) > 1 else 3.5)
    if a[:1] == ["report"]:
        return cmd_report()
    if a[:1] == ["hash"]:
        print(protocol_hash())
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

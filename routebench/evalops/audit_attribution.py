"""Read-only attribution preflight: no provider calls, no label/prompt changes.

Run before spending on a new experiment. A source match only checks local data
transport; it does not adjudicate the truth of a published annotation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .dataset import CaseRecord, Corpus
from .metrics import Confusion, kappa_of
from .splits import SplitPlan, select, verify


def missing_perfect_upper_bound(confusion: dict[str, int], missing: int) -> float | None:
    """Optimistic bound across all label mixes, assuming every missing answer correct.

    This is a mathematical bound, not imputation and not an observed kappa.
    """
    if missing < 0:
        raise ValueError("missing cannot be negative")
    values = []
    for positives in range(missing + 1):
        c = Confusion(tp=confusion["tp"] + positives, fp=confusion["fp"],
                      fn=confusion["fn"], tn=confusion["tn"] + missing - positives)
        kappa = kappa_of(c)[0]
        if kappa is not None:
            values.append(kappa)
    return max(values) if values else None


def audit(corpus: Corpus, splits: SplitPlan, raw: list[dict[str, Any]],
          exposed: list[dict[str, Any]], bundle: dict[str, Any],
          report: dict[str, Any]) -> dict[str, Any]:
    verify(corpus.cases, splits)
    labels = corpus.resolved_labels()
    # Index candidate content first, then verify document prefixes. IDs differ
    # across experiments and cannot be used to detect cross-experiment reuse.
    index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in raw:
        index[(row["subset"], row["claim"].strip())].append(row["doc"].strip())
    truncated, unmatched = [], []
    for case in corpus.cases:
        subset = case.source.removeprefix("LLM-AggreFact/")
        docs = index.get((subset, case.candidate_output.strip()), [])
        if case.context in docs:
            continue
        longer = [d for d in docs if case.context and d.startswith(case.context)]
        if longer:
            truncated.append({"case_id": case.case_id, "subset": subset,
                              "split": splits.of_case(case), "label": labels[case.case_id].label,
                              "supplied_chars": len(case.context),
                              "raw_chars": min(map(len, longer))})
        else:
            unmatched.append(case.case_id)
    docs = {c["context"].strip() for c in exposed if c.get("context", "").strip()}
    pairs = {(c["context"].strip(), c["candidate_output"].strip()) for c in exposed}
    exposure = {}
    for split in ("train", "dev", "test"):
        cases = select(corpus.cases, splits, split)
        same_doc = [c.case_id for c in cases if c.context.strip() in docs]
        same_pair = [c.case_id for c in cases
                     if (c.context.strip(), c.candidate_output.strip()) in pairs]
        exposure[split] = {"total": len(cases), "document_overlap": same_doc,
                           "document_claim_overlap": same_pair}
    expected = {"dataset_hash": corpus.cases_hash, "annotations_hash": corpus.annotations_hash,
                "split_membership_hash": splits.membership_hash, "split_seed": splits.seed}
    binding_errors = {k: {"recorded": bundle.get(k), "expected": v}
                      for k, v in expected.items() if bundle.get(k) != v}
    completion = report.get("completion", {})
    track = report.get("tracks", {}).get("human_judgment_of_response", {})
    missing = completion.get("unjudged", 0)
    upper = (missing_perfect_upper_bound(track["confusion"], missing)
             if track.get("confusion") else None)
    blockers = []
    for condition, code in (
        (bool(truncated), "source_truncated"), (bool(unmatched), "source_unmatched"),
        (bool(exposure["test"]["document_overlap"]), "test_previously_exposed"),
        (bool(binding_errors), "missing_or_stale_bindings"),
        (completion.get("completion_rate", 0) != 1, "test_incomplete"),
    ):
        if condition:
            blockers.append(code)
    return {"n_cases": len(corpus.cases), "blockers": blockers,
            "truncation": {"total": len(truncated),
                           "positive": sum(r["label"] == 1 for r in truncated),
                           "by_split": dict(Counter(r["split"] for r in truncated)),
                           "cases": truncated},
            "unmatched_source_cases": unmatched, "prior_probe_exposure": exposure,
            "binding_errors": binding_errors, "completion": completion,
            "reported_kappa": track.get("kappa"),
            "optimistic_missing_correct_bound": upper,
            "note": "Read-only diagnostic; no labels changed; bound is not a measured score."}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    args = p.parse_args(argv)
    root, attr = args.data_root, args.data_root / "attribution"
    corpus, splits = Corpus.load(attr), SplitPlan.load(attr / "splits.json")
    raw = [json.loads(s) for s in (root / "raw/aggrefact.jsonl").read_text(
        encoding="utf-8").splitlines() if s.strip()]
    exposed = []
    for path in (root / "probe").glob("cases_*.jsonl"):
        exposed.extend(CaseRecord.model_validate_json(s).model_dump() for s in path.read_text(
            encoding="utf-8").splitlines() if s.strip())
    bundle = json.loads((attr / "calibration_bundle.json").read_text(encoding="utf-8"))
    report = json.loads((attr / "calibration_report.json").read_text(encoding="utf-8"))
    result = audit(corpus, splits, raw, exposed, bundle, report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare source-complete TRAIN demonstrations with published expert reasons.

Offline only. Never joins dev/test labels or selects by judge errors. Ambiguous
sentence joins, label conflicts, cropped documents and oversized examples are
excluded. This is a GEDD-inspired input artifact, not a measured judge improvement.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .dataset import Corpus, LabelProvenance, content_hash
from .splits import SplitPlan, select, verify

MAX_SOURCE = 6000
MAX_CANDIDATE = 700


def _eligible(corpus: Corpus, splits: SplitPlan, notes: list[dict[str, str]],
              raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verify(corpus.cases, splits)
    index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in notes:
        index[(row["subset"], row["summ_sent"].strip())].append(row)
    full_sources = {(r["subset"], r["doc"].strip(), r["claim"].strip()) for r in raw}
    labels = corpus.resolved_labels()
    eligible = []
    for case in select(corpus.cases, splits, "train"):
        ann = labels.get(case.case_id)
        subset = case.source.removeprefix("LLM-AggreFact/")
        rows = index.get((subset, case.candidate_output.strip()), [])
        if (case.task_class != "summarize" or ann is None
                or ann.provenance != LabelProvenance.HUMAN_EXPERT or len(rows) != 1
                or not case.context or len(case.context) > MAX_SOURCE
                or len(case.candidate_output) > MAX_CANDIDATE
                or (subset, case.context, case.candidate_output.strip()) not in full_sources):
            continue
        row = rows[0]
        published = {"yes": 1, "no": 0}.get(row["sent_label"].strip().lower())
        memo = row.get("exp", "").strip()
        codes = row.get("type", "").strip()
        if published != ann.label or (not published and (not memo or not codes)):
            continue
        eligible.append({"case_id": case.case_id, "group_id": case.group_id,
                         "task_input": case.task_input, "context": case.context,
                         "candidate_output": case.candidate_output,
                         "verdict": "PASS" if published else "FAIL",
                         "why": f"{codes}: {memo}" if not published else
                         "Published expert annotation: supported by the supplied document.",
                         "error_types": codes, "source": case.source,
                         "annotation_file": row.get("annotation_file", ""),
                         "annotation_id": row.get("annotation_id", ""),
                         "source_doc_id": row.get("doc_id", "")})
    eligible.sort(key=lambda e: (len(e["context"]), len(e["candidate_output"]), e["case_id"]))
    return eligible


def build(corpus: Corpus, splits: SplitPlan, notes: list[dict[str, str]],
          raw: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = _eligible(corpus, splits, notes, raw)
    chosen, covered = [], set()
    for ex in eligible:
        codes = {c.strip() for c in ex["error_types"].split(",") if c.strip()}
        if ex["verdict"] == "FAIL" and codes - covered:
            chosen.append(ex)
            covered.update(codes)
            if len(chosen) == 6:
                break
    chosen += [e for e in eligible if e["verdict"] == "PASS"][:2]
    if not any(e["verdict"] == "FAIL" for e in chosen) or not any(
            e["verdict"] == "PASS" for e in chosen):
        raise ValueError("need unambiguous, source-complete expert train examples of both classes")
    return {"source_split": "train", "format_version": "source-complete-v2",
            "dataset_hash": corpus.cases_hash, "annotations_hash": corpus.annotations_hash,
            "split_membership_hash": splits.membership_hash, "split_seed": splits.seed,
            "selection_rule": "shortest full source; cover up to six failure types plus two passes; train only",
            "published_notes_hash": content_hash(notes), "raw_source_rows_hash": content_hash(raw),
            "eligible_train_examples": len(eligible), "covered_failure_types": sorted(covered),
            "exemplars": {"summarize": chosen}}


#: Failure explanations that hinge on what the document LEAVES OUT or on how completely it is
#: characterised. The RouteBench rubric says a terse or incomplete response is a PASS as long as
#: what it says is supported, so demonstrating such a FAIL would teach a rule the rubric contradicts.
#: Applied to the whole eligible pool by the same lexicon; a documented exclusion, not a relabel.
COMPLETENESS_MARKERS = re.compile(
    r"not entirely|incomplete|omit|leaves? out|does not (?:fully|only|include all)|"
    r"also includes|more than just|not the only|missing", re.IGNORECASE)


def build_diverse(corpus: Corpus, splits: SplitPlan, notes: list[dict[str, str]],
                  raw: list[dict[str, Any]], *, n_fail: int = 4, n_pass: int = 2) -> dict[str, Any]:
    """A compact packet with ONE example per source document, chosen by a fixed rule.

    Rule (train only; never reads judge output or any dev/test label): from the eligible pool drop
    FAIL examples whose expert memo matches COMPLETENESS_MARKERS; take failure codes in order of
    train frequency and for each pick the shortest full source from a document not yet used;
    then take PASS examples from unused documents, alternating TofuEval subsets, shortest first.
    """
    pool = _eligible(corpus, splits, notes, raw)
    excluded = [e["case_id"] for e in pool
                if e["verdict"] == "FAIL" and COMPLETENESS_MARKERS.search(e["why"])]
    pool = [e for e in pool if e["case_id"] not in set(excluded)]
    freq: dict[str, int] = defaultdict(int)
    for e in pool:
        if e["verdict"] == "FAIL":
            freq[e["error_types"].strip()] += 1
    used: set[str] = set()
    chosen: list[dict[str, Any]] = []
    for code, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
        if len([c for c in chosen if c["verdict"] == "FAIL"]) >= n_fail:
            break
        if "," in code:
            continue
        for e in pool:  # already sorted shortest source first
            if e["verdict"] == "FAIL" and e["error_types"].strip() == code and e["group_id"] not in used:
                chosen.append(e)
                used.add(e["group_id"])
                break
    subsets = ["TofuEval-MeetB", "TofuEval-MediaS"]
    for i in range(n_pass):
        want = subsets[i % 2]
        for e in pool:
            if e["verdict"] == "PASS" and e["group_id"] not in used and e["source"].endswith(want):
                chosen.append(e)
                used.add(e["group_id"])
                break
    if (sum(e["verdict"] == "FAIL" for e in chosen) < n_fail
            or sum(e["verdict"] == "PASS" for e in chosen) < n_pass):
        raise ValueError("could not fill the diverse packet from the eligible train pool")
    doc_hashes = [content_hash(e["context"]) for e in chosen]
    return {"source_split": "train", "format_version": "source-complete-v2",
            "dataset_hash": corpus.cases_hash, "annotations_hash": corpus.annotations_hash,
            "split_membership_hash": splits.membership_hash, "split_seed": splits.seed,
            "selection_rule": (
                "train only; one example per source document; FAIL memos matching "
                "COMPLETENESS_MARKERS excluded; failure codes by train frequency, shortest full "
                "source from an unused document; PASS alternating MeetB/MediaS, shortest first; "
                "never reads judge output or dev/test labels"),
            "completeness_markers": COMPLETENESS_MARKERS.pattern,
            "excluded_completeness_case_ids": sorted(excluded),
            "published_notes_hash": content_hash(notes), "raw_source_rows_hash": content_hash(raw),
            "eligible_train_examples": len(pool), "distinct_documents": len(set(doc_hashes)),
            "document_content_hashes": doc_hashes,
            "covered_failure_types": sorted({e["error_types"] for e in chosen if e["error_types"]}),
            "exemplars": {"summarize": chosen}}


def _load_inputs() -> tuple[Corpus, SplitPlan, list[dict[str, str]], list[dict[str, Any]]]:
    root = Path(__file__).parent / "data"
    corpus, splits = Corpus.load(root), SplitPlan.load(root / "splits.json")
    notes = []
    for path in sorted((root / "raw/authoritative").glob("tofueval_*.csv")):
        subset = "TofuEval-MediaS" if "mediasum" in path.name else "TofuEval-MeetB"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            notes.extend(dict(r, subset=subset, annotation_file=path.name) for r in csv.DictReader(stream))
    raw = [json.loads(s) for s in (root / "raw/aggrefact.jsonl").read_text(
        encoding="utf-8").splitlines() if s.strip()]
    return corpus, splits, notes, raw


def main() -> None:
    import sys
    corpus, splits, notes, raw = _load_inputs()
    if "--diverse" in sys.argv:
        print(json.dumps(build_diverse(corpus, splits, notes, raw), indent=2, sort_keys=True))
        return
    print(json.dumps(build(corpus, splits, notes, raw), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

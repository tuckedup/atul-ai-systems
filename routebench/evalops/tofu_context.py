"""Recover the full generated summary (and topic) around each TofuEval sentence case.

TofuEval annotators judged one summary sentence while reading the WHOLE summary and its requested
topic. The RouteBench mirror kept only the sentence, so a judge could not resolve "This ..." or see
what a sentence was answering. This module rebuilds that context OFFLINE from the pinned published
annotation files. It never touches labels, cases or splits: the output is a separate, versioned
artifact (`data/tofu_context.json`) that only a judge config opting in via `context_path` reads.

A case is joined only when its sentence matches exactly one annotation row, that row's published
label equals the stored label, and the row's summary has contiguous sentence indices. Anything else
is left out and counted -- ambiguity is never resolved by guessing.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .dataset import Corpus, content_hash

SUBSET_FILES = {
    "TofuEval-MediaS": ("tofueval_mediasum_factual_eval_dev.csv",
                        "tofueval_mediasum_factual_eval_test.csv"),
    "TofuEval-MeetB": ("tofueval_meetingbank_factual_eval_dev.csv",
                       "tofueval_meetingbank_factual_eval_test.csv"),
}
VERSION = "tofu-context-v1"


def _rows(raw_dir: Path) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for subset, files in SUBSET_FILES.items():
        rows: list[dict[str, str]] = []
        for name in files:
            with (raw_dir / name).open(encoding="utf8", newline="") as fh:
                for r in csv.DictReader(fh):
                    r["annotation_file"] = name
                    rows.append(r)
        out[subset] = rows
    return out


def build(corpus: Corpus, raw_dir: Path) -> dict[str, Any]:
    labels = corpus.resolved_labels()
    rows = _rows(raw_dir)
    by_sentence: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_summary: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for subset, rs in rows.items():
        for r in rs:
            by_sentence[(subset, r["summ_sent"].strip())].append(r)
            by_summary[(subset, r["annotation_file"], r["doc_id"], r["annotation_id"], r["topic"],
                        r["model_name"])].append(r)
    entries: dict[str, Any] = {}
    skipped: dict[str, int] = defaultdict(int)
    for case in corpus.cases:
        subset = case.source.removeprefix("LLM-AggreFact/")
        if subset not in SUBSET_FILES or case.task_class != "summarize":
            continue
        matches = by_sentence.get((subset, case.candidate_output.strip()), [])
        if len(matches) != 1:
            skipped["ambiguous_or_missing_sentence_join"] += 1
            continue
        row = matches[0]
        ann = labels.get(case.case_id)
        published = {"yes": 1, "no": 0}.get(row["sent_label"].strip().lower())
        if ann is None or published != ann.label:
            skipped["published_label_differs_from_stored"] += 1
            continue
        group = by_summary[(subset, row["annotation_file"], row["doc_id"], row["annotation_id"], row["topic"],
                            row["model_name"])]
        group = sorted(group, key=lambda r: int(r["sent_idx"]))
        idx = [int(r["sent_idx"]) for r in group]
        if idx != list(range(1, len(idx) + 1)):
            skipped["non_contiguous_summary"] += 1
            continue
        entries[case.case_id] = {
            "topic": row["topic"].strip(),
            "sentences": [r["summ_sent"].strip() for r in group],
            "target_index": int(row["sent_idx"]) - 1,
            "source": {k: row[k] for k in ("annotation_file", "doc_id", "annotation_id",
                                           "model_name", "sent_idx")},
        }
    for cid, e in entries.items():
        assert e["sentences"][e["target_index"]] == corpus_case(corpus, cid).candidate_output.strip()
    return {"version": VERSION, "entries": entries, "skipped": dict(skipped),
            "content_hash": content_hash(json.dumps(entries, sort_keys=True))}


def corpus_case(corpus: Corpus, case_id: str) -> Any:
    return next(c for c in corpus.cases if c.case_id == case_id)


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf8"))["entries"]


def main() -> None:
    data = Path(__file__).parent / "data"
    corpus = Corpus.load(data)
    art = build(corpus, data / "raw" / "authoritative")
    (data / "tofu_context.json").write_text(json.dumps(art, indent=1, ensure_ascii=False),
                                            encoding="utf8")
    print(f"{len(art['entries'])} cases with recovered context; skipped {art['skipped']}")


if __name__ == "__main__":
    main()

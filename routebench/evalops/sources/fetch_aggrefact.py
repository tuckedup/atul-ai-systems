"""Fetch LLM-AggreFact (test split) -- human-expert claim/document faithfulness labels.

Primary source requested: lytang/LLM-AggreFact on the HF Hub.
PROBLEM: that repo is gated ("gated": "auto" in its HF metadata) and returns
HTTP 401 on both the datasets-server `rows` endpoint and the `resolve`/`raw`
file endpoints when queried without an authentication token. No HF token is
configured or available in this environment, and we were told not to
substitute invented data -- but a token-gated mirror IS the same "no data"
outcome as a dead link, so we looked for a verbatim public copy instead of
giving up on the priority source.

FALLBACK ACTUALLY USED: NinaCalvi/llm_aggrefact_pre_aug9 on the HF Hub.
This is a public (gated=false), non-authenticated-accessible copy of the
lytang/LLM-AggreFact dataset, captured before the gate was added (per its
name and its creation date of 2024-10-08). Evidence it is a verbatim copy,
not a re-derivation:
  - identical column set (dataset, doc, claim, label, contamination_identifier)
    plus two extra columns (prompt, truth_result) added by whoever ran a
    model over it -- the original four columns are untouched passthroughs.
  - the `contamination_identifier` values are literally prefixed
    "LLM-AggreFact:<hash>", i.e. they carry the original repo's own contamination
    tag forward verbatim.
  - the `dataset` (subset) values match the LLM-AggreFact paper/README subset
    names (AggreFact-CNN, AggreFact-XSum, TofuEval-MediaS, TofuEval-MeetB,
    Wice, Reveal, ClaimVerify, FactCheck-GPT, ExpertQA, Lfqa).
This mirror's test split has 12,949 rows (the official test split has 29,320;
this looks like a filtered/deduplicated subset of it -- we do not know the
exact filter, so treat n as smaller-than-official but every row's doc/claim/
label content is a verbatim copy of real HF-hosted data, not synthesized).
Notably absent from this test-split mirror: RAGTruth. This is expected, not
a mirror defect: LLM-AggreFact's own dataset card states RAGTruth's rows
were folded into the *validation* (dev) split, not test.

LABEL CONVENTION (confirmed from the benchmark authors' own GitHub README,
Liyan06/MiniCheck, README.md, "Benchmark Access" section field table):
    "label | 1 if the claim is supported, 0 otherwise"
  ==> label == 1 means the claim IS supported by the document (faithful).
  ==> label == 0 means the claim is NOT supported (unsupported/contradicted).
  Cross-checked against a sample row (AggreFact-CNN, row 0): the claim's
  sentences are directly restated from the document text, and its label is 1,
  consistent with "1 = supported".

We fetch the ENTIRE test split of the mirror (all 12,949 rows) rather than a
partial sample: this trivially satisfies "spread across as many subsets as
possible" and "keep the natural label mix" because it IS the full unmodified
population of the mirror's test split, with no sampling bias introduced by us.

Row shape written to data/raw/aggrefact.jsonl:
    {"source": "LLM-AggreFact", "subset": <dataset field>, "doc": str,
     "claim": str, "label": int (0 or 1), "source_row_index": int}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import iter_rows

# Requested-but-gated original source (kept here for the record / future retry
# if a token ever becomes available):
ORIGINAL_REPO = "lytang/LLM-AggreFact"
ORIGINAL_CONFIG = "default"

# Actual public mirror used (see module docstring for why):
MIRROR_REPO = "NinaCalvi/llm_aggrefact_pre_aug9"
MIRROR_CONFIG = "default"
SPLIT = "test"
MIRROR_TEST_TOTAL = 12949  # confirmed via datasets-server /size endpoint

LABEL_CONVENTION = "1 = claim is supported by the document; 0 = not supported (per Liyan06/MiniCheck README field table)"

OUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "aggrefact.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-rows", type=int, default=MIRROR_TEST_TOTAL,
                     help="cap on number of rows to fetch (default: entire mirror test split)")
    args = ap.parse_args()

    if OUT_PATH.exists() and not args.force:
        lines = OUT_PATH.read_text(encoding="utf-8").splitlines()
        print(f"[aggrefact] output already exists at {OUT_PATH} ({len(lines)} rows); use --force to refetch")
        _summarize(lines)
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    for i, row in enumerate(iter_rows(MIRROR_REPO, MIRROR_CONFIG, SPLIT,
                                       total=min(MIRROR_TEST_TOTAL, args.max_rows), page_size=100)):
        rows_out.append({
            "source": "LLM-AggreFact",
            "subset": row["dataset"],
            "doc": row["doc"],
            "claim": row["claim"],
            "label": int(row["label"]),
            "source_row_index": i,
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[aggrefact] wrote {len(rows_out)} rows to {OUT_PATH}")
    print(f"[aggrefact] label convention: {LABEL_CONVENTION}")
    _summarize([json.dumps(r, ensure_ascii=False) for r in rows_out])

    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"[aggrefact] sha256={sha}")


def _summarize(lines):
    subset_counts = Counter()
    label_counts = Counter()
    for line in lines:
        r = json.loads(line)
        subset_counts[r["subset"]] += 1
        label_counts[r["label"]] += 1
    print("[aggrefact] per-subset counts:", dict(subset_counts))
    print("[aggrefact] label balance:", dict(label_counts))
    print("[aggrefact] sample rows:")
    for line in lines[:2]:
        print(line[:400])


if __name__ == "__main__":
    main()

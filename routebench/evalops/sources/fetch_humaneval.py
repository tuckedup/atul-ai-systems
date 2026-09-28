"""Fetch HumanEval (openai/openai_humaneval, split=test) via HF datasets-server.

Source: https://huggingface.co/datasets/openai/openai_humaneval (MIT license,
public, no gate). All 164 rows.
Row shape written to data/raw/humaneval.jsonl:
    {"source": "HumanEval", "task_id": str, "prompt": str,
     "canonical_solution": str, "test": str, "entry_point": str,
     "source_row_index": int}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import iter_rows

REPO = "openai/openai_humaneval"
CONFIG = "openai_humaneval"
SPLIT = "test"
OUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "humaneval.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT_PATH.exists() and not args.force:
        lines = OUT_PATH.read_text(encoding="utf-8").splitlines()
        print(f"[humaneval] output already exists at {OUT_PATH} ({len(lines)} rows); use --force to refetch")
        _print_samples(lines)
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    for i, row in enumerate(iter_rows(REPO, CONFIG, SPLIT, total=None, page_size=100)):
        rows_out.append({
            "source": "HumanEval",
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
            "source_row_index": i,
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[humaneval] wrote {len(rows_out)} rows to {OUT_PATH}")
    _print_samples([json.dumps(r, ensure_ascii=False) for r in rows_out])

    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"[humaneval] sha256={sha}")


def _print_samples(lines):
    print("[humaneval] sample rows:")
    for line in lines[:2]:
        print(line[:400])


if __name__ == "__main__":
    main()

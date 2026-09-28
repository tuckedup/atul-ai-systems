"""Fetch GSM8K test split (openai/gsm8k, config=main) via HF datasets-server.

Source: https://huggingface.co/datasets/openai/gsm8k (MIT license, public, no gate).
Row shape written to data/raw/gsm8k.jsonl:
    {"source": "GSM8K", "question": str, "solution": str,
     "final_answer": str, "source_row_index": int}

`final_answer` is the numeric string after the "#### " marker in the
official `answer` field, stripped of whitespace and any thousands
commas are left as-is (verbatim substring), matching GSM8K convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import iter_rows

REPO = "openai/gsm8k"
CONFIG = "main"
SPLIT = "test"
N_ROWS = 300
OUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "gsm8k.jsonl"


def extract_final_answer(answer: str) -> str:
    marker = "####"
    idx = answer.rfind(marker)
    if idx == -1:
        return ""
    return answer[idx + len(marker):].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT_PATH.exists() and not args.force:
        lines = OUT_PATH.read_text(encoding="utf-8").splitlines()
        print(f"[gsm8k] output already exists at {OUT_PATH} ({len(lines)} rows); use --force to refetch")
        _print_samples(lines)
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    for i, row in enumerate(iter_rows(REPO, CONFIG, SPLIT, total=N_ROWS, page_size=100)):
        question = row["question"]
        answer = row["answer"]
        rows_out.append({
            "source": "GSM8K",
            "question": question,
            "solution": answer,
            "final_answer": extract_final_answer(answer),
            "source_row_index": i,
        })
        if len(rows_out) >= N_ROWS:
            break

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[gsm8k] wrote {len(rows_out)} rows to {OUT_PATH}")
    _print_samples([json.dumps(r, ensure_ascii=False) for r in rows_out])

    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"[gsm8k] sha256={sha}")


def _print_samples(lines):
    print("[gsm8k] sample rows:")
    for line in lines[:2]:
        print(line[:400])


if __name__ == "__main__":
    main()

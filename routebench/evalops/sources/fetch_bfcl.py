"""Fetch Berkeley Function Calling Leaderboard (BFCL) v3 "simple" subset.

Source: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
(public, not gated, Apache-2.0). This repo is a flat file drop (JSON Lines
files), not a `datasets`-loadable parquet dataset, so we resolve the raw
files directly rather than using datasets-server.

We use the "simple" category: single user turn, single available function,
single expected function call -- the cleanest question/schema/ground-truth
triple for a judge-calibration corpus:
    - BFCL_v3_simple.json            (questions + function schema)
    - possible_answer/BFCL_v3_simple.json  (gold function call)
Both files are JSON Lines (one JSON object per line, NOT a JSON array), keyed
by a shared "id" field (e.g. "simple_0"), which we use to join them.

Row shape written to data/raw/bfcl.jsonl:
    {"source": "BFCL", "bfcl_id": str, "question": str,
     "function": list (the row's raw "function" field, verbatim),
     "ground_truth": dict (the row's raw "ground_truth" field, verbatim),
     "source_row_index": int}

`question` is extracted as the concatenation of all user-role message
contents in the row's `question` field (for "simple" rows this is a single
single-turn conversation, so in practice it's just that one message's text).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import get_bytes

REPO_BASE = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
QUESTIONS_PATH = "BFCL_v3_simple.json"
ANSWERS_PATH = "possible_answer/BFCL_v3_simple.json"
MAX_ROWS = 200

OUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "bfcl.jsonl"
RAW_QUESTIONS_OUT = Path(__file__).parent.parent / "data" / "raw" / "bfcl_raw_questions.jsonl"
RAW_ANSWERS_OUT = Path(__file__).parent.parent / "data" / "raw" / "bfcl_raw_answers.jsonl"


def parse_jsonl_bytes(content: bytes):
    text = content.decode("utf-8")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def extract_question_text(question_field) -> str:
    """question_field is a list of turns, each turn a list of {"role","content"} msgs."""
    parts = []
    for turn in question_field:
        for msg in turn:
            if msg.get("role") == "user":
                parts.append(msg.get("content", ""))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT_PATH.exists() and not args.force:
        lines = OUT_PATH.read_text(encoding="utf-8").splitlines()
        print(f"[bfcl] output already exists at {OUT_PATH} ({len(lines)} rows); use --force to refetch")
        _print_samples(lines)
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        q_bytes = get_bytes(f"{REPO_BASE}/{QUESTIONS_PATH}")
        a_bytes = get_bytes(f"{REPO_BASE}/{ANSWERS_PATH}")
    except Exception as e:  # noqa: BLE001 - a fetch failure is reported and skipped, not fatal
        print(f"[bfcl] FAILED to fetch source files: {e}")
        return

    # Save raw verbatim downloads too, in case the join/shape below is imperfect.
    RAW_QUESTIONS_OUT.write_bytes(q_bytes)
    RAW_ANSWERS_OUT.write_bytes(a_bytes)
    print(f"[bfcl] saved raw questions file to {RAW_QUESTIONS_OUT} ({len(q_bytes)} bytes)")
    print(f"[bfcl] saved raw answers file to {RAW_ANSWERS_OUT} ({len(a_bytes)} bytes)")

    questions = parse_jsonl_bytes(q_bytes)
    answers = parse_jsonl_bytes(a_bytes)
    answers_by_id = {a["id"]: a for a in answers}

    rows_out = []
    for i, q in enumerate(questions):
        if len(rows_out) >= MAX_ROWS:
            break
        bfcl_id = q["id"]
        a = answers_by_id.get(bfcl_id)
        rows_out.append({
            "source": "BFCL",
            "bfcl_id": bfcl_id,
            "question": extract_question_text(q["question"]),
            "function": q["function"],
            "ground_truth": a["ground_truth"] if a else None,
            "source_row_index": i,
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_missing_gt = sum(1 for r in rows_out if r["ground_truth"] is None)
    print(f"[bfcl] wrote {len(rows_out)} rows to {OUT_PATH} ({n_missing_gt} missing ground_truth join)")
    _print_samples([json.dumps(r, ensure_ascii=False) for r in rows_out])

    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"[bfcl] sha256={sha}")


def _print_samples(lines):
    print("[bfcl] sample rows:")
    for line in lines[:2]:
        print(line[:500])


if __name__ == "__main__":
    main()

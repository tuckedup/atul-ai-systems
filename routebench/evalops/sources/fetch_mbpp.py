"""Fetch MBPP (google-research-datasets/mbpp, config=sanitized, split=test).

Source: https://huggingface.co/datasets/google-research-datasets/mbpp
(public, not gated). The `sanitized` config's test split has 257 rows;
we cap output at 250 as instructed (first 250 in dataset order).

Row shape written to data/raw/mbpp.jsonl:
    {"source": "MBPP", "task_id": int, "prompt": str, "code": str,
     "test_list": list[str], "source_row_index": int}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import get_json, iter_rows

REPO = "google-research-datasets/mbpp"
SPLIT = "test"
MAX_ROWS = 250
OUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "mbpp.jsonl"


def pick_config() -> str:
    """Prefer 'sanitized' if it has a test split, else fall back to 'full'."""
    try:
        data = get_json(f"https://datasets-server.huggingface.co/size?dataset={REPO}")
    except Exception:  # noqa: BLE001 - a probe failure falls back to a safe default, not fatal
        return "sanitized"
    configs = {c["config"] for c in data.get("size", {}).get("configs", [])}
    splits = {(c["config"], c["split"]) for c in data.get("size", {}).get("splits", [])}
    if ("sanitized", "test") in splits:
        return "sanitized"
    if ("full", "test") in splits:
        return "full"
    raise RuntimeError(f"No test split found in configs={configs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT_PATH.exists() and not args.force:
        lines = OUT_PATH.read_text(encoding="utf-8").splitlines()
        print(f"[mbpp] output already exists at {OUT_PATH} ({len(lines)} rows); use --force to refetch")
        _print_samples(lines)
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    config = pick_config()
    print(f"[mbpp] using config={config!r}")

    rows_out = []
    for i, row in enumerate(iter_rows(REPO, config, SPLIT, total=MAX_ROWS, page_size=100)):
        rows_out.append({
            "source": "MBPP",
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "code": row["code"],
            "test_list": row["test_list"],
            "source_row_index": i,
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[mbpp] wrote {len(rows_out)} rows to {OUT_PATH}")
    _print_samples([json.dumps(r, ensure_ascii=False) for r in rows_out])

    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"[mbpp] sha256={sha}")


def _print_samples(lines):
    print("[mbpp] sample rows:")
    for line in lines[:2]:
        print(line[:400])


if __name__ == "__main__":
    main()

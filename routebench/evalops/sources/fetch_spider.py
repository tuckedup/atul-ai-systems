"""Fetch Spider text-to-SQL (xlangai/spider, config=spider, split=validation).

Source: https://huggingface.co/datasets/xlangai/spider (public, not gated).
We use `validation` (the dev set with gold SQL) since Spider's real test set
is not publicly released; this matches "prefer validation" in the brief.
1034 rows available; we cap at 250.

Row shape written to data/raw/spider.jsonl:
    {"source": "Spider", "db_id": str, "question": str, "query": str,
     "source_row_index": int}

Table schema info: xlangai/spider's own repo only contains the two parquet
files (train/validation) -- no tables.json is hosted there (confirmed via
the HF tree API: https://huggingface.co/api/datasets/xlangai/spider/tree/main
lists only .gitattributes, README.md, and spider/*.parquet). As instructed,
we fell back to `richardr1126/spider-schema`, a public (non-gated) HF dataset
containing a flattened schema dump (`spider_schema_rows_v2.json`) for the
166 Spider databases: each row has db_id, a "Schema (values (type))" string,
"Primary Keys", and "Foreign Keys". This is NOT the original CSV-derived
Spider `tables.json` structure (column-list-of-lists, dtype tables, etc.) --
it is a real, verbatim, third-party JSON export of the same schema
information in a denormalized/flattened form. Saved verbatim to
data/raw/spider_tables.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _http import get_bytes, iter_rows

REPO = "xlangai/spider"
CONFIG = "spider"
SPLIT = "validation"
MAX_ROWS = 250
OUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "spider.jsonl"

TABLES_REPO_URL = ("https://huggingface.co/datasets/richardr1126/spider-schema/"
                    "resolve/main/spider_schema_rows_v2.json")
TABLES_OUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "spider_tables.json"


def fetch_rows(force: bool):
    if OUT_PATH.exists() and not force:
        lines = OUT_PATH.read_text(encoding="utf-8").splitlines()
        print(f"[spider] output already exists at {OUT_PATH} ({len(lines)} rows); use --force to refetch")
        _print_samples(lines)
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    for i, row in enumerate(iter_rows(REPO, CONFIG, SPLIT, total=MAX_ROWS, page_size=100)):
        rows_out.append({
            "source": "Spider",
            "db_id": row["db_id"],
            "question": row["question"],
            "query": row["query"],
            "source_row_index": i,
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[spider] wrote {len(rows_out)} rows to {OUT_PATH}")
    _print_samples([json.dumps(r, ensure_ascii=False) for r in rows_out])

    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"[spider] sha256={sha}")


def fetch_tables(force: bool):
    if TABLES_OUT_PATH.exists() and not force:
        print(f"[spider] tables file already exists at {TABLES_OUT_PATH}; use --force to refetch")
        return
    try:
        content = get_bytes(TABLES_REPO_URL)
    except Exception as e:  # noqa: BLE001 - a fetch failure is reported and skipped, not fatal
        print(f"[spider] FAILED to fetch table schema info: {e}")
        return
    TABLES_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES_OUT_PATH.write_bytes(content)
    try:
        count = str(len(json.loads(content)))
    except (json.JSONDecodeError, TypeError):
        count = "unknown (not a JSON array)"
    print(f"[spider] wrote table schema info ({count} db schemas) to {TABLES_OUT_PATH}")


def _print_samples(lines):
    print("[spider] sample rows:")
    for line in lines[:2]:
        print(line[:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    fetch_rows(args.force)
    fetch_tables(args.force)


if __name__ == "__main__":
    main()

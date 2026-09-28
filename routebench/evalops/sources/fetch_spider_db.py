"""Fetch Spider SQLite database files needed to execute the gold queries in
``routebench/evalops/data/raw/spider.jsonl``.

The upstream Spider release (xlangai/spider on the Hugging Face Hub) only
ships the question/SQL parquet files -- it does NOT include the SQLite
database files, which are required for a sound text-to-SQL oracle
(result-set equivalence: execute gold SQL and candidate SQL, compare rows).

This script fills that gap by pulling the per-database ``<db_id>.sqlite``
files from a Hub mirror that re-hosts the original Spider database bundle
with the directory layout ``database/<db_id>/<db_id>.sqlite``:

    https://huggingface.co/datasets/prem-research/spider

Only the db_ids actually referenced by spider.jsonl are downloaded (by
default: car_1, concert_singer, flight_2, pets_1), each capped at 400 MB
and the whole run capped at ~500 MB, per the size limits requested in the
task. Every file is written to
``routebench/evalops/data/raw/spider_db/<db_id>.sqlite`` and then verified by
opening it read-only and executing every gold query in spider.jsonl that
targets it.

Usage:
    .venv/Scripts/python.exe routebench/evalops/sources/fetch_spider_db.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3
import sys
import urllib.request

REPO = "prem-research/spider"
RESOLVE_TMPL = "https://huggingface.co/datasets/{repo}/resolve/main/database/{db_id}/{db_id}.sqlite"

RAW_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
SPIDER_JSONL = RAW_DIR / "spider.jsonl"
OUT_DIR = RAW_DIR / "spider_db"

MAX_SINGLE_FILE_BYTES = 400 * 1024 * 1024
MAX_TOTAL_BYTES = 500 * 1024 * 1024


def load_spider_rows() -> list[dict]:
    rows = []
    with open(SPIDER_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def download(db_id: str, dest: pathlib.Path) -> int:
    url = RESOLVE_TMPL.format(repo=REPO, db_id=db_id)
    req = urllib.request.Request(url, headers={"User-Agent": "routebench-spider-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        length = int(resp.headers.get("Content-Length", 0))
        if length and length > MAX_SINGLE_FILE_BYTES:
            raise RuntimeError(f"{db_id}: remote file too large ({length} bytes), skipping")
        data = resp.read(MAX_SINGLE_FILE_BYTES + 1)
        if len(data) > MAX_SINGLE_FILE_BYTES:
            raise RuntimeError(f"{db_id}: downloaded data exceeds cap, aborting")
    dest.write_bytes(data)
    return len(data)


def verify(db_id: str, dest: pathlib.Path, rows: list[dict]) -> tuple[int, int]:
    """Return (rows_targeting_this_db, rows_that_execute_cleanly)."""
    targeted = [r for r in rows if r["db_id"] == db_id]
    ok = 0
    uri = f"file:{dest.as_posix()}?mode=ro"
    for row in targeted:
        try:
            conn = sqlite3.connect(uri, uri=True)
            try:
                conn.execute(row["query"]).fetchall()
                ok += 1
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  [verify] {db_id} row {row.get('source_row_index')} FAILED: {exc}")
    return len(targeted), ok


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_spider_rows()
    db_ids = sorted({r["db_id"] for r in rows})
    print(f"spider.jsonl has {len(rows)} rows across {len(db_ids)} db_ids: {db_ids}")

    total_bytes = 0
    total_rows_ok = 0
    total_rows_present = 0

    for db_id in db_ids:
        dest = OUT_DIR / f"{db_id}.sqlite"
        if dest.exists():
            print(f"[skip] {db_id}: already present at {dest}")
        else:
            try:
                if total_bytes > MAX_TOTAL_BYTES:
                    print(f"[abort] total download cap reached, skipping {db_id}")
                    continue
                n = download(db_id, dest)
                total_bytes += n
                sha = hashlib.sha256(dest.read_bytes()).hexdigest()
                print(f"[ok] {db_id}: {n} bytes, sha256={sha}")
            except Exception as exc:  # noqa: BLE001
                print(f"[fail] {db_id}: {exc}")
                continue

        n_targeted, n_ok = verify(db_id, dest, rows)
        total_rows_present += n_targeted
        total_rows_ok += n_ok
        print(f"[verify] {db_id}: {n_ok}/{n_targeted} gold queries execute cleanly")

    print(f"\nTOTAL: {total_rows_present}/{len(rows)} rows have a DB present, "
          f"{total_rows_ok}/{len(rows)} rows execute cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())

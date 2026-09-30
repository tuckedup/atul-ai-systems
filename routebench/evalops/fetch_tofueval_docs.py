"""Retrieve ONLY the 100 TofuEval source transcripts (MediaSum / MeetingBank) needed to identify documents.

Downloads the two public, non-gated test-split files to a scratch directory, extracts the documents named in
`raw/tofueval_document_ids_dev_test_split.json`, writes `raw/tofueval_upstream/docs.jsonl`, then deletes the
scratch files. Sources (both CC-BY-NC-SA-4.0 mirrors; MediaSum authors additionally ask for research-only use):
  MeetingBank  huggingface.co/datasets/lytang/MeetingBank-transcript  test.csv   (TofuEval's own preprocessing)
  MediaSum     huggingface.co/datasets/nbroad/mediasum                  test.json
Output is for local, non-commercial evaluation only and is git-ignored; do not redistribute it.
Never judges, trains or changes a split.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
import tempfile
import urllib.request
from pathlib import Path

R = Path(__file__).parent / "data" / "raw"
OUTD = R / "tofueval_upstream"
URLS = {"mediasum": "https://huggingface.co/datasets/nbroad/mediasum/resolve/main/test.json",
        "meetingbank": "https://huggingface.co/datasets/lytang/MeetingBank-transcript/resolve/main/test.csv"}


def h(t: str) -> str:
    return hashlib.sha256(t.strip().encode()).hexdigest()


def nh(t: str) -> str:
    return hashlib.sha256(re.sub(r"[^A-Za-z0-9]+", "", t).lower().encode()).hexdigest()


def main() -> None:
    csv.field_size_limit(10**9)
    split = json.loads((R / "tofueval_document_ids_dev_test_split.json").read_text(encoding="utf8"))
    OUTD.mkdir(exist_ok=True)
    want_ms = set(split["dev"]["mediasum"]) | set(split["test"]["mediasum"])
    want_mb = set(split["dev"]["meetingbank"]) | set(split["test"]["meetingbank"])
    docs = []
    with tempfile.TemporaryDirectory() as tmp:
        ms_path, mb_path = Path(tmp) / "ms.json", Path(tmp) / "mb.csv"
        urllib.request.urlretrieve(URLS["mediasum"], ms_path)
        urllib.request.urlretrieve(URLS["meetingbank"], mb_path)
        for line in ms_path.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in want_ms:
                part = "dev" if r["id"] in split["dev"]["mediasum"] else "test"
                docs.append({"upstream_part": part, "source": "MediaSum", "doc_id": r["id"],
                             "text": "".join(f"{s}: {u}\n" for s, u in zip(r["speaker"], r["utt"]))})
        with mb_path.open(encoding="utf8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["meeting_id"] in want_mb:
                    part = "dev" if r["meeting_id"] in split["dev"]["meetingbank"] else "test"
                    docs.append({"upstream_part": part, "source": "MeetingBank", "doc_id": r["meeting_id"],
                                 "text": r["source"]})
    for d in docs:
        d.update(sha256=h(d["text"]), norm_sha256=nh(d["text"]), chars=len(d["text"]))
    counts = collections.Counter((d["upstream_part"], d["source"]) for d in docs)
    assert all(v in (15, 35) for v in counts.values()) and len(docs) == 100, counts
    (OUTD / "docs.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf8")
    print(dict(counts))


if __name__ == "__main__":
    main()

"""Offline, prediction/label-blind integrity copy of headline TRAIN/DEV only.

Never edits the original corpus, labels, splits or caches. Invalid encodings are
quarantined, not deleted. Curly-quote damage may be restored ONLY against a unique
local upstream-test transcript, with every other character identical. No inference,
downloads, label adjudication, truncation repairs or acceptance-test evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

VERSION = "source-integrity-v1"
SUBSETS = {"TofuEval-MediaS", "TofuEval-MeetB", "AggreFact-CNN"}
FIELDS = ("task_input", "context", "reference", "candidate_output")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    return digest(text.encode("utf8"))


def restore(text: str, transcripts: list[dict]) -> tuple[str, dict | None]:
    """Require unique full-text alignment, not topical/alphanumeric similarity.

    Literal opening curly quotes may correspond to straight quotes. Replacement
    characters may recover one punctuation character only, never words/numbers.
    End-of-file newlines are ignored for matching but preserved from the input.
    """
    if "\ufffd" not in text and "\u201c" not in text:
        return text, None
    body = text.rstrip("\r\n")
    tail = text[len(body):]
    pattern = "".join(
        '(?:"|\u201c)' if c == "\u201c" else "([^\\w\\s])" if c == "\ufffd" else re.escape(c)
        for c in body
    )
    matches = []
    for transcript in transcripts:
        target = transcript["text"].rstrip("\r\n")
        match = re.fullmatch(pattern, target)
        if match and all(unicodedata.category(c).startswith("P") for c in match.groups()):
            matches.append(transcript)
    if len(matches) != 1:
        return text, None
    selected = matches[0]
    fixed = selected["text"].rstrip("\r\n") + tail
    if fixed == text:
        return text, None
    return fixed, {"doc_id": selected["doc_id"], "source": selected["source"],
                   "upstream_text_sha256": text_hash(selected["text"])}


def build(cases_path: Path, splits_path: Path, upstream_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    # Snapshot inputs before deriving outputs; concurrent changes abort publication.
    snapshots = {p: p.read_bytes() for p in (cases_path, splits_path, upstream_path)}
    assignment = json.loads(snapshots[splits_path])["assignment"]
    transcripts: dict[str, list[dict]] = {"TofuEval-MediaS": [], "TofuEval-MeetB": []}
    upstream_ids = set()
    for line in snapshots[upstream_path].splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # Do not inspect reserved upstream-dev transcript content.
        if row["upstream_part"] != "test":
            continue
        key = (row["source"], row["doc_id"])
        if key in upstream_ids or not isinstance(row["text"], str):
            raise ValueError("Duplicate or malformed upstream transcript")
        upstream_ids.add(key)
        subset = {"MediaSum": "TofuEval-MediaS", "MeetingBank": "TofuEval-MeetB"}.get(row["source"])
        if subset:
            transcripts[subset].append(row)
    kept, quarantine, repairs, mapping = [], [], [], []
    original_ids, counts = set(), Counter()
    for line in snapshots[cases_path].splitlines():
        if not line.strip():
            continue
        case = json.loads(line)  # malformed records abort; never silently shrink scope
        split = assignment[case["group_id"]]
        if split not in ("train", "dev"):
            continue  # original held-out text is not audited, repaired or exported
        if case.get("meta", {}).get("subset") not in SUBSETS:
            continue
        cid = case["case_id"]
        if not isinstance(cid, str) or not cid or cid in original_ids:
            raise ValueError("Missing or duplicate case identity")
        original_ids.add(cid)
        if any(not isinstance(case.get(f), str) for f in FIELDS):
            raise ValueError(f"Malformed text fields: {cid}")
        counts[f"input_{split}"] += 1
        original = dict(case)
        fixed, provenance = restore(case["context"], transcripts.get(case["meta"]["subset"], []))
        case["context"] = fixed
        reasons = [{"field": f, "reason": reason} for f in FIELDS
                   for marker, reason in (("\ufffd", "unrecoverable_replacement_character"),
                                          ("\x00", "nul_character")) if marker in case[f]]
        if reasons:
            quarantine.append({"case": original, "split": split, "reasons": reasons})
            counts[f"quarantined_{split}"] += 1
            continue
        if provenance:
            repairs.append({"original_case_id": cid, "group_id": case["group_id"], "split": split,
                            "field": "context", "before_sha256": text_hash(original["context"]),
                            "after_sha256": text_hash(fixed), "provenance": provenance})
        # New IDs prevent accidental joins to old cached judgments. Group identity
        # and labels are NOT regenerated; explicit original-ID mapping is required.
        case["case_id"] = f"{VERSION}:{cid}"
        kept.append(case)
        mapping.append({"case_id": case["case_id"], "original_case_id": cid,
                        "group_id": case["group_id"], "split": split})
        counts[f"kept_{split}"] += 1
    for path, before in snapshots.items():
        if path.read_bytes() != before:
            raise RuntimeError(f"Input changed during audit: {path}")
    report = {"version": VERSION, "scope": "headline train/dev only; NOT acceptance",
              "labels_read": False, "predictions_read": False, "model_calls": 0,
              "counts": dict(counts), "repaired_cases": len(repairs),
              "repaired_documents": len({r["group_id"] for r in repairs}),
              "inputs": {str(p.resolve()): digest(raw) for p, raw in snapshots.items()},
              "code_sha256": digest(Path(__file__).read_bytes()), "repairs": repairs,
              "case_mapping": mapping, "status": "OFFLINE_COPY_ONLY_NO_KAPPA"}
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("cases.jsonl", kept), ("quarantine.jsonl", quarantine)):
        payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        (output / filename).write_text(payload, encoding="utf8")
    report["outputs"] = {name: digest((output / name).read_bytes())
                         for name in ("cases.jsonl", "quarantine.jsonl")}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.data / "cases.jsonl", args.data / "splits.json",
                   args.data / "raw/tofueval_upstream/docs.jsonl", args.output)
    print(json.dumps({k: report[k] for k in ("status", "counts", "repaired_cases", "repaired_documents")}))


if __name__ == "__main__":
    main()

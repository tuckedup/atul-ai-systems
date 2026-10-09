"""Versioned full-source restoration after the label-blind punctuation audit.

Keeps source_integrity v1 and its output unchanged. Uses no labels/predictions.
Only a unique raw row with matching subset, candidate and EXACT source prefix
can restore a builder-truncated transcript. Never removes cases for truncation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import source_integrity as v1

VERSION = "source-integrity-v2"


def restore_full(case: dict, row: dict) -> str:
    for field in ("doc", "claim", "subset"):
        if not isinstance(row.get(field), str):
            raise TypeError("Malformed raw source row")
    if row["subset"] != case["meta"]["subset"] or row["claim"].strip() != case["candidate_output"]:
        raise ValueError("Raw row does not bind to candidate/subset")
    full = row["doc"].strip()
    if not full.startswith(case["context"]):
        raise ValueError("Raw document is not an exact source prefix match")
    # Cannot import damaged text from a second, already-corrupt copy.
    if "\ufffd" in full or "\x00" in full:
        raise ValueError("Raw document contains encoding damage")
    return full


def build(data: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    raw_path = data / "raw/aggrefact.jsonl"
    raw_bytes = raw_path.read_bytes()
    # v1's unchanged command preserves an auditable intermediate copy.
    output.mkdir(parents=True, exist_ok=False)
    report = v1.build(data / "cases.jsonl", data / "splits.json",
                      data / "raw/tofueval_upstream/docs.jsonl", output / "punctuation-stage")
    cases = [json.loads(line) for line in (output / "punctuation-stage/cases.jsonl").read_bytes().splitlines()]
    targets = {case["meta"]["source_row_index"] for case in cases if len(case["context"]) == 6000}
    rows = {}
    for line in raw_bytes.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index = row.get("source_row_index")
        if index in targets:
            if index in rows:
                raise ValueError("Ambiguous raw source row index")
            # Discard label fields without accessing their values.
            rows[index] = {field: row.get(field) for field in ("doc", "claim", "subset")}
    full_repairs = []
    for case in cases:
        old_id = case["case_id"].removeprefix(v1.VERSION + ":")
        if len(case["context"]) == 6000:
            index = case["meta"]["source_row_index"]
            if index not in rows:
                raise ValueError("Missing source row for possible truncation")
            full = restore_full(case, rows[index])
            if len(full) > len(case["context"]):
                full_repairs.append({"original_case_id": old_id, "group_id": case["group_id"],
                                     "source_row_index": index, "before_chars": len(case["context"]),
                                     "after_chars": len(full), "before_sha256": v1.text_hash(case["context"]),
                                     "after_sha256": v1.text_hash(full), "reason": "exact_bound_source_prefix_restored"})
                case["context"] = full
        case["case_id"] = f"{VERSION}:{old_id}"
    # Refuse to publish a terminal manifest if any source input changed.
    if raw_path.read_bytes() != raw_bytes:
        raise RuntimeError("Raw source changed during audit")
    for filename in ("cases.jsonl", "splits.json", "raw/tofueval_upstream/docs.jsonl"):
        path = (data / filename).resolve()
        if v1.digest(path.read_bytes()) != report["inputs"][str(path)]:
            raise RuntimeError("Corpus input changed during audit")
    report["version"] = VERSION
    report["inputs"][str(raw_path.resolve())] = v1.digest(raw_bytes)
    report["punctuation_code_sha256"] = report.pop("code_sha256")
    report["code_sha256"] = v1.digest(Path(__file__).read_bytes())
    report["full_source_repairs"] = full_repairs
    report["case_mapping"] = [{**item, "case_id": f"{VERSION}:{item['original_case_id']}"}
                              for item in report["case_mapping"]]
    (output / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n"
                                               for c in cases), encoding="utf8")
    (output / "quarantine.jsonl").write_bytes((output / "punctuation-stage/quarantine.jsonl").read_bytes())
    report["outputs"] = {name: v1.digest((output / name).read_bytes())
                         for name in ("cases.jsonl", "quarantine.jsonl")}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.data, args.output)
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "punctuation_repaired_cases": report["repaired_cases"],
                      "full_source_repaired_cases": len(report["full_source_repairs"])}))


if __name__ == "__main__":
    main()

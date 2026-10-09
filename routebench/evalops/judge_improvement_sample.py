"""Deterministic, label-independent-by-document RAGTruth fitting sample for the judge-improvement screening (offline; reads a frozen manifest, calls nothing).

Rule `phase1-v1` (docs/ROUTEBENCH_JUDGE_IMPROVEMENT_PROPOSAL_V2_20261005.md): from the RAGTruth TRAIN partition of the frozen training manifest (CNN/DM only, "Recent News" and fingerprint-flagged
documents already excluded by the manifest), drop sentences whose label span crosses a sentence boundary, order DOCUMENT GROUPS by sha256(`phase1-v1|doc|<group_id>`) and keep the first `documents`;
keep EVERY unsupported sentence of those groups and the first `supported` supported sentences ordered by sha256(`phase1-v1|sent|<example_id>`). Groups are never split. Class-conditional error
rates are unaffected by sampling supported sentences at a different rate than unsupported ones, which is why the decision rule is defined through class-conditional rates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

RULE = "phase1-v1"
DOCUMENTS, SUPPORTED = 160, 712


def _h(tag: str, key: str) -> str:
    return hashlib.sha256(f"{RULE}|{tag}|{key}".encode()).hexdigest()


def draw(manifest: dict, documents: int = DOCUMENTS, supported: int = SUPPORTED) -> dict:
    cols = manifest["columns"]
    rows = [dict(zip(cols, r, strict=True)) for r in manifest["rows"]]
    train = [r for r in rows if r["partition"] == "train" and not r["crosses_boundary"]]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in train:
        groups[r["group_id"]].append(r)
    chosen = sorted(groups, key=lambda g: _h("doc", g))[:documents]
    pool = [r for g in chosen for r in groups[g]]
    unsupported = [r for r in pool if r["label"] == 0]
    sup_pool = sorted((r for r in pool if r["label"] == 1), key=lambda r: _h("sent", r["example_id"]))
    picked = unsupported + sup_pool[:supported]
    picked.sort(key=lambda r: (r["group_id"], r["example_id"]))
    out = {"rule": RULE, "documents_requested": documents, "supported_requested": supported, "manifest_sha256": manifest["manifest_sha256"],
           "counts": {"documents": len(chosen), "sentences": len(picked), "unsupported": len(unsupported), "supported": min(supported, len(sup_pool)), "supported_in_pool": len(sup_pool),
                      "dropped_crosses_boundary": sum(1 for r in rows if r["partition"] == "train" and r["crosses_boundary"])},
           "rows": [[r["example_id"], r["response_id"], r["source_id"], r["group_id"], r["label"], r["start"], r["end"]] for r in picked]}
    out["sample_sha256"] = hashlib.sha256(json.dumps(out["rows"], separators=(",", ":")).encode()).hexdigest()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        print("REFUSED: output exists", file=sys.stderr)
        return 2
    sample = draw(json.loads(a.manifest.read_text(encoding="utf8")))
    a.out.write_text(json.dumps(sample, indent=1), encoding="utf8")
    print(json.dumps({**sample["counts"], "sample_sha256": sample["sample_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

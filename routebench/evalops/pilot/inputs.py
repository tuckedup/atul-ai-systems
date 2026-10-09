"""Validated text inputs for the pilot: re-derive every needed example from the local RAGTruth files and check it against the manifest.

The manifest carries identifiers, offsets and labels but no text. The texts come from the local downloaded files, re-run through the
adapter, and every requested example must reproduce the manifest row exactly (partition, offsets, label, document fingerprint prefix).
The files themselves must match the hashes the manifest recorded. Any difference is refused; nothing is guessed or repaired.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .harness import ManifestMismatch
from .ingest_audit import FILES, read_jsonl
from .ragtruth import adapt


def load_texts(directory: Path, manifest: Mapping[str, Any], wanted_ids: Iterable[str]) -> dict[str, tuple[str, str]]:
    meta = manifest["meta"]
    for name in FILES:
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if digest != meta["file_sha256"][name]:
            raise ManifestMismatch(f"{name} does not match the hash recorded in the manifest")
    responses, _ = read_jsonl(directory / FILES[0])
    sources, _ = read_jsonl(directory / FILES[1])
    result = adapt(responses, sources, allowed_sources=frozenset(meta["allowed_sources"]),
                   offset_convention=meta["offset_convention"])
    derived = {x.example_id: x for x in result.examples}
    cols = manifest["columns"]
    idx = {k: cols.index(k) for k in ("example_id", "partition", "document_fp16", "label", "start", "end")}
    rows = {r[idx["example_id"]]: r for r in manifest["rows"]}
    out: dict[str, tuple[str, str]] = {}
    for eid in wanted_ids:
        x, row = derived.get(eid), rows.get(eid)
        if x is None or row is None:
            raise ManifestMismatch(f"example {eid} cannot be re-derived from the local files")
        if (x.partition, x.document_fingerprint[:16], x.label, x.start, x.end) != (
                row[idx["partition"]], row[idx["document_fp16"]], row[idx["label"]], row[idx["start"]], row[idx["end"]]):
            raise ManifestMismatch(f"example {eid} differs from its manifest row")
        out[eid] = (x.document, x.sentence)
    return out

"""Exact and near-duplicate document audit interface (fingerprints only; no reference text is stored).

A `ReferenceIndex` keeps, for each reference document, only a normalised-text SHA-256 and a set of hashed word 5-shingles. So a
sensitive reference set (for example the untouched reserve documents) can be indexed and checked without exposing its text,
labels, predictions or errors. Declared thresholds: a candidate is a near-duplicate of a reference when the Jaccard similarity of
their shingle sets is >= JACCARD_MIN, or when either document is >= CONTAINMENT_MIN contained in the other. Documents shorter
than SHINGLE_K tokens can only be matched exactly. This detects lexical overlap; it cannot prove the absence of semantic
overlap or of a pretrained model having seen a document.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

SHINGLE_K = 5
JACCARD_MIN = 0.5
CONTAINMENT_MIN = 0.8
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def exact_fingerprint(text: str) -> str:
    return hashlib.sha256("".join(_tokens(text)).encode()).hexdigest()


def shingle_hashes(text: str, k: int = SHINGLE_K) -> frozenset[int]:
    toks = _tokens(text)
    if len(toks) < k:
        return frozenset()
    return frozenset(int(hashlib.sha256(" ".join(toks[i:i + k]).encode()).hexdigest()[:12], 16)
                     for i in range(len(toks) - k + 1))


@dataclass(frozen=True)
class AuditMatch:
    candidate_id: str
    reference_name: str
    kind: str  # "exact" | "near_duplicate"
    jaccard: float
    containment_of_candidate: float
    containment_of_reference: float


@dataclass
class AuditReport:
    matches: list[AuditMatch] = field(default_factory=list)
    clean_ids: list[str] = field(default_factory=list)
    checked: int = 0

    @property
    def flagged_ids(self) -> set[str]:
        return {m.candidate_id for m in self.matches}


class ReferenceIndex:
    def __init__(self) -> None:
        self._names: list[str] = []
        self._fingerprints: dict[str, list[int]] = defaultdict(list)
        self._shingles: list[frozenset[int]] = []
        self._inverted: dict[int, list[int]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self._names)

    def add(self, name: str, text: str) -> None:
        idx = len(self._names)
        self._names.append(name)
        self._fingerprints[exact_fingerprint(text)].append(idx)
        sh = shingle_hashes(text)
        self._shingles.append(sh)
        for h in sh:
            self._inverted[h].append(idx)

    def matches(self, candidate_id: str, text: str, *, exclude_name: str | None = None) -> list[AuditMatch]:
        """Every reference that is an exact or near-duplicate of `text` (strongest first), optionally ignoring one name."""
        out: list[AuditMatch] = []
        fp = exact_fingerprint(text)
        exact_idx = set(self._fingerprints.get(fp, ()))
        for i in sorted(exact_idx):
            if self._names[i] != exclude_name:
                out.append(AuditMatch(candidate_id, self._names[i], "exact", 1.0, 1.0, 1.0))
        cand = shingle_hashes(text)
        if cand:
            shared: dict[int, int] = defaultdict(int)
            for h in cand:
                for i in self._inverted.get(h, ()):
                    shared[i] += 1
            for i, n in shared.items():
                if i in exact_idx or self._names[i] == exclude_name:
                    continue
                ref = self._shingles[i]
                union = len(cand) + len(ref) - n
                jac, c_cand, c_ref = n / union, n / len(cand), n / len(ref)
                if jac >= JACCARD_MIN or c_cand >= CONTAINMENT_MIN or c_ref >= CONTAINMENT_MIN:
                    out.append(AuditMatch(candidate_id, self._names[i], "near_duplicate", jac, c_cand, c_ref))
        return sorted(out, key=lambda m: (m.kind != "exact", -m.jaccard, m.reference_name))

    def check(self, candidate_id: str, text: str) -> AuditMatch | None:
        """Return the strongest match for `candidate_id`, or None. Exact matches win."""
        found = self.matches(candidate_id, text)
        return found[0] if found else None

    def save(self, path: Path) -> None:
        payload = {"k": SHINGLE_K, "names": self._names,
                   "fingerprints": {fp: idx for fp, idx in self._fingerprints.items()},
                   "shingles": [sorted(s) for s in self._shingles]}
        path.write_text(json.dumps(payload), encoding="utf8")

    @classmethod
    def load(cls, path: Path) -> ReferenceIndex:
        d = json.loads(path.read_text(encoding="utf8"))
        if d["k"] != SHINGLE_K:
            raise ValueError("index was built with a different shingle size")
        ix = cls()
        ix._names = list(d["names"])
        for fp, idx in d["fingerprints"].items():
            ix._fingerprints[fp] = list(idx)
        for i, sh in enumerate(d["shingles"]):
            s = frozenset(sh)
            ix._shingles.append(s)
            for h in s:
                ix._inverted[h].append(i)
        return ix


def audit(candidates: Mapping[str, str], index: ReferenceIndex) -> AuditReport:
    """Check every candidate document; nothing is dropped here, the caller decides what to exclude."""
    report = AuditReport(checked=len(candidates))
    for cid, text in candidates.items():
        m = index.check(cid, text)
        if m is None:
            report.clean_ids.append(cid)
        else:
            report.matches.append(m)
    return report

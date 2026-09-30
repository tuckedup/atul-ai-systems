"""Frozen, label-independent sampling manifest for the TofuEval upstream-DEV validation set.

Design (declared before any judging; seeds fixed):
  Stage 1  simple random sample of 20 of the 35 upstream-dev documents per source (seed 20261001).
  Stage 2  within each sampled document, a simple random sample of 8 summary sentences (seed 20261002).
           NOTHING in stages 1-2 reads a label. Inclusion probability = (20/35) * (8 / N_doc).
  Diagnostic stratum (reported SEPARATELY, never pooled into the primary estimate): from the same sampled
           documents, up to 6 further sentences whose published label is "no", drawn uniformly from the
           document's remaining negatives (seed 20261003). Inclusion probability within the document =
           min(6, M_neg_remaining) / M_neg_remaining, declared here so weighted estimates are possible.
The 15 remaining documents per source are held back untouched as a RESERVE validation set.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
A = DATA / "raw" / "authoritative"
N_DOCS, N_SENT, N_NEG = 20, 8, 6
SEEDS = {"docs": 20261001, "sent": 20261002, "neg": 20261003}
FILES = {"MediaSum": "tofueval_mediasum_factual_eval_dev.csv",
         "MeetingBank": "tofueval_meetingbank_factual_eval_dev.csv"}


def build() -> dict:
    rng_d = np.random.default_rng(SEEDS["docs"])
    rng_s = np.random.default_rng(SEEDS["sent"])
    rng_n = np.random.default_rng(SEEDS["neg"])
    out: dict = {"design": __doc__, "seeds": SEEDS, "sources": {}}
    for src, fn in FILES.items():
        with (A / fn).open(encoding="utf8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for i, r in enumerate(rows):
            r["row_index"] = i
        docs = sorted({r["doc_id"] for r in rows})
        chosen = sorted(rng_d.choice(docs, N_DOCS, replace=False).tolist())
        reserve = [d for d in docs if d not in chosen]
        primary, diagnostic = [], []
        for d in chosen:
            drows = [r for r in rows if r["doc_id"] == d]
            idx = sorted(rng_s.choice(len(drows), min(N_SENT, len(drows)), replace=False).tolist())
            picked = [drows[i] for i in idx]
            pi_doc = N_DOCS / len(docs)
            for r in picked:
                primary.append({"doc_id": d, "row_index": r["row_index"], "annotation_id": r["annotation_id"],
                                "model_name": r["model_name"], "sent_idx": r["sent_idx"],
                                "inclusion_prob": pi_doc * len(picked) / len(drows)})
            taken = {r["row_index"] for r in picked}
            neg = [r for r in drows if r["sent_label"].strip().lower() == "no" and r["row_index"] not in taken]
            if neg:
                k = min(N_NEG, len(neg))
                for i in sorted(rng_n.choice(len(neg), k, replace=False).tolist()):
                    r = neg[i]
                    diagnostic.append({"doc_id": d, "row_index": r["row_index"], "annotation_id": r["annotation_id"],
                                       "model_name": r["model_name"], "sent_idx": r["sent_idx"],
                                       "within_doc_inclusion_prob": k / len(neg)})
        out["sources"][src] = {"n_docs_upstream_dev": len(docs), "sampled_docs": chosen, "reserve_docs": reserve,
                               "primary": primary, "diagnostic_negatives": diagnostic}
    out["manifest_sha256"] = hashlib.sha256(json.dumps(out["sources"], sort_keys=True).encode()).hexdigest()
    return out


if __name__ == "__main__":
    m = build()
    (DATA / "tofueval_upstream_dev_sample_manifest.json").write_text(json.dumps(m, indent=1), encoding="utf8")
    for s, v in m["sources"].items():
        print(s, "docs", len(v["sampled_docs"]), "primary", len(v["primary"]), "diagnostic negatives", len(v["diagnostic_negatives"]))
    print("manifest_sha256", m["manifest_sha256"])

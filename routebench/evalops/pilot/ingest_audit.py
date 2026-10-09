"""Bounded RAGTruth ingestion audit (data availability only: no training, no model score, no benchmark evaluation).

    python -m evalops.pilot.ingest_audit download      # ONLY after explicit approval; fetches the two pinned files
    python -m evalops.pilot.ingest_audit audit DIR     # offline audit of DIR/response.jsonl and DIR/source_info.jsonl

`audit` reports, in `DIR/ingest_report.json`: file hashes, the offset-convention probe result, eligible documents / responses /
sentences per partition, class balance, every rejection reason with counts, exact and near-duplicate overlap against all protected
evaluation documents (by fingerprint index; no labels, predictions or errors are read), and a fixed-seed splitter audit sample.
Recent News (and any source outside `allowed_sources`) is downloaded but never eligible. The audit cannot resolve third-party
rights and cannot prove the absence of semantic overlap or of pretrained-model contamination.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .audit import ReferenceIndex, audit
from .ragtruth import (
    DEFAULT_ALLOWED_SOURCES,
    LABEL_POLICY_VERSION,
    SENTENCE_SPLITTER_VERSION,
    AdapterResult,
    SentenceExample,
    adapt,
    probe_offset_convention,
)

COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
RAW = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{COMMIT}/dataset"
FILES = ("response.jsonl", "source_info.jsonl")
SAMPLE_SEED, SAMPLE_RESPONSES = 20261101, 40
MIN_CONVENTION_SHARE = 0.99
PARTITIONS = ("train", "validation", "test")
AUDIT_VERSION = "v5"
MANIFEST_VERSION = "ragtruth-training-manifest-v5"
INTERNAL_RULE = ("exclude every TRAIN or VALIDATION document that is an exact or near-duplicate of a document in a later "
                 "partition (train vs validation/test, validation vs test); the RAGTruth test partition is never excluded by "
                 "this rule; decided by the fingerprint index alone, never by model errors")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        urllib.request.urlretrieve(f"{RAW}/{name}", target / name)


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows, bad = [], 0
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            bad += 1
    return rows, bad


def decide_offset_convention(probe: Mapping[str, int]) -> str:
    """Fix the convention once from the probe; refuse to guess if the evidence is not decisive."""
    total = sum(probe.values())
    if total == 0:
        raise SystemExit("no labelled spans to probe; cannot fix the offset convention")
    for name, extra in (("exclusive", probe["exclusive"] + probe["both"]), ("inclusive", probe["inclusive"] + probe["both"])):
        if extra / total >= MIN_CONVENTION_SHARE and (probe["inclusive"] if name == "exclusive" else probe["exclusive"]) == 0:
            return name
    raise SystemExit(f"offset convention is not decisive: {dict(probe)}; decide manually before converting")


def protected_documents(root: Path) -> dict[str, str]:
    """Every local evaluation, probe, attribution, exemplar and upstream document, keyed by a descriptive name."""
    docs: dict[str, str] = {}
    data = root / "evalops" / "data"
    for fname in ("cases.jsonl", "probe/cases_wice.jsonl", "probe/cases_reveal.jsonl", "attribution/cases.jsonl"):
        p = data / fname
        if p.exists():
            for line in p.read_text(encoding="utf8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("context"):
                        docs[f"{fname}:{rec.get('case_id')}"] = rec["context"]
    up = data / "raw" / "tofueval_upstream" / "docs.jsonl"
    if up.exists():
        for line in up.read_text(encoding="utf8").splitlines():
            if line.strip():
                rec = json.loads(line)
                docs[f"upstream:{rec['upstream_part']}:{rec['doc_id']}"] = rec["text"]
    for fname in ("exemplars.json", "gedd_train_exemplars.json", "gedd_diverse_exemplars.json"):
        p = data / fname
        if p.exists():
            blob = json.loads(p.read_text(encoding="utf8"))
            ex = blob.get("exemplars", blob)
            for task, items in (ex.items() if isinstance(ex, dict) else []):
                for i, e in enumerate(items if isinstance(items, list) else []):
                    if isinstance(e, dict) and e.get("context"):
                        docs[f"{fname}:{task}:{i}"] = e["context"]
    return docs


def _summary(result: AdapterResult) -> dict[str, Any]:
    by_part: dict[str, dict[str, Any]] = {}
    for part in ("train", "validation", "test"):
        ex = [x for x in result.examples if x.partition == part]
        by_part[part] = {
            "documents": len({x.document_fingerprint for x in ex}), "responses": len({x.response_id for x in ex}),
            "sentences": len(ex), "sentences_supported": sum(x.label for x in ex),
            "sentences_unsupported": sum(1 - x.label for x in ex),
            "responses_unsupported": len({x.response_id for x in ex if result.response_labels.get(x.response_id) == 0}),
        }
    return by_part


def splitter_audit(result: AdapterResult, responses: list[dict[str, Any]]) -> dict[str, Any]:
    # Inspection may lead to changing the splitter: keep validation and test text out of it.
    ids = sorted({x.response_id for x in result.examples if x.partition == "train"})
    picked = random.Random(SAMPLE_SEED).sample(ids, min(SAMPLE_RESPONSES, len(ids)))
    text = {str(r["id"]): r["response"] for r in responses if "id" in r and "response" in r}
    by_id: dict[str, list[Any]] = defaultdict(list)
    for x in result.examples:
        by_id[x.response_id].append(x)
    suspicious = Counter()
    sample = []
    for rid in picked:
        sents = sorted(by_id[rid], key=lambda x: x.sentence_index)
        for x in sents:
            if len(x.sentence) < 15:
                suspicious["shorter_than_15_chars"] += 1
            if x.sentence[-1:] not in ".!?\"'”’)":
                suspicious["no_terminal_punctuation"] += 1
            if x.sentence[:1].islower():
                suspicious["starts_lowercase"] += 1
        sample.append({"response_id": rid, "sentences": [{"text": x.sentence, "label": x.label} for x in sents],
                       "n_chars": len(text.get(rid, ""))})
    total = sum(len(s["sentences"]) for s in sample)
    return {"seed": SAMPLE_SEED, "partition": "train", "responses": len(picked), "sentences": total, "suspicious": dict(suspicious),
            "sample": sample, "note": "flags are heuristics for human review; they are not an error rate"}


def internal_overlap(
    examples: list[SentenceExample],
) -> tuple[list[SentenceExample], list[dict[str, Any]], dict[str, str]]:
    """Apply INTERNAL_RULE. Returns (kept examples, full exclusion manifest, document fingerprint -> near-duplicate group id).

    The rule is evaluated on the ORIGINAL partition contents, so the result does not depend on iteration order. Kept documents
    that are near-duplicates of each other inside one partition receive one shared group id, so downstream grouped
    cross-validation cannot split them.
    """
    docs: dict[str, dict[str, str]] = {p: {} for p in PARTITIONS}
    responses: dict[tuple[str, str], set[str]] = defaultdict(set)
    for x in examples:
        docs[x.partition][x.document_fingerprint] = x.document
        responses[(x.partition, x.document_fingerprint)].add(x.response_id)
    index: dict[str, ReferenceIndex] = {}
    for part, d in docs.items():
        ix = ReferenceIndex()
        for fp, text in d.items():
            ix.add(fp, text)
        index[part] = ix
    later = {"train": ("validation", "test"), "validation": ("test",), "test": ()}
    exclusions: list[dict[str, Any]] = []
    drop: set[tuple[str, str]] = set()
    for part in ("train", "validation"):
        for fp, text in sorted(docs[part].items()):
            for other in later[part]:
                found = index[other].matches(fp, text)
                if found:
                    m = found[0]
                    exclusions.append({"rule": "internal-later-partition-v1", "partition": part, "fingerprint": fp,
                                       "partner_partition": other, "partner_fingerprint": m.reference_name, "kind": m.kind,
                                       "jaccard": round(m.jaccard, 4), "response_ids": sorted(responses[(part, fp)])})
                    drop.add((part, fp))
                    break
    kept = [x for x in examples if (x.partition, x.document_fingerprint) not in drop]
    groups: dict[str, str] = {}
    for part in PARTITIONS:
        fps = sorted(fp for fp in docs[part] if (part, fp) not in drop)
        ix = ReferenceIndex()
        for fp in fps:
            ix.add(fp, docs[part][fp])
        parent = {fp: fp for fp in fps}

        def find(a: str, parent: dict[str, str] = parent) -> str:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for fp in fps:
            for m in ix.matches(fp, docs[part][fp], exclude_name=fp):
                ra, rb = find(fp), find(m.reference_name)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        for fp in fps:
            groups[fp] = "g-" + find(fp)[:16]
    return kept, exclusions, groups


def build_manifest(examples: list[SentenceExample], groups: Mapping[str, str], *, meta: Mapping[str, Any],
                   response_labels: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Training manifest: identifiers, partitions, groups, labels and original offsets. No source or response text.

    `response_labels` are the AUTHORITATIVE response labels from the adapter (annotated wrapper segments included). When omitted
    they are derived from the in-scope sentences only, which is weaker and is not used by the pilot.
    """
    counts: dict[str, Any] = {}
    for part in PARTITIONS:
        ex = [x for x in examples if x.partition == part]
        unsupported = sum(1 - x.label for x in ex)
        resp_labels: dict[str, int] = {}
        for x in ex:
            derived = min(resp_labels.get(x.response_id, 1), x.label)
            resp_labels[x.response_id] = derived if response_labels is None else int(response_labels[x.response_id])
        share = unsupported / len(ex) if ex else 0.0
        counts[part] = {
            "documents": len({x.document_fingerprint for x in ex}),
            "independent_documents": len({groups[x.document_fingerprint] for x in ex}),
            "responses": len(resp_labels), "responses_unsupported": sum(1 - v for v in resp_labels.values()),
            "sentences": len(ex), "sentences_supported": len(ex) - unsupported, "sentences_unsupported": unsupported,
            "unsupported_sentence_share": round(share, 6),
            "all_supported_baseline_accuracy": round(1 - share, 6) if ex else 0.0,
        }
    columns = ["example_id", "response_id", "source_id", "partition", "document_fp16", "group_id", "label", "start", "end",
               "implicit_true", "crosses_boundary"]
    rows = [[x.example_id, x.response_id, x.source_id, x.partition, x.document_fingerprint[:16],
             groups[x.document_fingerprint], x.label, x.start, x.end, x.implicit_true, x.crosses_boundary]
            for x in sorted(examples, key=lambda e: e.example_id)]
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION, "policy": LABEL_POLICY_VERSION, "splitter": SENTENCE_SPLITTER_VERSION,
        "meta": dict(meta), "counts": counts, "columns": columns, "rows": rows,
        "response_labels": {rid: int(lab) for rid, lab in sorted(
            (x.response_id, (response_labels or {}).get(x.response_id, 1)) for x in examples)}
        if response_labels is not None else {},
        "permissions": {
            "training_use_cleared": False,
            "blocking_items": ["Third-party rights to the underlying CNN/DM article text are not established by this audit",
                               "Recent News is excluded pending written terms",
                               "Derived sentence labels are not independent expert sentence annotations"],
            "note": "technical eligibility only; the RAGTruth test partition is not for tuning and is not the RouteBench test"},
    }
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest


def run_audit(directory: Path, protected: Mapping[str, str], *,
              allowed_sources: frozenset[str] = DEFAULT_ALLOWED_SOURCES) -> dict[str, Any]:
    resp_path, src_path = directory / FILES[0], directory / FILES[1]
    responses, bad_r = read_jsonl(resp_path)
    sources, bad_s = read_jsonl(src_path)
    summary_ids = {str(s.get("source_id")) for s in sources if s.get("task_type") == "Summary"}
    probe = probe_offset_convention(r for r in responses if str(r.get("source_id")) in summary_ids)
    convention = decide_offset_convention(probe)
    result = adapt(responses, sources, allowed_sources=allowed_sources, offset_convention=convention)

    index = ReferenceIndex()
    for name, text in protected.items():
        index.add(name, text)
    docs = {x.document_fingerprint: x.document for x in result.examples}
    report_overlap = audit(docs, index)
    flagged = report_overlap.flagged_ids
    kept = AdapterResult(
        examples=[x for x in result.examples if x.document_fingerprint not in flagged],
        response_labels={k: v for k, v in result.response_labels.items()
                         if k in {x.response_id for x in result.examples if x.document_fingerprint not in flagged}},
        rejected=result.rejected, cross_split_documents=result.cross_split_documents, stats=result.stats)
    final_examples, internal_ex, groups = internal_overlap(kept.examples)
    final_ids = {x.response_id for x in final_examples}
    final = AdapterResult(
        examples=final_examples, response_labels={k: v for k, v in kept.response_labels.items() if k in final_ids},
        rejected=result.rejected, cross_split_documents=result.cross_split_documents, stats=result.stats)
    cluster_sizes = Counter(groups.values())
    meta = {"commit": COMMIT, "file_sha256": {resp_path.name: _sha256(resp_path), src_path.name: _sha256(src_path)},
            "offset_convention": convention, "allowed_sources": sorted(allowed_sources),
            "external_exclusion_fingerprints": sorted(flagged),
            "internal_exclusion_fingerprints": sorted(e["fingerprint"] for e in internal_ex)}
    manifest = build_manifest(final_examples, groups, meta=meta, response_labels=final.response_labels)
    source_counts = Counter(str(s.get("source")) for s in sources if s.get("task_type") == "Summary")
    return {
        "_manifest": manifest,
        "wrappers": {"segments_removed": len(result.wrappers), "by_reason": dict(Counter(w.reason for w in result.wrappers)),
                     "with_span_kept": result.stats.get("wrapper_with_span", 0)},
        "external_exclusions": [{"fingerprint": m.candidate_id, "reference": m.reference_name, "kind": m.kind,
                                 "jaccard": round(m.jaccard, 4)} for m in report_overlap.matches],
        "internal_overlap": {"rule": INTERNAL_RULE, "excluded_documents": len(internal_ex),
                             "excluded_by_partition": dict(Counter(e["partition"] for e in internal_ex)),
                             "exclusions": internal_ex},
        "near_duplicate_groups": {"groups": len(cluster_sizes),
                                  "groups_with_more_than_one_document": sum(1 for v in cluster_sizes.values() if v > 1),
                                  "largest_group_documents": max(cluster_sizes.values(), default=0)},
        "eligible_final": _summary(final),
        "manifest_sha256": manifest["manifest_sha256"],
        "policy": LABEL_POLICY_VERSION, "splitter": SENTENCE_SPLITTER_VERSION, "commit": COMMIT,
        "file_sha256": {resp_path.name: _sha256(resp_path), src_path.name: _sha256(src_path)},
        "malformed_lines": {"response": bad_r, "source_info": bad_s},
        "records": {"responses": len(responses), "sources": len(sources), "summary_sources_by_source": dict(source_counts)},
        "offset_probe": probe, "offset_convention": convention, "allowed_sources": sorted(allowed_sources),
        "rejections": dict(Counter(r.reason for r in result.rejected)),
        "eligible_before_overlap": _summary(result),
        "overlap": {"protected_documents_indexed": len(protected), "candidate_documents": report_overlap.checked,
                    "flagged_documents": len(flagged),
                    "matches": [{"candidate": m.candidate_id[:12], "reference": m.reference_name, "kind": m.kind,
                                 "jaccard": round(m.jaccard, 3)} for m in report_overlap.matches]},
        "eligible_after_overlap": _summary(kept),
        "cross_split_documents": len(result.cross_split_documents),
        "splitter_audit": splitter_audit(final, responses),
        "note": "data-availability audit only; no training, no model score. Rights for the underlying articles remain unresolved.",
    }


def compare_manifests(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """Measured effect of a policy change between two manifests (identifiers and labels only)."""
    cols_old, cols_new = old["columns"], new["columns"]
    io, in_ = cols_old.index("example_id"), cols_new.index("example_id")
    lo, ln = cols_old.index("label"), cols_new.index("label")
    po, pn = cols_old.index("partition"), cols_new.index("partition")
    a = {r[io]: r for r in old["rows"]}
    b = {r[in_]: r for r in new["rows"]}
    removed, added = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    both = sorted(set(a) & set(b))
    relabelled = [k for k in both if a[k][lo] != b[k][ln]]
    repartitioned = [k for k in both if a[k][po] != b[k][pn]]
    by_part = {p: {"removed": sum(1 for k in removed if a[k][po] == p), "added": sum(1 for k in added if b[k][pn] == p)}
               for p in PARTITIONS}
    return {"old": old.get("version"), "new": new.get("version"), "old_rows": len(a), "new_rows": len(b),
            "rows_removed": len(removed), "rows_added": len(added), "rows_relabelled": len(relabelled),
            "rows_repartitioned": len(repartitioned), "by_partition": by_part,
            "removed_unsupported": sum(1 for k in removed if a[k][lo] == 0),
            "added_unsupported": sum(1 for k in added if b[k][ln] == 0),
            "count_delta": {p: {k: new["counts"][p][k] - old["counts"][p][k]
                                for k in ("documents", "responses", "sentences", "sentences_unsupported")}
                            for p in PARTITIONS}}


def main() -> int:
    a = sys.argv[1:]
    if a[:1] == ["download"] and len(a) == 2:
        download(Path(a[1]))
        return 0
    if a[:1] == ["audit"] and len(a) == 2:
        d = Path(a[1])
        root = Path(__file__).resolve().parents[2]
        out = d / AUDIT_VERSION
        if out.exists():
            raise SystemExit(f"{out} already exists; refusing to overwrite a versioned audit")
        report = run_audit(d, protected_documents(root))
        manifest = report.pop("_manifest")
        sample = report.pop("splitter_audit")
        out.mkdir()
        (out / f"ingest_report_{AUDIT_VERSION}.json").write_text(json.dumps(report, indent=1), encoding="utf8")
        (out / f"training_manifest_{AUDIT_VERSION}.json").write_text(json.dumps(manifest), encoding="utf8")
        (out / f"splitter_review_sample_train_{AUDIT_VERSION}.json").write_text(json.dumps(sample, indent=1), encoding="utf8")
        print(json.dumps({k: v for k, v in report.items() if k not in ("internal_overlap", "external_exclusions", "overlap")},
                         indent=1))
        print("internal exclusions:", report["internal_overlap"]["excluded_documents"])
        return 0
    if a[:1] == ["compare"] and len(a) == 3:
        old = json.loads(Path(a[1]).read_text(encoding="utf8"))
        new = json.loads(Path(a[2]).read_text(encoding="utf8"))
        print(json.dumps(compare_manifests(old, new), indent=1))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


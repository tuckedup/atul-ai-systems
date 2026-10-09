"""Synthetic end-to-end test of the ingestion audit. No network, no real data."""
import json
from types import SimpleNamespace

import pytest
from evalops.pilot import ingest_audit as ia
from evalops.pilot.ragtruth import SentenceExample, normalized_fingerprint

DOC1 = ("The city council met on Monday and voted on the annual budget after a long debate about road repairs. "
        "Dr. Smith presented the figures. The library will receive new funding next year.")
DOC2 = ("A marine biologist described how coral reefs recover after bleaching events and why water temperature "
        "remains the dominant factor. Slow growing species need decades to return.")
DOC3 = ("Farmers in the valley reported an early harvest of apples this autumn after a warm summer. "
        "Prices at the market fell by a third compared with last year.")
DOC_PROTECTED = ("Farmers in the valley reported an early harvest of apples this autumn after a warm summer. "
                 "Prices at the market fell by a third compared with last year and stayed low.")


def write(tmp_path, responses, sources):
    (tmp_path / "response.jsonl").write_text("\n".join(json.dumps(r) for r in responses) + "\n", encoding="utf8")
    (tmp_path / "source_info.jsonl").write_text("\n".join(json.dumps(s) for s in sources) + "\n", encoding="utf8")


def src(sid, text, source="CNN/DM"):
    return {"source_id": sid, "task_type": "Summary", "source": source, "source_info": text, "prompt": "p"}


def resp(rid, sid, response, labels=(), split="train", quality="good"):
    return {"id": rid, "source_id": sid, "model": "m", "temperature": 0.5, "labels": list(labels), "split": split,
            "quality": quality, "response": response}


def lab(response, text, end_offset=0):
    s = response.index(text)
    return {"start": s, "end": s + len(text) + end_offset, "text": text, "label_type": "Evident Baseless Info", "meta": ""}


R1 = "The council voted on the budget. Dr. Smith presented the figures. The library gets a new roof."
R2 = "Coral reefs recover slowly. Temperature is the dominant factor."
R3 = "Farmers had an early harvest. Prices fell."


def test_end_to_end_report_counts_rejections_and_overlap(tmp_path):
    write(tmp_path,
          [resp("a", "s1", R1, [lab(R1, "a new roof")]), resp("b", "s2", R2), resp("c", "s3", R3),
           resp("d", "s4", R1, quality="truncated"), resp("e", "s5", R1)],
          [src("s1", DOC1), src("s2", DOC2), src("s3", DOC3), src("s4", DOC1), src("s5", DOC2, source="Recent News")])
    report = ia.run_audit(tmp_path, {"protected:1": DOC_PROTECTED})
    assert report["offset_convention"] == "exclusive"
    assert report["rejections"] == {"quality_truncated": 1, "source_not_allowed": 1}
    before, after = report["eligible_before_overlap"], report["eligible_after_overlap"]
    total_docs = sum(v["documents"] for v in before.values())
    assert total_docs == 3
    assert report["overlap"]["flagged_documents"] == 1
    assert sum(v["documents"] for v in after.values()) == 2
    assert sum(v["sentences_unsupported"] for v in after.values()) == 1
    assert len(report["file_sha256"]) == 2 and report["records"]["summary_sources_by_source"]["Recent News"] == 1
    assert report["splitter_audit"]["responses"] == 2 and "Dr. Smith presented the figures." in json.dumps(
        report["splitter_audit"]["sample"])


def test_offset_convention_is_decided_from_data_and_refuses_to_guess(tmp_path):
    inclusive = [resp("a", "s1", R1, [lab(R1, "a new roof", end_offset=-1)])]
    write(tmp_path, inclusive, [src("s1", DOC1)])
    assert ia.run_audit(tmp_path, {})["offset_convention"] == "inclusive"
    mixed = [resp("a", "s1", R1, [lab(R1, "a new roof")]), resp("b", "s2", R2, [lab(R2, "Temperature", -1)])]
    write(tmp_path, mixed, [src("s1", DOC1), src("s2", DOC2)])
    with pytest.raises(SystemExit):
        ia.run_audit(tmp_path, {})
    write(tmp_path, [resp("a", "s1", R1)], [src("s1", DOC1)])
    with pytest.raises(SystemExit):
        ia.run_audit(tmp_path, {})


def test_malformed_lines_are_counted_not_ignored(tmp_path):
    write(tmp_path, [resp("a", "s1", R1, [lab(R1, "a new roof")])], [src("s1", DOC1)])
    with (tmp_path / "response.jsonl").open("a", encoding="utf8") as fh:
        fh.write("{not json}\n[1,2]\n")
    report = ia.run_audit(tmp_path, {})
    assert report["malformed_lines"] == {"response": 2, "source_info": 0}


def test_protected_documents_collects_local_sources(tmp_path):
    data = tmp_path / "evalops" / "data"
    (data / "raw" / "tofueval_upstream").mkdir(parents=True)
    (data / "cases.jsonl").write_text(json.dumps({"case_id": "c1", "context": DOC1}) + "\n", encoding="utf8")
    (data / "raw" / "tofueval_upstream" / "docs.jsonl").write_text(
        json.dumps({"upstream_part": "dev", "doc_id": "d1", "text": DOC2}) + "\n", encoding="utf8")
    (data / "exemplars.json").write_text(json.dumps({"exemplars": {"summarize": [{"context": DOC3}]}}), encoding="utf8")
    names = set(ia.protected_documents(tmp_path))
    assert names == {"cases.jsonl:c1", "upstream:dev:d1", "exemplars.json:summarize:0"}


def test_splitter_review_does_not_expose_validation_or_test_responses():
    examples = [SimpleNamespace(response_id=part, partition=part, sentence_index=0,
                                sentence=f"Synthetic {part} text.", label=1)
                for part in ("train", "validation", "test")]
    result = ia.splitter_audit(SimpleNamespace(examples=examples), [])
    assert result["partition"] == "train"
    assert [row["response_id"] for row in result["sample"]] == ["train"]


# ---- internal cross-partition overlap safeguard and versioned manifest ------------------------------------------------

BASE = ("The county board voted on Tuesday to adopt the annual budget after a lengthy debate about road repairs and library "
        "funding while several residents spoke in favour of a new community centre near the river.")
NEAR = BASE.replace("Tuesday", "Wednesday") + " The clerk recorded the vote."
FAR1 = ("Marine biologists described how coral reefs recover after bleaching events and why water temperature remains the "
        "dominant factor for slow growing species.")
FAR2 = ("Farmers in the valley reported an early harvest of apples this autumn after a warm summer and prices at the market "
        "fell by a third compared with last year.")


def ex(doc, partition, rid, label=1, idx=0):
    return SentenceExample(example_id=f"{rid}#{idx}", response_id=rid, source_id="s", original_split="train",
                           partition=partition, document=doc, document_fingerprint=normalized_fingerprint(doc),
                           sentence="Some sentence text.", sentence_index=idx, start=0, end=19, label=label)


def test_internal_overlap_excludes_train_and_validation_documents_matching_a_later_partition():
    examples = [ex(BASE, "train", "t1"), ex(FAR1, "validation", "v1"), ex(NEAR, "validation", "v2"),
                ex(FAR2, "train", "t2"), ex(FAR1 + " Reefs also depend on light.", "test", "x1"),
                ex(NEAR + " Afterwards the meeting closed.", "test", "x2")]
    kept, exclusions, _ = ia.internal_overlap(examples)
    by = {(e["partition"], e["fingerprint"][:8]): e for e in exclusions}
    excluded = {(e["partition"], e["response_ids"][0]) for e in exclusions}
    assert ("train", "t1") in excluded  # matches the validation/test copies of the same document
    assert ("validation", "v1") in excluded and ("validation", "v2") in excluded  # match test documents
    assert {x.response_id for x in kept} == {"t2", "x1", "x2"}
    assert all(e["rule"] and e["partner_partition"] in ("validation", "test") and e["kind"] for e in exclusions)
    assert len(by) == len(exclusions)


def test_within_partition_near_duplicates_share_a_group_id():
    examples = [ex(BASE, "train", "t1"), ex(NEAR, "train", "t2"), ex(FAR1, "train", "t3")]
    kept, exclusions, groups = ia.internal_overlap(examples)
    assert exclusions == [] and len(kept) == 3
    g = {normalized_fingerprint(d): groups[normalized_fingerprint(d)] for d in (BASE, NEAR, FAR1)}
    assert g[normalized_fingerprint(BASE)] == g[normalized_fingerprint(NEAR)] != g[normalized_fingerprint(FAR1)]


def test_manifest_has_ids_and_hashes_but_no_source_or_response_text():
    examples = [ex(BASE, "train", "t1", label=0), ex(BASE, "train", "t1", label=1, idx=1), ex(FAR2, "validation", "v1")]
    _, _, groups = ia.internal_overlap(examples)
    m = ia.build_manifest(examples, groups, meta={"commit": "c"})
    blob = json.dumps(m)
    assert "Some sentence text" not in blob and "county board" not in blob
    assert m["counts"]["train"]["sentences"] == 2 and m["counts"]["train"]["independent_documents"] == 1
    assert m["counts"]["train"]["unsupported_sentence_share"] == 0.5
    assert m["manifest_sha256"] == ia.build_manifest(examples, groups, meta={"commit": "c"})["manifest_sha256"]
    assert m["permissions"]["training_use_cleared"] is False


def test_compare_manifests_reports_added_removed_and_relabelled_rows():
    examples = [ex(BASE, "train", "t1", label=1), ex(FAR2, "validation", "v1", label=1)]
    _, _, groups = ia.internal_overlap(examples)
    old = ia.build_manifest(examples, groups, meta={})
    changed = [ex(BASE, "train", "t1", label=0), ex(FAR1, "test", "x1", label=0)]
    _, _, groups2 = ia.internal_overlap(changed)
    new = ia.build_manifest(changed, groups2, meta={})
    diff = ia.compare_manifests(old, new)
    assert diff["rows_removed"] == 1 and diff["rows_added"] == 1 and diff["rows_relabelled"] == 1
    assert diff["by_partition"]["validation"]["removed"] == 1 and diff["by_partition"]["test"]["added"] == 1
    assert diff["added_unsupported"] == 1


def test_manifest_carries_authoritative_response_labels_including_annotated_wrappers():
    examples = [ex(BASE, "train", "t1", label=1), ex(BASE, "train", "t1", label=1, idx=1), ex(FAR2, "validation", "v1", label=1)]
    _, _, groups = ia.internal_overlap(examples)
    # t1 is unsupported at RESPONSE level (an annotated wrapper segment) although both in-scope sentences are supported
    m = ia.build_manifest(examples, groups, meta={}, response_labels={"t1": 0, "v1": 1})
    assert m["response_labels"] == {"t1": 0, "v1": 1}
    assert m["counts"]["train"]["responses_unsupported"] == 1
    assert m["version"] == "ragtruth-training-manifest-v5"

"""Synthetic-fixture tests for the RAGTruth adapter. No real data, no network."""
from itertools import pairwise

import pytest
from evalops.pilot.ragtruth import (
    adapt,
    normalized_fingerprint,
    probe_offset_convention,
    sentence_spans,
)

DOC_A = "The city council met on Monday. It approved a new budget of five million dollars. Roads will be repaired."
DOC_B = "A bakery opened downtown last week. The owner previously worked in Paris."
RESP = "The council met Monday. It approved a six million dollar budget. Roads will be repaired."


def src(sid, text=DOC_A, source="CNN/DM", task="Summary"):
    return {"source_id": sid, "task_type": task, "source": source, "source_info": text, "prompt": "p"}


def span(text, resp=RESP, label_type="Evident Conflict", **extra):
    start = resp.index(text)
    return {"start": start, "end": start + len(text), "text": text, "label_type": label_type, "meta": "", **extra}


def rsp(rid="r1", sid="s1", split="train", labels=None, quality="good", response=RESP):
    return {"id": rid, "source_id": sid, "model": "m", "temperature": 0.7, "labels": [] if labels is None else labels,
            "split": split, "quality": quality, "response": response}


def reasons(res):
    return {r.reason for r in res.rejected}


def test_sentence_spans_are_offset_preserving():
    spans = sentence_spans(RESP)
    assert [RESP[s:e] for s, e in spans] == ["The council met Monday.", "It approved a six million dollar budget.",
                                             "Roads will be repaired."]
    assert sentence_spans("  \n ") == []
    assert sentence_spans("One line.\nSecond line here.") == [(0, 9), (10, 27)]
    assert sentence_spans("One line\nSecond line here.") == [(0, 26)]  # wrapped line: no terminal punctuation, no cut


def test_unlabelled_good_response_is_supported_only_when_labels_is_an_explicit_list():
    res = adapt([rsp()], [src("s1")])
    assert [x.label for x in res.examples] == [1, 1, 1]
    assert res.response_labels == {"r1": 1}
    bad = rsp()
    del bad["labels"]
    assert reasons(adapt([bad], [src("s1")])) == {"missing_labels"}
    bad2 = rsp()
    bad2["labels"] = None
    assert reasons(adapt([bad2], [src("s1")])) == {"missing_labels"}


def test_span_marks_only_the_intersecting_sentence_and_aggregates_response():
    res = adapt([rsp(labels=[span("six million")])], [src("s1")])
    assert [x.label for x in res.examples] == [1, 0, 1]
    assert res.response_labels["r1"] == 0


def test_overlapping_spans_merge_and_boundary_crossing_span_marks_every_touched_sentence():
    a, b = span("six million dollar"), span("million dollar budget")
    res = adapt([rsp(labels=[a, b])], [src("s1")])
    assert [x.label for x in res.examples] == [1, 0, 1]
    cross = span("Monday. It approved")
    res2 = adapt([rsp(labels=[cross])], [src("s1")])
    assert [x.label for x in res2.examples] == [0, 0, 1]
    assert res2.examples[0].crosses_boundary and res2.examples[1].crosses_boundary


def test_offset_problems_reject_the_response_explicitly():
    bad = span("six million")
    bad["text"] = "seven million"
    assert reasons(adapt([rsp(labels=[bad])], [src("s1")])) == {"offset_mismatch"}
    oob = {**span("six million"), "end": 10_000}
    assert reasons(adapt([rsp(labels=[oob])], [src("s1")])) == {"offset_out_of_range"}
    weird = {**span("six million"), "start": "3"}
    assert reasons(adapt([rsp(labels=[weird])], [src("s1")])) == {"malformed_label"}


def test_inclusive_convention_is_explicit_and_probeable():
    s = span("six million")
    inclusive = {**s, "end": s["end"] - 1}
    assert probe_offset_convention([rsp(labels=[inclusive])])["inclusive"] == 1
    assert probe_offset_convention([rsp(labels=[s])])["exclusive"] == 1
    res = adapt([rsp(labels=[inclusive])], [src("s1")], offset_convention="inclusive")
    assert [x.label for x in res.examples] == [1, 0, 1]
    assert reasons(adapt([rsp(labels=[inclusive])], [src("s1")])) == {"offset_mismatch"}


def test_unknown_label_type_due_to_null_and_gap_spans_are_rejected():
    assert reasons(adapt([rsp(labels=[span("six million", label_type="Mystery")])], [src("s1")])) == {"unknown_label_type"}
    assert reasons(adapt([rsp(labels=[span("six million", due_to_null=True)])], [src("s1")])) == {"unexpected_due_to_null"}
    gap = {"start": RESP.index(". It") + 1, "end": RESP.index(". It") + 2, "text": " ", "label_type": "Evident Conflict",
           "meta": ""}
    assert reasons(adapt([rsp(labels=[gap])], [src("s1")])) == {"span_in_gap"}


def test_implicit_true_is_never_silently_supported():
    lab = span("six million", implicit_true=True)
    default = adapt([rsp(labels=[lab])], [src("s1")])
    assert [x.label for x in default.examples] == [1, 0, 1] and default.examples[1].implicit_true
    excluded = adapt([rsp(labels=[lab])], [src("s1")], implicit_true_policy="exclude")
    assert excluded.examples == [] and reasons(excluded) == {"implicit_true_excluded"}
    with pytest.raises(ValueError):
        adapt([], [], implicit_true_policy="supported")


def test_quality_flags_and_unknown_quality():
    assert reasons(adapt([rsp(quality="truncated")], [src("s1")])) == {"quality_truncated"}
    assert reasons(adapt([rsp(quality="incorrect_refusal")], [src("s1")])) == {"quality_incorrect_refusal"}
    assert reasons(adapt([rsp(quality="???")], [src("s1")])) == {"unknown_quality"}


def test_source_join_task_and_provisional_source_exclusions():
    assert reasons(adapt([rsp()], [])) == {"source_join_failed"}
    assert reasons(adapt([rsp()], [src("s1"), src("s1")])) == {"source_join_failed"}
    assert reasons(adapt([rsp()], [src("s1", task="QA")])) == {"not_summary_task"}
    assert reasons(adapt([rsp()], [src("s1", source="Recent News")])) == {"source_not_allowed"}
    ok = adapt([rsp()], [src("s1", source="Recent News")], allowed_sources=frozenset({"Recent News"}))
    assert len(ok.examples) == 3
    dict_src = {**src("s1"), "source_info": {"passage": "x"}}
    assert reasons(adapt([rsp()], [dict_src])) == {"source_info_not_text"}
    assert reasons(adapt([{**rsp(), "split": "dev"}], [src("s1")])) == {"invalid_split"}


def test_ids_and_splits_preserved_and_documents_never_straddle_partitions():
    sources = [src("s1"), src("s2", DOC_A.upper()), src("s3", DOC_B), src("s4", DOC_B + " ")]
    resp = [rsp("r1", "s1"), rsp("r2", "s2"), rsp("r3", "s3"), rsp("r4", "s4"), rsp("r5", "s3", split="train")]
    res = adapt(resp, sources, validation_fraction=0.5, seed=7)
    by_doc = {}
    for x in res.examples:
        by_doc.setdefault(x.document_fingerprint, set()).add(x.partition)
        assert x.example_id.startswith(x.response_id) and x.original_split == "train"
    assert all(len(v) == 1 for v in by_doc.values())
    assert normalized_fingerprint(DOC_A) == normalized_fingerprint(DOC_A.upper())
    assert len(by_doc) == 2


def test_document_in_both_original_splits_is_excluded_and_reported():
    res = adapt([rsp("r1", "s1", split="train"), rsp("r2", "s2", split="test")], [src("s1"), src("s2")])
    assert res.examples == []
    assert reasons(res) == {"document_in_train_and_test"}
    assert res.cross_split_documents == [normalized_fingerprint(DOC_A)]


def test_original_test_documents_are_kept_in_the_test_partition():
    res = adapt([rsp("r1", "s1", split="test")], [src("s1")])
    assert {x.partition for x in res.examples} == {"test"}


def test_malformed_records_are_rejected_not_defaulted():
    assert reasons(adapt([{"id": "r1"}], [src("s1")])) == {"malformed_record"}
    assert reasons(adapt([rsp(response="")], [src("s1")])) == {"malformed_record"}


# ---- regression tests for three independently reproduced defects -------------------------------------------------------
def test_overlap_is_detected_even_when_the_test_response_is_later_rejected():
    for bad_test in (rsp("r2", "s2", split="test", quality="truncated"),
                     {k: v for k, v in rsp("r2", "s2", split="test").items() if k != "labels"},
                     rsp("r2", "s2", split="test", labels=[{**span("six million"), "text": "wrong"}])):
        res = adapt([rsp("r1", "s1", split="train"), bad_test], [src("s1"), src("s2")])
        assert res.examples == [], "a document with test membership must not survive as training data"
        assert res.cross_split_documents == [normalized_fingerprint(DOC_A)]
        assert {r.response_id: r.reason for r in res.rejected}["r1"] == "document_in_train_and_test"


def test_duplicate_response_ids_are_rejected_without_silent_overwrite():
    a = rsp("dup", "s1")
    b = rsp("dup", "s1", labels=[span("six million")])
    res = adapt([a, b, rsp("ok", "s3")], [src("s1"), src("s3", DOC_B)])
    assert sorted(r.reason for r in res.rejected) == ["duplicate_response_id", "duplicate_response_id"]
    assert {x.response_id for x in res.examples} == {"ok"}
    assert set(res.response_labels) == {"ok"}
    assert len({x.example_id for x in res.examples}) == len(res.examples)


def _texts(t):
    return [t[s:e] for s, e in sentence_spans(t)]


def test_titles_initials_and_abbreviations_do_not_split_sentences():
    assert _texts("Dr. Smith approved the budget.") == ["Dr. Smith approved the budget."]
    assert _texts("J. K. Rowling wrote it. Critics agreed.") == ["J. K. Rowling wrote it.", "Critics agreed."]
    assert _texts("Sen. Mary L. Jones said the U.S. economy grew in Aug. 2020. The rate rose.") == [
        "Sen. Mary L. Jones said the U.S. economy grew in Aug. 2020.", "The rate rose."]
    assert _texts("He works for Acme Inc. The firm is small.") == ["He works for Acme Inc.", "The firm is small."]
    assert _texts("The rate was 3.5 percent. It fell.") == ["The rate was 3.5 percent.", "It fell."]
    assert _texts("Mr. and Mrs. Lee arrived at 5 p.m. They left later.") == [
        "Mr. and Mrs. Lee arrived at 5 p.m.", "They left later."]


def test_quotations_and_line_wrapping_are_handled():
    assert _texts("“We will act,” she said. “Soon.”") == [
        "“We will act,” she said.", "“Soon.”"]
    assert _texts('He asked, "Is it done?" She nodded.') == ['He asked, "Is it done?"', "She nodded."]
    assert _texts("The council met on\nMonday and approved the\nbudget. Roads will be repaired.") == [
        "The council met on\nMonday and approved the\nbudget.", "Roads will be repaired."]
    assert _texts("Budget approved\n\nRoads will be repaired.") == ["Budget approved", "Roads will be repaired."]
    assert _texts("- Roads repaired\n- Library funded") == ["- Roads repaired", "- Library funded"]
    assert _texts("It ended.\nNext item begins here.") == ["It ended.", "Next item begins here."]


def test_offsets_always_index_the_original_text_and_never_overlap():
    text = 'Dr. Smith said, "Yes." The U.S. team won.\n\nThen it rained on\nTuesday.'
    spans = sentence_spans(text)
    assert all(0 <= s < e <= len(text) and text[s:e] == text[s:e].strip() for s, e in spans)
    assert all(a[1] <= b[0] for a, b in pairwise(spans))


def test_single_newline_respects_titles_abbreviations_and_bare_closing_quotes():
    for text in ("Dr.\nSmith approved the budget.", "The U.S.\neconomy grew.",
                 'The report called it "promising"\nand recommended further study.',
                 "J.\nK. Rowling wrote it.", "The board met in Aug.\n2020 was a busy year."):
        assert sentence_spans(text) == [(0, len(text))], text
    # an abbreviation followed by a real sentence starter on the next line still ends the sentence
    two = "He moved to the U.S.\nThe economy grew."
    assert [two[s:e] for s, e in sentence_spans(two)] == ["He moved to the U.S.", "The economy grew."]
    # declared behaviour retained: terminal punctuation, blank lines and list markers separate
    assert len(sentence_spans("It ended.\nNext item begins here.")) == 2
    assert len(sentence_spans('He said "stop."\nShe left.')) == 2
    assert len(sentence_spans("Budget approved\n\nRoads will be repaired.")) == 2
    assert len(sentence_spans("- Roads repaired\n- Library funded")) == 2


# ---- policy ragtruth-sentence-v2 / splitter regex-v3 ------------------------------------------------------------------
def test_html_entities_act_as_whitespace_without_changing_offsets():
    t = "The panel met in room II.&nbsp;The women spoke first."
    spans = sentence_spans(t)
    assert [t[s:e] for s, e in spans] == ["The panel met in room II.", "The women spoke first."]
    amp = "Tom &amp; Jerry left the show."
    assert [amp[s:e] for s, e in sentence_spans(amp)] == [amp]  # a non-space entity is left in place


def test_abbreviation_inside_an_open_quotation_does_not_end_a_sentence():
    t = 'The show "A.D. The Bible Continues" premiered in April. It ran for ten weeks.'
    assert [t[s:e] for s, e in sentence_spans(t)] == [
        'The show "A.D. The Bible Continues" premiered in April.', "It ran for ten weeks."]
    assert len(sentence_spans("He earned a Ph.D. Then he left.")) == 2
    assert len(sentence_spans("He met R.J. Reynolds at noon.")) == 1


def wrapped(response, labels=()):
    return adapt([rsp(response=response, labels=list(labels))], [src("s1")])


def test_summary_preface_wrappers_are_not_supported_training_sentences():
    r = "Sure!\nHere is a summary of the article in 50 words:\n\nThe council met Monday. It approved the budget."
    res = wrapped(r)
    assert [x.sentence for x in res.examples] == ["The council met Monday.", "It approved the budget."]
    assert [w.reason for w in res.wrappers] == ["interjection", "summary_preface"]
    assert res.response_labels == {"r1": 1}
    assert res.stats["wrapper_sentences"] == 2
    only = wrapped("Sure!\nHere is a summary of the article in 50 words:")
    assert only.examples == [] and reasons(only) == {"no_content_sentences"}


def test_wrappers_are_only_removed_as_a_prefix():
    mid = wrapped("The council met Monday. Sure! It approved the budget.")
    assert [x.sentence for x in mid.examples] == ["The council met Monday.", "Sure!", "It approved the budget."]
    assert wrapped("Sure, the council met Monday.").examples[0].sentence == "Sure, the council met Monday."


# ---- pilot-readiness pass: text-only input construction ---------------------------------------------------------------
def segments(res):
    return [(x.sentence_index, x.sentence, x.start, x.end) for x in res.examples]


def test_input_segments_do_not_depend_on_annotations():
    r = "Sure!\nHere is a summary of the article in 50 words:\n\nThe council met Monday. It approved the six million dollar budget."
    plain = wrapped(r)
    for target in ("Sure", "Here is a summary", "council met", "six million"):
        annotated = wrapped(r, [span(target, resp=r)])
        assert segments(annotated) == segments(plain), target
        assert annotated.wrappers == [] or [w.sentence_index for w in annotated.wrappers] == [0, 1]
    assert [w.sentence_index for w in plain.wrappers] == [0, 1]


def test_annotated_wrapper_content_is_reported_and_keeps_the_response_unsupported():
    r = "Sure! The council met Monday."
    touched = wrapped(r, [span("Sure", resp=r)])
    assert [x.sentence for x in touched.examples] == ["The council met Monday."]
    assert touched.response_labels == {"r1": 0}  # the annotated span is not lost at response level
    assert touched.stats["wrapper_with_span"] == 1 and touched.wrappers[0].unsupported is True
    clean = wrapped(r)
    assert clean.response_labels == {"r1": 1} and clean.wrappers[0].unsupported is False


def test_list_prefaces_that_carry_specific_content_are_kept():
    kept = "Here are the best and worst moments from the 2015 MTV Movie Awards according to Billboard:"
    res = wrapped(kept + "\n\nThe host opened the show.")
    assert res.wrappers == [] and res.examples[0].sentence == kept
    assert [w.reason for w in wrapped("Here are a few recent examples:\n\nThe host opened.").wrappers] == ["list_preface"]


def test_period_before_a_closing_quote_can_end_the_sentence():
    t = 'He said "I live in the U.S." She left.'
    assert [t[s:e] for s, e in sentence_spans(t)] == ['He said "I live in the U.S."', "She left."]
    q = 'The show "A.D. The Bible Continues" premiered in April. It ran for ten weeks.'
    assert len(sentence_spans(q)) == 2 and sentence_spans(q)[0][1] == q.index("April.") + 6
    assert len(sentence_spans('He said "Mr. Smith left." She agreed.')) == 2


def test_list_prefaces_and_numbered_list_markers():
    for r in ("Here are a few recent examples:\n\nThe council met Monday.",
              "Here are the top news stories from this week:\nThe council met Monday."):
        res = wrapped(r)
        assert [w.reason for w in res.wrappers] == ["list_preface"]
        assert [x.sentence for x in res.examples] == ["The council met Monday."]
    keep = wrapped("The council said this: it approved the budget.")  # a colon inside a normal claim is not a preface
    assert len(keep.examples) == 1 and keep.wrappers == []
    numbered = "1. The council met Monday.\n2. Roads will be repaired."
    assert [numbered[s:e] for s, e in sentence_spans(numbered)] == ["1. The council met Monday.", "2. Roads will be repaired."]
    assert len(sentence_spans("He was 86. She was 80.")) == 2  # a number ending a sentence still splits

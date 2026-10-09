"""Synthetic-fixture tests for the exact / near-duplicate audit interface."""
from evalops.pilot.audit import ReferenceIndex, audit, exact_fingerprint, shingle_hashes

BASE = ("The county board voted on Tuesday to adopt the annual budget after a lengthy debate about road repairs "
        "and library funding, while several residents spoke in favour of a new community centre.")
OTHER = ("A marine biologist described how coral reefs recover after bleaching events, noting that slow growing "
         "species take decades to return and that water temperature remains the dominant factor.")


def test_exact_match_ignores_case_punctuation_and_spacing():
    ix = ReferenceIndex()
    ix.add("ref-1", BASE)
    m = ix.check("c1", BASE.upper().replace(",", " ;") + "  ")
    assert m is not None and m.kind == "exact" and m.reference_name == "ref-1"
    assert exact_fingerprint(BASE) == exact_fingerprint(BASE.lower())


def test_near_duplicate_is_flagged_and_unrelated_is_not():
    ix = ReferenceIndex()
    ix.add("ref-1", BASE)
    edited = BASE.replace("Tuesday", "Wednesday") + " The vote was recorded by the clerk afterwards."
    m = ix.check("c1", edited)
    assert m is not None and m.kind == "near_duplicate" and m.jaccard > 0.5
    assert ix.check("c2", OTHER) is None


def test_containment_catches_an_excerpt():
    ix = ReferenceIndex()
    long_ref = BASE + " " + OTHER
    ix.add("ref-long", long_ref)
    m = ix.check("excerpt", BASE)
    assert m is not None and m.containment_of_candidate >= 0.8


def test_short_documents_only_match_exactly():
    ix = ReferenceIndex()
    ix.add("short", "too short")
    assert shingle_hashes("too short") == frozenset()
    assert ix.check("c", "too short!").kind == "exact"
    assert ix.check("c", "another short text") is None


def test_saved_index_holds_no_reference_text_and_round_trips(tmp_path):
    ix = ReferenceIndex()
    ix.add("reserve-1", BASE)
    path = tmp_path / "ref.json"
    ix.save(path)
    raw = path.read_text(encoding="utf8")
    assert "county" not in raw and "budget" not in raw and "reserve-1" in raw
    loaded = ReferenceIndex.load(path)
    assert loaded.check("c", BASE) is not None and loaded.check("c", OTHER) is None
    assert len(loaded) == 1


def test_audit_reports_every_candidate_and_drops_none():
    ix = ReferenceIndex()
    ix.add("ref-1", BASE)
    report = audit({"dup": BASE, "clean": OTHER}, ix)
    assert report.checked == 2 and report.flagged_ids == {"dup"} and report.clean_ids == ["clean"]

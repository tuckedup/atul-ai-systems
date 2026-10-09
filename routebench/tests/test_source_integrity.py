import json

import pytest
from evalops.source_integrity import build, restore


def transcript(text, doc_id="doc1"):
    return {"text": text, "doc_id": doc_id, "source": "MediaSum", "upstream_part": "test"}


def case(cid="c1", context="A source.", candidate="Unsupported is a valid negative.", group="g1"):
    return {"case_id": cid, "group_id": group, "context": context, "candidate_output": candidate,
            "task_input": "Summarize", "reference": "", "meta": {"subset": "TofuEval-MediaS"}}


def inputs(tmp_path, rows, upstream=None):
    paths = [tmp_path / name for name in ("cases.jsonl", "splits.json", "upstream.jsonl")]
    paths[0].write_text("\n".join(json.dumps(r) for r in rows), encoding="utf8")
    paths[1].write_text(json.dumps({"assignment": {"g1": "train", "g2": "dev", "g3": "test"}}))
    paths[2].write_text("\n".join(json.dumps(r) for r in (upstream or [])), encoding="utf8")
    return paths


def test_quote_repair_preserves_all_other_characters_and_newlines():
    text = 'He said \u201cyes\u201c.\r\n'
    fixed, provenance = restore(text, [transcript('He said "yes".\n')])
    assert fixed == 'He said "yes".\r\n'
    assert provenance["doc_id"] == "doc1"


@pytest.mark.parametrize("target", ['She said "yes".', 'He said "no".', 'He said "yes". Extra.'])
def test_no_semantic_or_truncation_repair(target):
    text = 'He said \u201cyes\u201c.'
    assert restore(text, [transcript(target)]) == (text, None)


def test_ambiguous_matches_are_not_guessed():
    text = 'He said \u201cyes\u201c.'
    assert restore(text, [transcript('He said "yes".'), transcript('He said "yes".', "doc2")]) == (text, None)


def test_replacement_recovers_only_punctuation():
    assert restore("They\ufffdre here.", [transcript("They're here.")])[0] == "They're here."
    assert restore("\ufffdere.", [transcript("Here.")]) == ("\ufffdere.", None)


def test_quarantine_preservation_namespace_and_test_exclusion(tmp_path):
    rows = [case(context='He said \u201cyes\u201c.'),
            case("bad", candidate="Damaged\ufffd", group="g2"),
            case("negative", group="g2"),
            case("heldout", context=None, candidate=None, group="g3")]
    paths = inputs(tmp_path, rows, [transcript('He said "yes".')])
    before = [p.read_bytes() for p in paths]
    out = tmp_path / "result"
    report = build(*paths, out)
    assert [p.read_bytes() for p in paths] == before
    assert report["counts"] == {"input_train": 1, "kept_train": 1, "input_dev": 2,
                                "quarantined_dev": 1, "kept_dev": 1}
    assert report["repaired_cases"] == report["repaired_documents"] == 1
    kept = [json.loads(l) for l in (out / "cases.jsonl").read_text(encoding="utf8").splitlines()]
    assert [r["case_id"] for r in kept] == ["source-integrity-v1:c1", "source-integrity-v1:negative"]
    assert json.loads((out / "quarantine.jsonl").read_text(encoding="utf8"))["case"] == rows[1]
    assert all(r["original_case_id"] != "heldout" for r in report["case_mapping"])
    assert not report["labels_read"] and not report["predictions_read"]
    assert not (out / "annotations.jsonl").exists()
    with pytest.raises(FileExistsError):
        build(*paths, out)


def test_reserved_upstream_dev_cannot_repair(tmp_path):
    upstream = transcript('He said "yes".')
    upstream["upstream_part"] = "dev"
    paths = inputs(tmp_path, [case(context='He said \u201cyes\u201c.')], [upstream])
    assert build(*paths, tmp_path / "out")["repaired_cases"] == 0


@pytest.mark.parametrize("rows", [[case(), case()], [case(context=None)]])
def test_malformed_scope_aborts_before_publication(tmp_path, rows):
    paths = inputs(tmp_path, rows)
    with pytest.raises(ValueError):
        build(*paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_bad_json_aborts_not_silent_exclusion(tmp_path):
    paths = inputs(tmp_path, [case()])
    paths[0].write_text("{bad json")
    with pytest.raises(ValueError):
        build(*paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()

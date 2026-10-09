import json

import pytest
from evalops import source_integrity_v2 as v2


def case():
    return {"case_id": "c1", "group_id": "g1", "context": "A" * 6000,
            "candidate_output": "Unchanged candidate", "task_input": "Summarize", "reference": "",
            "meta": {"subset": "AggreFact-CNN", "source_row_index": 12}}


def raw():
    return {"doc": "A" * 6000 + " Full ending.", "claim": "Unchanged candidate",
            "subset": "AggreFact-CNN", "source_row_index": 12, "label": 0}


def test_exact_prefix_full_source_only():
    assert v2.restore_full(case(), raw()) == raw()["doc"]


@pytest.mark.parametrize("field,value", [("doc", "Different document"), ("claim", "Different candidate"),
                                         ("subset", "TofuEval-MediaS"), ("doc", "A" * 6000 + "\ufffd")])
def test_wrong_binding_or_damaged_raw_source_refused(field, value):
    row = raw()
    row[field] = value
    with pytest.raises(ValueError):
        v2.restore_full(case(), row)


def prepare(tmp_path, rows):
    (tmp_path / "raw/tofueval_upstream").mkdir(parents=True)
    (tmp_path / "cases.jsonl").write_text(json.dumps(case()), encoding="utf8")
    (tmp_path / "splits.json").write_text(json.dumps({"assignment": {"g1": "dev"}}))
    (tmp_path / "raw/tofueval_upstream/docs.jsonl").write_text("")
    (tmp_path / "raw/aggrefact.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf8")


def test_copy_and_label_blindness(tmp_path):
    row = raw()
    row["label"] = {"this field": "is never used"}
    prepare(tmp_path, [row])
    original = (tmp_path / "cases.jsonl").read_bytes()
    report = v2.build(tmp_path, tmp_path / "out")
    exported = json.loads((tmp_path / "out/cases.jsonl").read_text(encoding="utf8"))
    assert exported["case_id"] == "source-integrity-v2:c1"
    assert exported["context"] == row["doc"]
    assert exported["candidate_output"] == case()["candidate_output"]
    assert not report["labels_read"] and len(report["full_source_repairs"]) == 1
    assert (tmp_path / "cases.jsonl").read_bytes() == original
    assert (tmp_path / "out/punctuation-stage/manifest.json").exists()
    assert not (tmp_path / "out/annotations.jsonl").exists()
    with pytest.raises(FileExistsError):
        v2.build(tmp_path, tmp_path / "out")


@pytest.mark.parametrize("rows", [[], [raw(), raw()]])
def test_missing_duplicate_source_refuses_terminal_report(tmp_path, rows):
    prepare(tmp_path, rows)
    with pytest.raises(ValueError):
        v2.build(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out/manifest.json").exists()

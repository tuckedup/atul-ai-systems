"""Tests for the offline blind-review packet (synthetic data; plus a leakage scan of the real packet when it exists). No model, no GPU, no label change."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from evalops import blind_review_packet as bp
from evalops import judge_v19_phase1 as jv


def world(n=120):
    ids = [f"gnd-{i:04d}" for i in range(n)]
    labels = {c: int(i % 2 == 0) for i, c in enumerate(ids)}
    verdicts = {}
    for i, c in enumerate(ids):
        if i % 10 == 0:  # persistent error: all five wrong
            verdicts[c] = [1 - labels[c]] * 5
        elif i % 7 == 0:  # disagreeing
            verdicts[c] = [labels[c], 1 - labels[c], labels[c], labels[c], 1 - labels[c]]
        else:
            verdicts[c] = [labels[c]] * 5
    cases = {c: SimpleNamespace(candidate_output=f"Sentence number {i} about the council.", context=f"Document {i}. The council met on Monday.\n  It voted on item {i}.") for i, c in enumerate(ids)}
    return ids, labels, verdicts, cases


def test_selection_is_deterministic_equal_sized_and_uses_only_unanimous_cases():
    ids, labels, verdicts, _ = world()
    a, b = bp.select(ids, labels, verdicts), bp.select(ids, labels, verdicts)
    assert a == b
    assert len(a["persistent"]) == len(a["controls"]) == 12 and not set(a["persistent"]) & set(a["controls"])
    assert all(len(set(verdicts[c])) == 1 and verdicts[c][0] != labels[c] for c in a["persistent"])
    assert all(len(set(verdicts[c])) == 1 and verdicts[c][0] == labels[c] for c in a["controls"])
    assert set(a["order"]) == set(a["persistent"]) | set(a["controls"]) and len(a["order"]) == 24


def test_order_is_shuffled_so_group_membership_is_not_readable_from_position():
    ids, labels, verdicts, _ = world()
    sel = bp.select(ids, labels, verdicts)
    flags = [c in set(sel["persistent"]) for c in sel["order"]]
    assert flags != sorted(flags) and flags != sorted(flags, reverse=True)
    assert sel["order"] != sorted(sel["order"])


def test_selection_refuses_when_the_control_pool_is_too_small():
    ids = [f"c{i}" for i in range(6)]
    labels = {c: 1 for c in ids}
    verdicts = {c: [0] * 5 for c in ids}  # every case is a persistent error, none correct
    with pytest.raises(bp.PacketError):
        bp.select(ids, labels, verdicts)


def test_packet_items_hold_only_id_sentence_and_source_and_reveal_nothing_else(tmp_path):
    ids, labels, verdicts, cases = world()
    sel = bp.select(ids, labels, verdicts)
    items, mapping = bp.build_items(sel["order"], cases)
    assert [set(i) for i in items] == [set(bp.PACKET_KEYS)] * 24 and [i["review_id"] for i in items] == [f"R{n:03d}" for n in range(1, 25)]
    hashes = bp.write_packet(tmp_path / "packet", items)
    text = "".join(p.read_text(encoding="utf8") for p in (tmp_path / "packet").iterdir())
    for secret in [*ids, "persistent", "control", "v18", "published label", "false pass", "false rejection", "unanimous"]:
        assert secret not in text, secret
    assert set(hashes) == {"review_items.jsonl", "review_items.md", "INSTRUCTIONS.md", "response_format_example.jsonl"} and len(mapping) == 24
    manifest = json.loads((tmp_path / "packet" / "packet_manifest.json").read_text(encoding="utf8"))
    assert manifest["items"] == 24 and "seed" not in json.dumps(manifest) and set(manifest["outcomes"]) == {"supported", "unsupported", "uncertain"}


def test_instructions_keep_three_outcomes_require_evidence_and_separate_ai_assistance():
    t = bp.INSTRUCTIONS.format(n=88)
    for needle in ("supported", "unsupported", "uncertain", "evidence", "explanation", "ai_assisted", "never counted as independent human annotation", "verbatim"):
        assert needle in t
    assert "Uncertain is a legitimate outcome" in t and "hypothesis" not in t.lower()


def make_packet(tmp_path):
    ids, labels, verdicts, cases = world()
    items, _ = bp.build_items(bp.select(ids, labels, verdicts)["order"], cases)
    bp.write_packet(tmp_path / "packet", items)
    return tmp_path / "packet", items


def resp(rid, **kw):
    base = {"review_id": rid, "reviewer_id": "r1", "reviewer_kind": "human", "outcome": "supported", "evidence": [], "explanation": "The sentence matches what the document states."}
    return {**base, **kw}


def test_validator_accepts_good_responses_including_whitespace_differences_in_quotes(tmp_path):
    packet, items = make_packet(tmp_path)
    src = items[0]["source"]
    quote = src.split("\n")[0] + " " + src.split("\n")[1].strip()  # joined across a newline: whitespace differs from the source
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(resp("R001", evidence=[quote])) + "\n" + json.dumps(resp("R002", outcome="uncertain", explanation="Cannot tell who the pronoun refers to here.")) + "\n", encoding="utf8")
    out = bp.validate_responses(packet, path)
    assert out["problems"] == [] and out["outcome_counts"] == {"supported": 1, "uncertain": 1}


@pytest.mark.parametrize("patch,needle", [({"outcome": "maybe"}, "outcome"), ({"reviewer_kind": "robot"}, "reviewer_kind"), ({"evidence": ["not in the document at all"]}, "verbatim"),
                                          ({"explanation": "ok"}, "explanation"), ({"reviewer_id": ""}, "reviewer_id"), ({"evidence": "text"}, "evidence"), ({"review_id": "R999"}, "unknown")])
def test_validator_flags_each_kind_of_malformed_response(tmp_path, patch, needle):
    packet, _ = make_packet(tmp_path)
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(resp("R001", **patch)) + "\n", encoding="utf8")
    assert any(needle in p for p in bp.validate_responses(packet, path)["problems"])


def test_validator_flags_duplicates_and_keeps_ai_assisted_responses_separate(tmp_path):
    packet, _ = make_packet(tmp_path)
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in (resp("R001"), resp("R001"), resp("R002", reviewer_kind="ai_assisted"))) + "not json\n", encoding="utf8")
    out = bp.validate_responses(packet, path)
    assert any("duplicate" in p for p in out["problems"]) and any("not valid JSON" in p for p in out["problems"])
    assert out["reviewer_kind_counts"] == {"human": 2, "ai_assisted": 1} and "never counted as independent human annotation" in out["note"]


def test_cli_validate_returns_nonzero_on_problems_and_build_refuses_without_hash(tmp_path):
    packet, _ = make_packet(tmp_path)
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(resp("R001", outcome="maybe")) + "\n", encoding="utf8")
    assert bp.main(["validate", "--packet", str(packet), "--responses", str(path)]) == 3
    assert bp.main(["build", "--out", str(tmp_path / "x")]) == 2


def test_the_real_packet_if_built_has_no_leak_and_reconciliation_is_separate():
    root = jv.WORK / "blind-review-v1"
    if not root.exists():
        pytest.skip("real packet not built")
    packet, recon = root / "packet", json.loads((root / "reconciliation" / "reconciliation.json").read_text(encoding="utf8"))
    text = "".join(p.read_text(encoding="utf8") for p in packet.iterdir())
    for it in recon["items"].values():
        assert it["case_id"] not in text and it["document_group"] not in text
    # selection vocabulary must not appear in the files WE write (documents are quoted material and may contain any word)
    ours = "".join((packet / n).read_text(encoding="utf8").lower() for n in ("INSTRUCTIONS.md", "packet_manifest.json", "response_format_example.jsonl"))
    for word in ("v18", "persistent", "false pass", "false rejection", "published label", "unanimous", "control", "seed", "gnd-"):
        assert word not in ours, word
    assert recon["selection"]["persistent_errors"] == recon["selection"]["controls"] == 44 and len(recon["items"]) == 88
    assert "No threshold" in recon["analysis_rule"] and recon["status"] == "DO NOT SHARE WITH REVIEWERS"
    assert not any(p.name == "reconciliation.json" for p in packet.rglob("*"))

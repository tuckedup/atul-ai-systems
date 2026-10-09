"""Offline tests for the data-quality workflow (dqdata) and the claim-relation candidate. Synthetic text only; no network, no GPU, no model."""
import json
from pathlib import Path

import pytest
from evalops import claim_relation as cr
from evalops import dqdata as dq
from evalops.pilot.audit import ReferenceIndex

SRC_TEXT = "The council met on Monday. The clerk read the minutes aloud and the mayor thanked staff for the quarterly report."


def perm(status="permitted"):
    return {k: {"status": status, "evidence": "synthetic test record" if status != "unresolved" else ""} for k in dq.PERMISSION_KINDS}


def source(sid="s1", family="fam-a", text=SRC_TEXT, **kw):
    row = {"source_id": sid, "document_family": family, "text": text, "text_sha256": dq.sha256(text), "provenance": "synthetic", "policy_version": dq.POLICY_VERSION,
           "collection_record": "synthetic-test", "source_permissions": perm()}
    row.update(kw)
    return row


def case(cid="c1", sid="s1", text=SRC_TEXT, cand="The clerk read the minutes aloud.", **kw):
    row = {"case_id": cid, "source_id": sid, "source_sha256": dq.sha256(text), "candidate": cand, "candidate_sha256": dq.sha256(cand) if cand is not None else None, "provenance": "synthetic",
           "annotation_permissions": {"train": {"status": "permitted", "evidence": "synthetic test record"}}}
    row.update(kw)
    return row


def human(aid, cid="c1", who="h1", outcome="supported", **kw):
    row = {"annotation_id": aid, "case_id": cid, "annotator_id": who, "kind": "human", "collection_process_id": "collect-1", "policy_version": dq.POLICY_VERSION,
           "outcome": outcome}
    row.update(kw)
    return row


def ai(aid, cid="c1", outcome="supported"):
    return {"annotation_id": aid, "case_id": cid, "annotator_id": "model-x", "kind": "ai", "model_identity": {"name": "mock", "version": "unknown"},
            "policy_version": dq.POLICY_VERSION, "outcome": outcome}


def write_incoming(tmp_path, sources=(), cases=(), annotations=(), adjudications=()):
    d = tmp_path / "incoming"
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sources", sources), ("cases", cases), ("annotations", annotations), ("adjudications", adjudications)):
        (d / f"{name}.jsonl").write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows), encoding="utf8")
    return d


def workspace(tmp_path, **kw):
    ws = dq.Workspace(tmp_path / "ws")
    dq.import_records(ws, write_incoming(tmp_path, **kw))
    return ws


# ------------------------------------------------------------------------------------------------ 1 leakage
def test_blind_packets_use_an_allow_list_and_cannot_carry_labels_or_predictions(tmp_path):
    c = case(label="LEAK-LABEL", v18_prediction="LEAK-PRED", target_score="LEAK-0.74", previous_review="LEAK-REVIEW", expected_outcome="LEAK-OUT")
    ws = workspace(tmp_path, sources=[source(provenance="synthetic")], cases=[c])
    packets, mapping = dq.build_packets(ws, salt="s")
    blob = json.dumps(packets)
    assert "LEAK" not in blob and set(packets[0]) == set(dq.PACKET_ALLOW) and "c1" not in blob and list(mapping.values()) == ["c1"]
    with pytest.raises(dq.DataQualityError):
        dq.assert_packet_clean({**packets[0], "label": 1})


def test_claim_relation_prompt_never_contains_labels_or_prior_verdicts():
    prompt = cr.build_prompt(SRC_TEXT, "The clerk read the minutes aloud.")
    assert not any(w in prompt.lower() for w in ("gold", "v18", "reference label", "expected"))
    assert SRC_TEXT in prompt


# ------------------------------------------------------------------------------------------------ 2 permissions
def _full_workspace(tmp_path, train="permitted", ann_train="permitted"):
    s = source(source_permissions={**perm(), "train": {"status": train, "evidence": "e" if train != "unresolved" else ""}})
    c = case()
    c["annotation_permissions"] = {"train": {"status": ann_train, "evidence": "e"}}
    return workspace(tmp_path, sources=[s], cases=[c], annotations=[human("a1"), human("a2", who="h2")])


@pytest.mark.parametrize("train,ann", [("unresolved", "permitted"), ("prohibited", "permitted"), ("permitted", "unresolved"), ("permitted", "prohibited")])
def test_training_export_refuses_unresolved_or_prohibited_rights(tmp_path, train, ann):
    with pytest.raises(dq.DataQualityError, match="rights blocks"):
        dq.export_training(_full_workspace(tmp_path, train, ann), "out", uncertain_policy="exclude_from_binary_export")


def test_training_export_with_permitted_rights_and_declared_uncertain_policy(tmp_path):
    ws = _full_workspace(tmp_path)
    manifest = dq.export_training(ws, "out", uncertain_policy="exclude_from_binary_export")
    assert manifest["exported"] == 1 and manifest["ai_labels_exported"] == 0
    with pytest.raises(dq.DataQualityError, match="declared"):
        dq.export_training(_full_workspace(tmp_path / "x"), "out2", uncertain_policy="")


# ------------------------------------------------------------------------------------------------ 3 provenance
def test_ai_annotations_are_never_exported_as_human_labels(tmp_path):
    ws = workspace(tmp_path, sources=[source()], cases=[case()], annotations=[ai("a1"), ai("a2")])
    assert dq.quality_report(ws)["case_status"] == {"incomplete": 1}  # two AI votes are not two humans
    with pytest.raises(dq.DataQualityError, match="incomplete"):
        dq.export_training(ws, "out", uncertain_policy="exclude_from_binary_export")


def test_human_annotation_without_a_collection_process_and_ai_without_model_identity_are_rejected(tmp_path):
    bad_h = human("a1")
    del bad_h["collection_process_id"]
    bad_ai = ai("a2")
    del bad_ai["model_identity"]
    ws = workspace(tmp_path, sources=[source()], cases=[case()], annotations=[bad_h, bad_ai])
    reasons = " ".join(r for rej in ws.read_jsonl("rejected.jsonl") for r in rej["reasons"])
    assert "collection_process_id" in reasons and "model_identity" in reasons and ws.read_jsonl("annotations.jsonl") == []


# ------------------------------------------------------------------------------------------------ 4 overlap and groups
def test_exact_and_near_duplicate_overlap_is_flagged_and_limits_are_stated(tmp_path):
    index = ReferenceIndex()
    index.add("protected-doc", SRC_TEXT + " Extra closing sentence for the original document that is reused.")
    ws = workspace(tmp_path, sources=[source("s1", text=SRC_TEXT + " Extra closing sentence for the original document that is reused."),
                                      source("s2", "fam-b", text="A completely unrelated passage about harbour dredging schedules in spring and autumn weather.")],
                   cases=[])
    report = dq.overlap_report(ws, index)
    assert [f["source_id"] for f in report["flagged"]] == ["s1"] and report["flagged"][0]["kind"] == "exact" and "semantic" in report["limits"]


def test_document_family_must_stay_in_one_split(tmp_path):
    sources = [source("s1", "fam-a"), source("s1b", "fam-a", text=SRC_TEXT + " v2"), source("s2", "fam-b", text="Other text for family b only.")]
    assert dq.check_split_groups(sources, {"fam-a": "train", "fam-b": "dev"}) == {"leaked_families": [], "sources_without_assignment": []}
    srcs = {s["source_id"]: s for s in sources}
    cases = [case("c1", "s1"), case("c2", "s1b", text=SRC_TEXT + " v2")]
    assert dq.check_case_split_consistency(cases, srcs, {"c1": "train", "c2": "dev"})["leaked_families"] == ["fam-a"]
    assert dq.check_split_groups(sources, {"fam-a": "train"})["sources_without_assignment"] == ["s2"]


# ------------------------------------------------------------------------------------------------ 5 malformed records
def test_missing_duplicate_malformed_and_altered_records_are_kept_with_reasons(tmp_path):
    altered = source("s2", "fam-b", text="Original text.")
    altered["text"] = "Edited text."  # hash no longer matches
    ws = workspace(
        tmp_path,
        sources=[source("s1"), altered, "{not json", source("s3", "fam-c", text="Third text.", provenance="invented")],
        cases=[case("c1"), case("c1"), case("c9", "missing-source"), case("c2", "s1", cand=None), {**case("c3"), "source_sha256": "0" * 64}],
        annotations=[human("a1"), human("a1", who="h2"), human("a3", outcome="maybe"), human("a4", cid="nope")])
    rej = ws.read_jsonl("rejected.jsonl")
    reasons = " | ".join(r for x in rej for r in x["reasons"])
    for needle in ("altered source", "malformed line", "provenance", "duplicate case_id", "not in accepted sources", "missing candidate", "does not match the accepted source",
                   "duplicate annotation_id", "outcome must be one of", "not in accepted cases"):
        assert needle in reasons, needle
    report = dq.quality_report(ws)
    inv = report["inventory"]
    assert inv["rejected_rows"] == len(rej) and inv["sources"] + sum(1 for r in rej if r["kind"] == "source") == 4  # nothing vanished from the denominator


# ------------------------------------------------------------------------------------------------ 6 uncertainty and adjudication
def test_uncertain_and_disagreement_are_retained_with_full_inventory_counts(tmp_path):
    ws = workspace(
        tmp_path, sources=[source()],
        cases=[case("c1"), case("c2", cand="The mayor thanked staff."), case("c3", cand="The council met on Monday."), case("c4", cand="The clerk resigned.")],
        annotations=[human("a1"), human("a2", who="h2", outcome="unsupported"),                      # c1 disagreement, adjudicated
                     human("a3", "c2"), human("a4", "c2", who="h2", outcome="uncertain"),             # c2 disagreement, NOT adjudicated
                     human("a5", "c3", outcome="uncertain"), human("a6", "c3", who="h2", outcome="uncertain"),  # c3 agreed uncertain
                     human("a7", "c4")],                                                              # c4 only one annotator
        adjudications=[{"adjudication_id": "j1", "case_id": "c1", "annotation_ids": ["a1", "a2"], "reviewer_id": "h3", "reviewer_kind": "human",
                        "final_outcome": "unsupported", "reason": "synthetic"}])
    r = dq.quality_report(ws)
    assert r["inventory"]["cases"] == 4 and sum(r["case_status"].values()) == 4
    assert r["case_status"] == {"complete": 2, "disagreement_unadjudicated": 1, "incomplete": 1}
    assert r["uncertain_retained"] == 3 and r["disagreements"] == 2 and r["adjudicated"] == 1
    assert r["final_outcomes"]["uncertain"] == 1  # agreed-uncertain case c3 stays uncertain, not mapped to pass/fail


def test_adjudicator_must_differ_from_the_annotators(tmp_path):
    ws = workspace(tmp_path, sources=[source()], cases=[case()], annotations=[human("a1"), human("a2", who="h2", outcome="unsupported")],
                   adjudications=[{"adjudication_id": "j1", "case_id": "c1", "annotation_ids": ["a1", "a2"], "reviewer_id": "h1", "reviewer_kind": "human",
                                   "final_outcome": "supported", "reason": "x"}])
    assert ws.read_jsonl("adjudications.jsonl") == [] and "adjudicator must differ" in json.dumps(ws.read_jsonl("rejected.jsonl"))


def test_assertion_outcomes_must_agree_with_the_case_outcome_and_spans_must_be_exact(tmp_path):
    good = {"candidate_span": {"start": 4, "end": 9, "text": "clerk"}, "evidence": [{"start": 31, "end": 36, "text": "clerk"}], "outcome": "supported"}
    bad_span = {**good, "candidate_span": {"start": 4, "end": 9, "text": "mayor"}}
    ws = workspace(tmp_path, sources=[source()], cases=[case()],
                   annotations=[human("a1", assertions=[good]), human("a2", who="h2", assertions=[{**good, "outcome": "unsupported"}]),
                                human("a3", who="h3", assertions=[bad_span])])
    reasons = " ".join(r for x in ws.read_jsonl("rejected.jsonl") for r in x["reasons"])
    assert "inconsistent" in reasons and "offsets do not match" in reasons and [a["annotation_id"] for a in ws.read_jsonl("annotations.jsonl")] == ["a1"]


# ------------------------------------------------------------------------------------------------ 7 spans versus support
def test_authentic_quotes_do_not_make_an_unsupported_claim_supported():
    fx = next(f for f in cr.synthetic_fixtures() if f["case_id"] == "syn-11")  # attribute transfer; both quotes exist in the source
    rec = cr.judge_case({"case_id": "syn-11", "source": fx["source"], "candidate": fx["candidate"]}, lambda p: cr.Response(cr.scripted_response(fx)))
    a = rec["structured"]["assertions"][0]
    assert all(a["evidence_authentic"]) and rec["structured"]["verdict"] == "unsupported"


def test_unsupported_addition_needs_no_contradicting_quote():
    fx = next(f for f in cr.synthetic_fixtures() if f["case_id"] == "syn-15")
    rec = cr.judge_case({"case_id": "x", "source": fx["source"], "candidate": fx["candidate"]}, lambda p: cr.Response(cr.scripted_response(fx)))
    assert rec["structured"]["verdict"] == "unsupported" and rec["structured"]["assertions"][1]["evidence"] == []


def test_a_supported_fragment_of_a_multi_claim_sentence_is_not_a_complete_review():
    fx16 = next(f for f in cr.synthetic_fixtures() if f["case_id"] == "syn-16")
    rec = cr.judge_case({"case_id": "x", "source": fx16["source"], "candidate": fx16["candidate"]}, lambda p: cr.Response(cr.scripted_response(fx16)))
    s = rec["structured"]
    assert s["verdict"] == "uncertain" and "unreviewed_substantive_words" in s["flags"] and "clerk" in s["coverage"]["uncovered_words"]
    fx17 = next(f for f in cr.synthetic_fixtures() if f["case_id"] == "syn-17")  # one broad span hides a second claim
    rec = cr.judge_case({"case_id": "x", "source": fx17["source"], "candidate": fx17["candidate"]}, lambda p: cr.Response(cr.scripted_response(fx17)))
    assert rec["structured"]["verdict"] == "uncertain" and "possible_broad_span_hiding_second_claim" in rec["structured"]["flags"]


def test_support_without_authentic_evidence_is_not_accepted():
    fx = next(f for f in cr.synthetic_fixtures() if f["case_id"] == "syn-18")
    rec = cr.judge_case({"case_id": "x", "source": fx["source"], "candidate": fx["candidate"]}, lambda p: cr.Response(cr.scripted_response(fx)))
    assert rec["structured"]["verdict"] == "uncertain" and "supported_without_authentic_evidence" in rec["structured"]["flags"]


def test_wrong_span_offsets_are_repaired_only_when_the_text_is_unique_and_flagged():
    fx = next(f for f in cr.synthetic_fixtures() if f["case_id"] == "syn-12")
    raw = json.loads(cr.scripted_response(fx))
    raw["assertions"][0]["candidate_span"]["start"] += 3
    rec = cr.judge_case({"case_id": "x", "source": fx["source"], "candidate": fx["candidate"]}, lambda p: cr.Response(json.dumps(raw)))
    assert "candidate_span_offsets_repaired" in rec["structured"]["assertions"][0]["flags"] and rec["structured"]["verdict"] == "supported"
    raw["assertions"][0]["candidate_span"]["text"] = "text that is not in the candidate"
    rec = cr.judge_case({"case_id": "x", "source": fx["source"], "candidate": fx["candidate"]}, lambda p: cr.Response(json.dumps(raw)))
    assert rec["structured"]["verdict"] == "uncertain" and rec["structured"]["coverage"]["invalid_candidate_spans"] == 1


@pytest.mark.parametrize("mutate", [lambda r: r["assertions"][0].pop("actor"), lambda r: r["assertions"][0].update(modality="certain"),
                                    lambda r: r["assertions"][0].update(verdict="maybe"), lambda r: r["assertions"][0].update(relation={"type": "magic"}),
                                    lambda r: r.update(assertions=[]), lambda r: r["assertions"][0].update(rationale=5)])
def test_malformed_structured_responses_are_recorded_failures_not_guesses(mutate):
    fx = next(f for f in cr.synthetic_fixtures() if f["case_id"] == "syn-12")
    raw = json.loads(cr.scripted_response(fx))
    mutate(raw)
    rec = cr.judge_case({"case_id": "x", "source": fx["source"], "candidate": fx["candidate"]}, lambda p: cr.Response(json.dumps(raw)))
    assert rec["status"] == "invalid_structure" and rec["structured"] is None and rec["raw_response"] is not None


# ------------------------------------------------------------------------------------------------ 8 failed/partial runs
def _records_for(tmp_path, fail_ids=()):
    fixtures = cr.synthetic_fixtures()
    by_prompt = {cr.sha256(cr.build_prompt(f["source"], f["candidate"])): f for f in fixtures}

    def request(prompt):
        f = by_prompt[cr.sha256(prompt)]
        if f["case_id"] in fail_ids:
            raise RuntimeError("provider down")
        return cr.Response(cr.scripted_response(f), usage={"prompt_tokens": 10, "completion_tokens": 5}, model="MOCK")

    cases = [{"case_id": f["case_id"], "source": f["source"], "candidate": f["candidate"]} for f in fixtures]
    terminal = cr.run_cases(cases, request, tmp_path / "run")
    recs = [json.loads(x) for x in (tmp_path / "run" / "judgments.jsonl").read_text(encoding="utf8").splitlines()]
    return fixtures, terminal, recs


def test_runs_do_not_overwrite_and_records_keep_inputs_raw_responses_usage_and_model(tmp_path):
    _, _, recs = _records_for(tmp_path)
    r = recs[0]
    assert r["raw_response"] and r["usage"] == {"prompt_tokens": 10, "completion_tokens": 5} and r["model"] == "MOCK" and r["prompt_sha256"] == cr.sha256(r["prompt"])
    assert r["input"]["source_sha256"] == cr.sha256(r["input"]["source"])
    with pytest.raises(cr.ClaimRelationError, match="refusing to overwrite"):
        cr.run_cases([{"case_id": "x", "source": "s", "candidate": "c"}], lambda p: cr.Response("{}"), tmp_path / "run")
    unknown = cr.judge_case({"case_id": "x", "source": "S one.", "candidate": "S one."}, lambda p: cr.Response("not json"))
    assert unknown["usage"] is None and unknown["model"] is None and unknown["status"] == "invalid_structure"  # unknown stays null


# ------------------------------------------------------------------------------------------------ 9 identity and protection
def test_frozen_v18_identity_and_the_candidate_variant_are_distinct():
    ident = cr.check_v18_identity()
    assert ident["unchanged"] and ident["v18_config_hash"] == "12fc8b550d954359" and ident["candidate_variant"] == "claim-relation-v1"
    assert ident["candidate_content_hash"] != ident["v18_config_hash"] and len(cr.content_hash()) == 64


def test_original_inputs_remain_byte_identical_to_the_frozen_protocol():
    protocol = Path(dq.ROOT) / ".local" / "audit2-v3" / "protocol.json"
    if not protocol.exists():
        pytest.skip("frozen protocol not present")
    from evalops import audit2_v3 as a3
    frozen = json.loads(protocol.read_text())["hashes"]
    current = a3.code_and_data_hashes()
    for key in ("data/cases.jsonl", "data/annotations.jsonl", "data/splits.json", "data/judgment_cache.jsonl"):
        assert current[key] == frozen[key], key


@pytest.mark.parametrize("target", [Path(dq.__file__).parent / "data" / "new_dev", Path(dq.__file__).parent / "data", dq.ROOT / ".local" / "fresh-eval-v1" / "x"])
def test_workspaces_inside_protected_locations_are_refused(target):
    with pytest.raises(dq.DataQualityError, match="protected"):
        dq.Workspace(target)


def test_workspace_writes_cannot_escape_and_do_not_overwrite(tmp_path):
    ws = dq.Workspace(tmp_path / "w")
    ws.write_json("a.json", {})
    with pytest.raises(dq.DataQualityError, match="refusing to overwrite"):
        ws.write_json("a.json", {})
    with pytest.raises(dq.DataQualityError, match="escapes"):
        ws.write_json("../outside.json", {})


def test_protected_index_cannot_be_written_into_protected_folders():
    with pytest.raises(dq.DataQualityError):
        dq.build_protected_index(Path(dq.__file__).parent / "data" / "index.json")


def test_end_to_end_cli_with_mocked_inference_on_synthetic_fixtures(tmp_path, capsys):
    assert cr.main(["demo", "--out", str(tmp_path / "demo")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["terminal"]["status"] == "COMPLETE" and out["scripted_expectations_met"] == "19/19" and out["identity"]["unchanged"] and "MOCK" not in json.dumps(out["terminal"])
    inc = write_incoming(tmp_path, sources=[source()], cases=[case()], annotations=[human("a1"), human("a2", who="h2")])
    root = tmp_path / "dqws"
    assert dq.main(["init", "--root", str(root)]) == 0
    assert dq.main(["import", "--root", str(root), "--incoming", str(inc)]) == 0
    assert dq.main(["report", "--root", str(root)]) == 0
    assert dq.main(["packets", "--root", str(root), "--salt", "s"]) == 0
    assert dq.main(["export-training", "--root", str(root), "--out", "exp", "--uncertain-policy", "exclude_from_binary_export"]) == 0
    assert (root / "exp.jsonl").exists() and "AI" not in (root / "exp.manifest.json").read_text()


def test_build_cases_is_label_free_excludes_the_sealed_test_and_writes_once(tmp_path):
    info = cr.build_cases(["train", "dev"], tmp_path / "in")
    assert info["cases"] == 257
    cases = json.loads((tmp_path / "in" / "cases.json").read_text())
    assert all(set(c) == {"case_id", "source", "candidate"} for c in cases)
    from evalops.multi_judge import load_cases
    assert {c["case_id"] for c in cases} == {c.case_id for c, _y, _s in load_cases(["train", "dev"])}
    assert len(json.loads((tmp_path / "in" / "labels.json").read_text())) == 257
    with pytest.raises(cr.ClaimRelationError, match="sealed"):
        cr.build_cases(["test"], tmp_path / "in2")
    with pytest.raises(cr.ClaimRelationError, match="refusing to overwrite"):
        cr.build_cases(["train", "dev"], tmp_path / "in")



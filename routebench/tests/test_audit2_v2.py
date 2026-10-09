import json

import pytest
from evalops import audit2_v2 as a

SRC = "Mayor Lee said on Monday that the council will vote on the budget next week. She hoped repairs would start in spring."
SUPPORTED = json.dumps({"verdict": "supported"})


def item(i, label=1, v18=1):
    return {"id": f"c{i}", "source": SRC, "candidate": "The council will vote next week.", "label": label, "v18": v18}


def frozen(tmp_path, items):
    """A frozen work dir for synthetic items (code/data hashes are the real ones)."""
    protocol = {"protocol": "audit2-v2", "hashes": a.code_and_data_hashes(), "cap_usd_total": a.TOTAL_CAP_USD,
                "stages": {"train": len(items), "dev": len(items)}, "population_digest": {s: a.population_digest(items) for s in ("train", "dev")}}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf8")
    return tmp_path


def bound():
    return a.worst_case_usd(a.build_prompt(SRC, "The council will vote next week."))


def test_cost_above_reservation_aborts_and_is_not_complete(tmp_path):  # the $0.025-cap reproduction
    items = [item(0), item(1)]
    out = a.execute_stage(frozen(tmp_path, items), "train", items, lambda p: (SUPPORTED, 0.10), cap_usd=0.025)
    assert out["status"].startswith("ABORTED")
    assert out["spent_usd_total"] >= 0.10 and out["report"]["status"] == "WITHHELD_INCOMPLETE"


def test_cap_is_checked_against_the_journal_and_survives_a_crash(tmp_path):
    items = [item(0)]
    work = frozen(tmp_path, items)
    journal = a.Journal(work / "spend_journal.jsonl", cap_usd=bound() * 1.5)
    journal.reserve("crashed", 1, bound())  # process died before settle: still counted at its bound
    assert journal.committed() == pytest.approx(bound())
    out = a.execute_stage(work, "train", items, lambda p: (SUPPORTED, 0.001), cap_usd=bound() * 1.5)
    assert out["status"].startswith("ABORTED") and out["audited"] == 0  # a second bound does not fit under the cap


@pytest.mark.parametrize("cost", [float("nan"), -0.01, float("inf"), "0.01", True])
def test_invalid_costs_stop_the_run(tmp_path, cost):
    items = [item(0)]
    out = a.execute_stage(frozen(tmp_path, items), "train", items, lambda p: (SUPPORTED, cost))
    assert out["status"].startswith("ABORTED")


def test_missing_cost_is_charged_at_the_bound(tmp_path):
    items = [item(0)]
    out = a.execute_stage(frozen(tmp_path, items), "train", items, lambda p: (SUPPORTED, None))
    assert out["status"] == "COMPLETE" and out["spent_usd_total"] == pytest.approx(bound(), rel=1e-3)


def test_every_provider_attempt_including_failures_is_journaled(tmp_path):
    items = [item(0)]
    work = frozen(tmp_path, items)
    a.execute_stage(work, "train", items, lambda p: ("not json", 0.002))
    entries = [json.loads(x) for x in (work / "spend_journal.jsonl").read_text().splitlines()]
    assert [e["op"] for e in entries] == ["reserve", "settle", "reserve", "settle"]  # MAX_ATTEMPTS=2, both billed


def test_changed_inputs_or_wrong_hash_stop_before_any_provider_call(tmp_path):
    items = [item(0), item(1)]
    work = frozen(tmp_path, items)
    calls = []

    def call(prompt):
        calls.append(prompt)
        return SUPPORTED, 0.001

    for changed in ([item(0), {**item(1), "candidate": "The council cancelled the vote."}], [item(0), {**item(1), "label": 0}],
                    [item(0), {**item(1), "v18": 0}], [item(0), {**item(1), "source": SRC + " Extra."}]):
        with pytest.raises(a.FreezeError, match="digest"):
            a.execute_stage(work, "train", changed, call)
    assert calls == []


def test_duplicate_missing_and_malformed_cases_are_rejected(tmp_path):
    work = frozen(tmp_path, [item(0), item(1)])
    for bad, msg in (([item(0), item(0)], "duplicate"), ([item(0)], "expected 2"), ([item(0), {**item(1), "label": 2}], "malformed")):
        with pytest.raises(a.FreezeError, match=msg):
            a.execute_stage(work, "train", bad, lambda p: (SUPPORTED, 0.001))


def test_code_or_data_change_invalidates_the_freeze(tmp_path):
    items = [item(0)]
    work = frozen(tmp_path, items)
    proto = json.loads((work / "protocol.json").read_text())
    proto["hashes"]["audit2.py"] = "0" * 64
    (work / "protocol.json").write_text(json.dumps(proto))
    with pytest.raises(a.FreezeError, match="changed since the freeze"):
        a.execute_stage(work, "train", items, lambda p: (SUPPORTED, 0.001))


@pytest.mark.parametrize("field", ["explanation", "unsupported_details", "claim_says", "error_type"])
def test_malformed_field_types_are_recorded_failures_not_crashes(tmp_path, field):
    bad = {"verdict": "unsupported", "error_type": "reasoning_error", "disputed_assertion": "x", "claim_says": "a", "source_says": "b",
           "evidence_quote": "q", "unsupported_details": [], "explanation": "fine words here"}
    bad[field] = [1, 2] if field == "unsupported_details" else {"nested": 1}
    items = [item(0)]
    out = a.execute_stage(frozen(tmp_path, items), "train", items, lambda p: (json.dumps(bad), 0.001))
    assert out["status"] == "INCOMPLETE: failed audits" and out["report"]["status"] == "WITHHELD_INCOMPLETE"
    assert json.loads((tmp_path / "rows_train.jsonl").read_text().splitlines()[0])["status"] == "failed"


def test_partial_run_withholds_the_comparison(tmp_path):
    items = [item(i) for i in range(3)]
    out = a.execute_stage(frozen(tmp_path, items), "train", items, lambda p: (SUPPORTED, bound() * 0.9), cap_usd=bound() * 2.5)
    assert out["status"].startswith("ABORTED") and out["report"]["status"] == "WITHHELD_INCOMPLETE" and "kappa_audited" not in out["report"]
    assert a.score_population(items, {})["status"] == "WITHHELD_INCOMPLETE"


def test_resume_continues_without_repeating_audited_cases(tmp_path):
    items = [item(i) for i in range(3)]
    work = frozen(tmp_path, items)
    first = a.execute_stage(work, "train", items, lambda p: (SUPPORTED, bound() * 0.9), cap_usd=bound() * 2.5)
    assert first["audited"] == 2
    (work / "terminal_train.json").unlink()  # an operator reviews the ABORTED terminal before resuming
    seen = []
    second = a.execute_stage(work, "train", items, lambda p: (seen.append(1), (SUPPORTED, 0.001))[1], cap_usd=3.0, resume=True)
    assert second["status"] == "COMPLETE" and len(seen) == 1


def test_undefined_kappa_is_none_not_one():
    assert a.kappa([1, 1, 1], [1, 1, 1]) is None
    assert a.kappa([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0


def test_dev_requires_a_complete_train_stage(tmp_path):
    items = [item(0)]
    work = frozen(tmp_path, items)
    (work / "terminal_train.json").write_text(json.dumps({"status": "INCOMPLETE: failed audits"}))
    with pytest.raises(a.FreezeError, match="COMPLETE train"):
        a.execute_stage(work, "dev", items, lambda p: (SUPPORTED, 0.001))


def test_prepare_freezes_the_real_population_and_matches_it_on_verify(tmp_path):
    body = a.prepare(tmp_path / "w")
    assert body["stages"] == {"train": 94, "dev": 88}
    pop = json.loads((tmp_path / "w" / "population.json").read_text())
    for stage in ("train", "dev"):
        assert a.verify_freeze(tmp_path / "w", stage, pop[stage])["population_digest"][stage] == a.population_digest(pop[stage])

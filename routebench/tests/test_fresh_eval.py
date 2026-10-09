"""Offline tests for the frozen fresh-document evaluation: mocked providers only, no network, no spend."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest
from evalops import audit2 as a1
from evalops import fresh_eval as fe
from evalops import rubrics

CRITERIA = sorted(rubrics.for_task("summarize").criterion_ids)


def v18_reply(passed: bool) -> str:
    return json.dumps({"criteria": [{"id": c, "verdict": "yes" if passed else "no", "evidence": ""} for c in CRITERIA], "rationale": "x"})


def audit_reply(unsupported: bool, **kw) -> str:
    if not unsupported:
        return json.dumps({"verdict": "supported"})
    base = {"verdict": "unsupported", "disputed_assertion": "__CLAIM__", "error_type": "reasoning_error", "claim_says": "a causal link",
            "source_says": "nothing of the sort", "evidence_quote": None, "unsupported_details": [],
            "explanation": "The summary draws a conclusion that the source never states anywhere."}
    base.update(kw)
    return json.dumps(base)


USAGE = {"prompt_tokens": 900, "completion_tokens": 600}


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    work = tmp_path_factory.mktemp("fresh") / "w"
    fe.prepare(work)
    return work


def fresh_copy(frozen, tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    for f in frozen.iterdir():
        if f.name in {"protocol.json", "items.json", "labels.json", "manifest.json"}:
            (work / f.name).write_bytes(f.read_bytes())
    return work


def make_calls(items, labels, *, pass_ids, flip_ids, cost_usage=USAGE):
    by_claim = {i["claim"]: i for i in items}

    def call_v18(messages, max_tokens):
        text = messages[0]["content"]
        item = next(i for i in items if "<<<BEGIN_CANDIDATE\n" + i["claim"] + "\nEND_CANDIDATE>>>" in text)
        return v18_reply(item["id"] in pass_ids), cost_usage, "o4-mini-2025-04-16"

    def call_audit(messages, max_tokens):
        text = messages[0]["content"]
        item = next(i for i in items if f"<summary>\n{i['claim']}\n</summary>" in text)
        reply = audit_reply(item["id"] in flip_ids).replace("__CLAIM__", by_claim[item["claim"]]["claim"].replace('"', "'"))
        return reply, cost_usage, "o4-mini-2025-04-16"
    return call_v18, call_audit


# ------------------------------------------------------------------------------------------------------- sampling
def test_sample_is_240_cases_8_per_document_from_the_30_reserved_documents(frozen):
    manifest = json.loads((frozen / "manifest.json").read_text())
    first = json.loads(fe.FIRST_SAMPLE_MANIFEST.read_text())
    taken = {d for v in first["sources"].values() for d in v["sampled_docs"]}
    docs = {r["doc_id"] for r in manifest["rows"]}
    assert len(manifest["rows"]) == 240 and len(docs) == 30 and not docs & taken
    per_doc = {d: sum(r["doc_id"] == d for r in manifest["rows"]) for d in docs}
    assert set(per_doc.values()) == {8}
    assert all(0 < r["inclusion_prob"] <= 1 for r in manifest["rows"])


def test_sampling_does_not_depend_on_labels(monkeypatch):
    real = csv.DictReader

    def shuffled(*args, **kwargs):
        rows = list(real(*args, **kwargs))
        rng = random.Random(7)
        for r in rows:
            if "sent_label" in r:
                r["sent_label"] = rng.choice(["yes", "no"])
        return iter(rows)

    baseline = fe.sample_manifest()
    monkeypatch.setattr(fe.csv, "DictReader", shuffled)
    assert fe.sample_manifest() == baseline


def test_judge_inputs_carry_no_label(frozen):
    items = json.loads((frozen / "items.json").read_text())
    assert all(set(i) == {"id", "source", "doc_id", "row_index", "text", "claim", "weight"} for i in items)
    assert len(json.loads((frozen / "labels.json").read_text())) == 240


def test_the_run_never_reads_the_labels_file(frozen, tmp_path, monkeypatch):
    work = fresh_copy(frozen, tmp_path)
    items = json.loads((work / "items.json").read_text())
    real = Path.read_text

    def guard(self, *a, **k):
        if self.name == "labels.json":
            raise AssertionError("labels read during the run")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", guard)
    call_v18, call_audit = make_calls(items, None, pass_ids=set(), flip_ids=set())
    out = fe.run_all(work, call_v18, call_audit, sleep=lambda s: None)
    assert out["status"] == "COMPLETE" and out["v18_passes"] == 0


# --------------------------------------------------------------------------------------------------------- freeze
def test_changed_items_or_code_hash_stop_the_run_before_any_call(frozen, tmp_path):
    work = fresh_copy(frozen, tmp_path)
    items = json.loads((work / "items.json").read_text())
    items[0]["claim"] += " changed"
    (work / "items.json").write_text(json.dumps(items))
    calls = []
    with pytest.raises(fe.FreezeError, match="frozen set"):
        fe.run_all(work, lambda *a: calls.append(1), lambda *a: calls.append(1))
    work2 = fresh_copy(frozen, tmp_path / "x") if (tmp_path / "x").mkdir() is None else None
    proto = json.loads((work2 / "protocol.json").read_text())
    proto["hashes"]["fresh_eval.py"] = "0" * 64
    (work2 / "protocol.json").write_text(json.dumps(proto))
    with pytest.raises(fe.FreezeError, match="changed since the freeze"):
        fe.run_all(work2, lambda *a: calls.append(1), lambda *a: calls.append(1))
    assert calls == []


def test_protocol_binds_prompt_rule_config_and_cap(frozen):
    proto = json.loads((frozen / "protocol.json").read_text())
    h = proto["hashes"]
    assert h["v18_config_hash"] == fe.V18_HASH and h["auditor_prompt"] == "f52f7800a9c163dd8ad386a9d529f6da29a475f00e4992d068421590ea612f4c"
    assert {"fresh_eval.py", "judge.py", "audit2.py", "rubric_bundle", "aisys/llm.py", "aisys/pricing.yaml"} <= set(h)
    assert proto["cap_usd"] == 4.60 and proto["n_cases"] == 240 and proto["n_docs"] == 30


# ------------------------------------------------------------------------------------------------------- decision rule
def test_final_rule_flips_on_the_auditors_own_verdict_without_quote_or_overlap_gates():
    cand = "Rudy Giuliani admits pushing out Ambassador Yovanovitch to pursue personal interests."
    result = a1.AuditResult("unsupported", "pushing out Ambassador Yovanovitch to pursue personal interests", "unsupported_addition", "motive stated",
                            "nothing", None, ("personal interests",), "The source never says why the ambassador left her post or what motive he had.")
    assert fe.decide_final(result, "source", cand) == (True, "flip_final_rule")


@pytest.mark.parametrize("change,reason", [
    ({"verdict": "supported"}, "auditor_supported"),
    ({"disputed_assertion": "something never in the summary"}, "assertion_not_in_candidate"),
    ({"explanation": "Wrong."}, "explanation_too_short"),
    ({"claim_says": None}, "claim_source_statements_missing_or_equal"),
    ({"claim_says": "Voted.", "source_says": "voted"}, "claim_source_statements_missing_or_equal"),
    ({"error_type": "vibes"}, "unknown_error_type"),
])
def test_final_rule_blocks(change, reason):
    base = dict(verdict="unsupported", disputed_assertion="the council voted", error_type="tense_or_modality", claim_says="voted", source_says="will vote",  # noqa: C408
                evidence_quote=None, unsupported_details=(), explanation="The source only says the vote is planned for next week, not done.")
    base.update(change)
    assert fe.decide_final(a1.AuditResult(**base), "src", "The council voted on Monday.") == (False, reason)


def test_the_rule_reproduces_the_diagnostic_on_cached_train_and_dev_rows():
    root = Path(fe.ROOT) / ".local" / "audit2-v3"
    if not (root / "rows_dev.jsonl").exists():
        pytest.skip("cached audit rows not present")
    from evalops import audit2_v2 as a2
    from evalops import audit2_v3  # noqa: F401
    full = a1._population()
    out = {}
    for stage in ("train", "dev"):
        rows = {json.loads(x)["id"]: json.loads(x) for x in (root / f"rows_{stage}.jsonl").read_text().splitlines()}
        cand = {i["id"]: i for i in full[stage]}
        flips = set()
        for i, r in rows.items():
            res = a1.AuditResult(**{**r["result"], "unsupported_details": tuple(r["result"]["unsupported_details"])})
            if fe.decide_final(res, cand[i]["source"], cand[i]["candidate"])[0]:
                flips.add(i)
        gold, base = {i["id"]: i["label"] for i in full[stage]}, {i["id"]: i["v18"] for i in full[stage]}
        new = {i: 0 if i in flips else base[i] for i in gold}
        out[stage] = (sum(base[i] != gold[i] and new[i] == gold[i] for i in gold), sum(base[i] == gold[i] and new[i] != gold[i] for i in gold))
        assert a2.kappa(list(gold.values()), list(new.values())) is not None
    assert out == {"train": (1, 1), "dev": (3, 0)}  # = the raw-verdict diagnostic: the frozen rule keeps the four correct flips and the one wrong flip


# --------------------------------------------------------------------------------------------------------- metrics
def test_kappa_definition_and_weighting_are_separate():
    assert fe.kappa_counts(40, 10, 10, 40) == pytest.approx(0.6)
    assert fe.kappa_counts(10, 0, 0, 0) is None  # chance agreement total: undefined, not 1.0
    y, p = np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])
    unweighted, weighted = fe.kappa_counts(*fe.table(y, p, np.ones(4))), fe.kappa_counts(*fe.table(y, p, np.array([1.0, 5.0, 1.0, 1.0])))
    assert unweighted != weighted
    assert unweighted == pytest.approx(0.5) and weighted == pytest.approx(0.0909, abs=1e-3)


def test_paired_bootstrap_is_deterministic_and_identical_predictions_gain_nothing():
    rng = np.random.default_rng(0)
    n = 240
    y, base = rng.integers(0, 2, n), rng.integers(0, 2, n)
    w, src = np.ones(n), np.array(["A"] * 120 + ["B"] * 120)
    doc = np.array([f"d{i // 8}" for i in range(n)])
    one, two = fe.paired_bootstrap(y, base, base, w, src, doc), fe.paired_bootstrap(y, base, base, w, src, doc)
    assert one == two and one["gain_weighted_ci95"] == [0.0, 0.0] and one["draws_used"] > 0


# ------------------------------------------------------------------------------------------------------- full pipeline
def test_complete_run_reports_coverage_spend_corrections_and_outcomes(frozen, tmp_path):
    work = fresh_copy(frozen, tmp_path)
    items = json.loads((work / "items.json").read_text())
    labels = json.loads((work / "labels.json").read_text())
    pass_ids = {i["id"] for k, i in enumerate(items) if k % 4 != 0}
    flip_ids = {i for i in pass_ids if labels[i] == 0}  # an auditor that flips exactly the false passes
    call_v18, call_audit = make_calls(items, labels, pass_ids=pass_ids, flip_ids=flip_ids)
    terminal = fe.run_all(work, call_v18, call_audit, sleep=lambda s: None)
    assert terminal["status"] == "COMPLETE" and terminal["v18_ok"] == 240 and terminal["audits_ok"] == len(pass_ids)
    assert 0 < terminal["spent_usd_total"] <= fe.CAP_USD
    result = fe.report(work)
    base_fp = sum(1 for i in pass_ids if labels[i] == 0)
    assert result["status"] == "COMPLETE" and result["corrected"] == base_fp and result["introduced"] == 0
    assert result["pipeline_v18_plus_auditor"]["kappa_weighted"] >= result["baseline_v18"]["kappa_weighted"]
    assert set(result["outcomes"]) >= {"IMPROVED", "REACHED_0.74"} and result["paired_document_bootstrap"]["draws_used"] > 0
    with pytest.raises(fe.FreezeError, match="exists"):
        fe.report(work)


def test_run_is_once_only(frozen, tmp_path):
    work = fresh_copy(frozen, tmp_path)
    items = json.loads((work / "items.json").read_text())
    c18, cau = make_calls(items, None, pass_ids=set(), flip_ids=set())
    fe.run_all(work, c18, cau, sleep=lambda s: None)
    with pytest.raises(fe.FreezeError, match="run once"):
        fe.run_all(work, c18, cau, sleep=lambda s: None)


def test_shared_hard_cap_covers_both_judges_retries_and_failures(frozen, tmp_path):
    work = fresh_copy(frozen, tmp_path)
    items = json.loads((work / "items.json").read_text())
    expensive = {"prompt_tokens": 2000, "completion_tokens": 7900}  # ~ $0.037 per v18 call
    c18, cau = make_calls(items, None, pass_ids={i["id"] for i in items}, flip_ids=set(), cost_usage=expensive)
    terminal = fe.run_all(work, c18, cau, sleep=lambda s: None)
    assert terminal["status"].startswith("ABORTED") and terminal["spent_usd_total"] <= fe.CAP_USD + 1e-9
    assert terminal["v18_ok"] < 240
    result = fe.report(work)
    assert result["status"] == "WITHHELD_INCOMPLETE" and "baseline_v18" not in result


def test_failed_requests_are_retried_and_charged_and_a_failure_withholds_the_comparison(frozen, tmp_path):
    work = fresh_copy(frozen, tmp_path)
    items = json.loads((work / "items.json").read_text())
    c18, cau = make_calls(items, None, pass_ids=set(), flip_ids=set())
    seen = {"n": 0}

    def flaky(messages, max_tokens):
        seen["n"] += 1
        if seen["n"] <= 2:
            raise RuntimeError("HTTP 503")
        return c18(messages, max_tokens)

    terminal = fe.run_all(work, flaky, cau, sleep=lambda s: None)
    ops = [json.loads(x)["op"] for x in (work / "spend_journal.jsonl").read_text().splitlines()]
    assert terminal["status"] == "COMPLETE" and ops.count("reserve") == 242  # 2 failed requests + 240 successful, each journaled
    assert json.loads((work / "rows_v18.jsonl").read_text().splitlines()[0])["attempts"] == 3
    work2 = fresh_copy(frozen, tmp_path / "b") if (tmp_path / "b").mkdir() is None else None

    first_claim = items[0]["claim"]

    def broken(messages, max_tokens):
        if "<<<BEGIN_CANDIDATE" + chr(10) + first_claim + chr(10) + "END_CANDIDATE>>>" in messages[0]["content"]:
            raise RuntimeError("HTTP 503")  # one case fails every attempt; the rest succeed
        return c18(messages, max_tokens)

    out = fe.run_all(work2, broken, cau, sleep=lambda s: None)
    assert out["status"] == "INCOMPLETE: failed judgments" and fe.report(work2)["status"] == "WITHHELD_INCOMPLETE"


def test_real_client_path_is_one_http_request_per_reservation(frozen, tmp_path, monkeypatch):
    from aisys import llm
    work = fresh_copy(frozen, tmp_path)
    items = json.loads((work / "items.json").read_text())[:1]
    posts = []

    class Resp:
        status_code, headers = 200, {}

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

        def post(self, url, headers=None, json=None):
            posts.append(json)
            text = v18_reply(False)
            return Resp({"choices": [{"message": {"content": text}}], "model": "o4-mini-2025-04-16", "usage": USAGE})

    monkeypatch.setattr(llm.httpx, "Client", Client)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("llm.chat retries internally")))
    journal = fe.Journal(tmp_path / "j.jsonl", fe.CAP_USD)
    row = fe.judge_v18(items[0], fe.http_call("o4-mini", {"reasoning_effort": "medium"}), journal, lambda s: None)
    assert row["status"] == "ok" and len(posts) == 1 and posts[0]["max_completion_tokens"] == 8000 and posts[0]["reasoning_effort"] == "medium"
    assert [json.loads(x)["op"] for x in (tmp_path / "j.jsonl").read_text().splitlines()] == ["reserve", "settle"]


def test_v18_prompt_is_the_judge_modules_prompt_for_the_frozen_config(frozen):
    from evalops import judge
    from evalops.run_calibration import GRID
    cfg = next(c for c in GRID if c.variant_id == fe.V18)
    item = json.loads((frozen / "items.json").read_text())[0]
    case = fe.to_case(item)
    prompt = judge.build_prompt(case, rubrics.for_task("summarize", cfg.rubric_dir), cfg)
    assert item["claim"] in prompt and item["text"].strip() in prompt and cfg.config_hash == fe.V18_HASH

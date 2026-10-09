"""Offline tests for the matched v16/v18 comparison protocol. The HTTP layer is replaced by a fake: no provider call, no spend."""
import json
from pathlib import Path

import pytest
from aisys import llm
from evalops import judge, rubrics
from evalops import matched_v16_v18 as m
from evalops.audit2_v2 import SpendError

CRITERIA = sorted(rubrics.for_task("summarize").criterion_ids)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    work = tmp_path_factory.mktemp("matched") / "w"
    body = m.freeze(work)
    return work, body, m._sha(work / "protocol.json")


def copy_work(frozen, tmp_path):
    work, _b, sha = frozen
    dst = tmp_path / "w"
    dst.mkdir()
    for p in work.iterdir():
        if p.name in {"protocol.json", "cases.jsonl", "population.json", "labels.json"}:
            (dst / p.name).write_bytes(p.read_bytes())
    return dst, sha


class Fake:
    def __init__(self, pass_all=True, bad_for=None, quota_at=None, served="gpt-4.1-2025-04-14"):
        self.calls, self.pass_all, self.bad_for, self.quota_at, self.served = [], pass_all, bad_for or set(), quota_at, served

    def __call__(self, model, effort, temperature, max_tokens, messages):
        self.calls.append({"model": model, "effort": effort, "temperature": temperature, "max_tokens": max_tokens, "prompt": messages[0]["content"]})
        n = len(self.calls)
        if self.quota_at == n:
            raise llm.QuotaExceededError("HTTP 429: insufficient_quota")
        if model in self.bad_for:
            return "this is not json at all", {"prompt_tokens": 500, "completion_tokens": 50}, self.served
        verdict = "yes" if self.pass_all or n % 3 else "no"
        body = {"criteria": [{"id": c, "verdict": verdict, "evidence": ""} for c in CRITERIA], "rationale": "synthetic"}
        served = self.served if model == "gpt-4.1" else "o4-mini-2025-04-16"
        return json.dumps(body), {"prompt_tokens": 800, "completion_tokens": 200}, served


# ------------------------------------------------------------------------------------------------ provenance of the historical peak
def test_the_historical_v16_reveal_peak_is_reproduced_from_cached_judgments_and_labels():
    items, labels, cases = m.build_population()
    ids = [i["case_id"] for i in items if i["pop"] == "reveal"]
    scores = {c: m.verified_cached(cases[c], "v16-holistic-4.1", "reveal") for c in ids}
    usable = {c: s["score"] for c, s in scores.items() if s}
    assert len(ids) == 128 and len(usable) == 124
    y = [labels[c] for c in sorted(usable)]
    strict = [int(usable[c] >= 1.0) for c in sorted(usable)]
    historical = [int(usable[c] >= 0.30) for c in sorted(usable)]
    assert strict == historical  # the 0.30 threshold does no work: the strict all-criteria rule gives identical predictions
    assert m._kap(y, strict) == pytest.approx(0.838835456199636)
    assert (sum(a == 1 and b == 1 for a, b in zip(y, strict, strict=True)), sum(a == 0 and b == 1 for a, b in zip(y, strict, strict=True)),
            sum(a == 1 and b == 0 for a, b in zip(y, strict, strict=True)), sum(a == 0 and b == 0 for a, b in zip(y, strict, strict=True))) == (56, 3, 7, 58)
    assert all(s["model_served"] == "gpt-4.1-2025-04-14" for s in scores.values() if s)


def test_the_recipe_identity_is_pinned():
    cfg = m.cfg_of("v16-holistic-4.1")
    assert (cfg.config_hash, cfg.model, cfg.mode, cfg.max_tokens, cfg.temperature, cfg.samples, cfg.include_boundary_examples, cfg.exemplars_path) == (
        "51695bd659270b7f", "gpt-4.1", "rubric", 2000, 0.0, 1, True, "")
    assert rubrics.for_task("summarize").hash == "d72496fbb2991071" and judge.prompt_template_hash("rubric") == "ef895d791c0bd860"
    assert m.cfg_of("v18-reasoning-o4mini").config_hash == "12fc8b550d954359"


@pytest.mark.parametrize("mutate", [lambda j: setattr(j, "model_served", "gpt-4o"), lambda j: setattr(j, "config_hash", "0" * 16), lambda j: setattr(j, "rubric_version", "summarize@1.0.0+bad"),
                                    lambda j: setattr(j, "status", "provider_error"), lambda j: setattr(j, "variant_id", "other")])
def test_cached_judgments_that_do_not_match_the_current_identity_are_not_reused(monkeypatch, mutate):
    items, _labels, cases = m.build_population()
    it = next(i for i in items if i["pop"] == "dev")
    real = m.cache_for(it["pop"])
    j = real.get(m._cache_key(cases[it["case_id"]], m.cfg_of("v16-holistic-4.1")))
    assert j is not None and m.verified_cached(cases[it["case_id"]], "v16-holistic-4.1", "dev") is not None
    tampered = j.model_copy(deep=True)
    mutate(tampered)

    class Stub:
        def get(self, key):
            return tampered

    monkeypatch.setattr(m, "cache_for", lambda pop: Stub())
    assert m.verified_cached(cases[it["case_id"]], "v16-holistic-4.1", "dev") is None


# ------------------------------------------------------------------------------------------------ freeze
def test_the_freeze_fixes_populations_plan_rule_cap_and_binds_code(frozen):
    _w, b, _s = frozen
    assert b["populations"] == {"dev": 128, "train": 129, "reveal": 128, "wice": 128}
    assert b["subsets"] == {"AggreFact-CNN": 76, "Reveal": 128, "TofuEval-MediaS": 126, "TofuEval-MeetB": 55, "Wice": 128}
    assert b["new_calls"]["v16"] == 135 and b["new_calls"]["v18"] == 256 and b["cached_reuse"]["pairs"] == 635 and b["cap_usd"] == 6.0
    assert b["decision_rule"]["pass_iff_score_at_least"] == 1.0 and b["decision_rule"]["thresholds_searched"] is False
    assert b["sensitivity_not_primary"]["v16_threshold"] == 0.30 and set(b["code"]) >= {"matched_v16_v18.py", "judge.py", "rubric_bundle", "aisys/llm.py", "aisys/pricing.yaml"}
    assert b["judges"]["v16-holistic-4.1"]["model"] == "gpt-4.1" and b["judges"]["v18-reasoning-o4mini"]["model"] == "o4-mini"
    assert b["estimate"]["expected_usd_estimate"] < b["cap_usd"] and "estimates, not bounds" in b["estimate"]["note"]


def test_freeze_never_overwrites(frozen):
    with pytest.raises(m.MatchedError, match="never overwritten"):
        m.freeze(frozen[0])


@pytest.mark.parametrize("name", ["cases.jsonl", "population.json", "labels.json"])
def test_altered_inputs_after_the_freeze_are_refused_before_any_call(frozen, tmp_path, name):
    work, sha = copy_work(frozen, tmp_path)
    (work / name).write_text((work / name).read_text(encoding="utf8") + " ", encoding="utf8")
    fake = Fake()
    with pytest.raises(m.MatchedError):
        m.run(work, sha, fake)
    assert fake.calls == []


def test_a_wrong_protocol_hash_or_changed_code_is_refused(frozen, tmp_path, monkeypatch):
    work, sha = copy_work(frozen, tmp_path)
    with pytest.raises(m.MatchedError, match="freeze time"):
        m.run(work, "0" * 64, Fake())
    monkeypatch.setattr(m, "code_hashes", lambda: {"x": "y"})
    with pytest.raises(m.MatchedError, match="code"):
        m.run(work, sha, Fake())


# ------------------------------------------------------------------------------------------------ running
def test_requests_reproduce_the_registered_recipe_per_judge_and_only_missing_pairs_are_called(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    fake = Fake()
    t = m.run(work, sha, fake)
    assert t["status"] == "COMPLETE" and len(fake.calls) == 391 and t["spent_usd_journal"] <= 6.0
    v16 = [c for c in fake.calls if c["model"] == "gpt-4.1"]
    v18 = [c for c in fake.calls if c["model"] == "o4-mini"]
    assert len(v16) == 135 and len(v18) == 256
    assert {(c["max_tokens"], c["temperature"], c["effort"]) for c in v16} == {(2000, 0.0, "")}  # the historical recipe, not the 8,000-token variant
    assert {(c["max_tokens"], c["effort"]) for c in v18} == {(8000, "medium")}
    items, _l, cases = m.build_population()
    cfg = m.cfg_of("v16-holistic-4.1")
    sample = next(i for i in items if i["pop"] == "train")
    assert judge.build_prompt(cases[sample["case_id"]], rubrics.for_task("summarize"), cfg) in {c["prompt"] for c in v16}


def test_complete_run_compares_per_population_and_subset_with_the_same_strict_rule(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    m.run(work, sha, Fake())
    out = m.compare(work, sha)
    assert out["status"] == "COMPLETE" and set(out["populations"]) == {"dev", "train", "reveal", "wice"} and set(out["subsets"]) >= {"Reveal", "Wice", "AggreFact-CNN"}
    assert out["populations"]["dev"]["n"] == 128 and out["populations"]["reveal"]["n"] == 128 and out["populations"]["dev"]["difference_95_group_bootstrap"]
    assert out["decision_rule"]["thresholds_searched"] is False and "sensitivity_v16_at_0.30_not_primary" in out["populations"]["reveal"]
    assert out["original_groundedness_dev_plus_train_supplementary"]["n"] == 257 and "never pooled into a claim" in out["note"]
    with pytest.raises(m.MatchedError, match="written once"):
        m.compare(work, sha)


def test_restart_does_not_repeat_finished_pairs(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    first = Fake(quota_at=20)
    t1 = m.run(work, sha, first)
    assert t1["status"] == "ABORTED" and len(first.calls) == 20 and t1["new_ok"] == 19
    second = Fake()
    t2 = m.run(work, sha, second)
    assert t2["status"] == "COMPLETE" and len(second.calls) == 391 - 19 and t2["new_ok"] == 391


def test_quota_or_served_model_mismatch_stops_with_zero_further_calls(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    f = Fake(quota_at=1)
    assert m.run(work, sha, f)["status"] == "ABORTED" and len(f.calls) == 1
    work2, sha2 = copy_work(frozen, tmp_path / "b") if (tmp_path / "b").mkdir() is None else (None, None)
    g = Fake(served="gpt-4o-2024")
    t = m.run(work2, sha2, g)
    assert t["status"] == "ABORTED" and len(g.calls) == 1 and "not 'gpt-4.1'" in t["stopped_on"]


def test_failed_pairs_are_retained_not_dropped_and_withhold_everything_without_reading_labels(frozen, tmp_path, monkeypatch):
    work, sha = copy_work(frozen, tmp_path)
    t = m.run(work, sha, Fake(bad_for={"gpt-4.1"}))
    assert t["status"] == "INCOMPLETE" and len(t["failed"]) == 135
    real, touched = Path.read_text, []

    def guard(self, *a, **k):
        if self.name == "labels.json":
            touched.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", guard)
    out = m.compare(work, sha)
    assert out["status"] == "WITHHELD_INCOMPLETE" and out["expected_pairs"] == 2 * 513 and touched == [] and not (work / "comparison.json").exists()
    assert "labels were not read" in out["note"]


def test_the_run_never_reads_labels(frozen, tmp_path, monkeypatch):
    work, sha = copy_work(frozen, tmp_path)
    real, touched = Path.read_text, []

    def guard(self, *a, **k):
        if self.name == "labels.json":
            touched.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", guard)
    m.run(work, sha, Fake())
    assert touched == []


def test_cli_refuses_without_hash_or_confirmation(capsys):
    assert m.main(["run"]) == 2 and "--protocol-sha256" in capsys.readouterr().err
    assert m.main(["run", "--protocol-sha256", "0" * 64]) == 2 and "--confirm" in capsys.readouterr().err
    assert SpendError is not None

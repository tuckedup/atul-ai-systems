"""Offline tests for the source-integrity-v2 fresh-judgment evaluation. The HTTP layer is a fake: no provider call, no spend."""
import hashlib
import json
from pathlib import Path

import pytest
from aisys import llm
from evalops import rubrics
from evalops import srcint_eval as se

CRITERIA = sorted(rubrics.for_task("summarize").criterion_ids)
JOINS = ("labels.json", "baseline_cached_v18.json")


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    work = tmp_path_factory.mktemp("srcint") / "w"
    return work, se.freeze(work), se._sha(work / "protocol.json")


def copy_work(frozen, tmp_path):
    work, _b, sha = frozen
    dst = tmp_path / "w"
    dst.mkdir()
    for p in work.iterdir():
        if p.name in {"protocol.json", "cases.jsonl", "population.json", *JOINS}:
            (dst / p.name).write_bytes(p.read_bytes())
    return dst, sha


class Fake:
    def __init__(self, flip_every=0, quota_at=None, bad=False):
        self.calls, self.flip_every, self.quota_at, self.bad = [], flip_every, quota_at, bad

    def __call__(self, model, effort, temperature, max_tokens, messages):
        self.calls.append({"model": model, "effort": effort, "max_tokens": max_tokens, "prompt": messages[0]["content"]})
        n = len(self.calls)
        if self.quota_at == n:
            raise llm.QuotaExceededError("HTTP 429: insufficient_quota")
        if self.bad:
            return "not json", {"prompt_tokens": 500, "completion_tokens": 50}, "o4-mini-2025-04-16"
        ok = not (self.flip_every and n % self.flip_every == 0)
        body = {"criteria": [{"id": c, "verdict": "yes" if ok else "no", "evidence": ""} for c in CRITERIA], "rationale": "synthetic"}
        return json.dumps(body), {"prompt_tokens": 900, "completion_tokens": 300}, "o4-mini-2025-04-16"


def test_the_freeze_keeps_all_257_cases_and_records_what_changed(frozen):
    _w, b, _s = frozen
    p = b["population"]
    assert (p["cases"], p["dev"], p["train"], p["source_changed"], p["unchanged_noise_controls"], p["removed_cases"]) == (257, 128, 129, 63, 194, 0)
    assert p["changed_by_pop_subset"] == {"dev/TofuEval-MediaS": 47, "train/AggreFact-CNN": 1, "train/TofuEval-MediaS": 15}
    assert b["judge"]["config_hash"] == "12fc8b550d954359" and b["dataset"] == "routebench-srcint-v2" and b["cap_usd"] == 3.0
    assert b["estimate"]["expected_usd"] < b["cap_usd"] and "not a bound" in b["estimate"]["basis"]
    assert b["fresh_judgments"].startswith("all 257")


def test_repaired_cases_have_their_own_ids_and_never_reuse_original_ones(frozen):
    work, _b, _s = frozen
    items = json.loads((work / "population.json").read_text(encoding="utf8"))
    assert all(i["case_id"] != i["original_case_id"] for i in items)
    assert sum(i["source_sha256_repaired"] != i["source_sha256_original"] for i in items) == 63


def test_the_original_benchmark_files_recorded_in_the_repair_manifest_are_unchanged():
    manifest = json.loads((se.SRC / "manifest.json").read_text(encoding="utf8"))
    for path, digest in manifest["inputs"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_every_case_is_judged_fresh_in_an_isolated_store_with_the_frozen_recipe(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    f = Fake()
    t = se.run(work, sha, f)
    assert t["status"] == "COMPLETE" and len(f.calls) == 257 and t["spent_usd_journal"] <= 3.0
    assert {(c["model"], c["effort"], c["max_tokens"]) for c in f.calls} == {("o4-mini", "medium", 8000)}
    assert (work / "results.jsonl").exists() and not (work / "judgment_cache.jsonl").exists()
    items = json.loads((work / "population.json").read_text(encoding="utf8"))
    changed = next(i for i in items if i["source_changed"])
    cases = {json.loads(x)["case_id"]: json.loads(x) for x in (work / "cases.jsonl").read_text(encoding="utf8").splitlines()}
    assert any(cases[changed["case_id"]]["context"].strip()[:200] in c["prompt"] for c in f.calls)  # the repaired text is what the judge saw


def test_the_run_never_reads_labels_or_the_cached_baseline(frozen, tmp_path, monkeypatch):
    work, sha = copy_work(frozen, tmp_path)
    real, touched = Path.read_text, []

    def guard(self, *a, **k):
        if self.name in JOINS:
            touched.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", guard)
    se.run(work, sha, Fake())
    assert touched == []


def test_complete_run_compares_dev_first_with_a_noise_control(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    se.run(work, sha, Fake(flip_every=7))
    out = se.compare(work, sha)
    assert out["status"] == "COMPLETE" and out["dev_primary"]["n"] == 128 and out["train_diagnostic"]["n"] == 129 and out["removed_cases"] == 0
    assert out["dev_primary"]["kappa_difference_fresh_minus_cached_95_document_bootstrap"] and "not the original benchmark" in out["label"]
    nc = out["noise_control"]
    assert nc["changed_inputs"]["n"] == 63 and nc["unchanged_inputs"]["n"] == 194 and 0 <= nc["fisher_exact_two_sided_p"] <= 1
    assert set(out["dev_subsets"]) == {"AggreFact-CNN", "TofuEval-MediaS", "TofuEval-MeetB"}
    assert {"false_passes", "false_rejections", "accuracy", "kappa"} <= set(out["dev_primary"]["fresh_repaired_input"])
    with pytest.raises(se.SrcIntError, match="written once"):
        se.compare(work, sha)


def test_failed_cases_are_retained_not_dropped_and_withhold_everything_without_reading_labels(frozen, tmp_path, monkeypatch):
    work, sha = copy_work(frozen, tmp_path)
    t = se.run(work, sha, Fake(bad=True))
    assert t["status"] == "INCOMPLETE" and len(t["failed"]) == 257
    real, touched = Path.read_text, []

    def guard(self, *a, **k):
        if self.name in JOINS:
            touched.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", guard)
    out = se.compare(work, sha)
    assert out["status"] == "WITHHELD_INCOMPLETE" and out["expected"] == 257 and touched == [] and not (work / "comparison.json").exists()


def test_restart_does_not_repeat_finished_cases_and_quota_stops_at_once(frozen, tmp_path):
    work, sha = copy_work(frozen, tmp_path)
    first = Fake(quota_at=11)
    t1 = se.run(work, sha, first)
    assert t1["status"] == "ABORTED" and len(first.calls) == 11 and t1["ok"] == 10
    second = Fake()
    t2 = se.run(work, sha, second)
    assert t2["status"] == "COMPLETE" and len(second.calls) == 247


@pytest.mark.parametrize("name", ["cases.jsonl", "population.json", *JOINS])
def test_altered_inputs_after_the_freeze_are_refused_before_any_call(frozen, tmp_path, name):
    work, sha = copy_work(frozen, tmp_path)
    (work / name).write_text((work / name).read_text(encoding="utf8") + " ", encoding="utf8")
    f = Fake()
    with pytest.raises(se.SrcIntError):
        se.run(work, sha, f)
    assert f.calls == []


def test_wrong_hash_or_missing_confirmation_is_refused(frozen, tmp_path, capsys):
    work, _sha = copy_work(frozen, tmp_path)
    with pytest.raises(se.SrcIntError, match="freeze time"):
        se.run(work, "0" * 64, Fake())
    assert se.main(["run", "--protocol-sha256", "0" * 64]) == 2 and "--confirm" in capsys.readouterr().err


def test_fisher_exact_matches_known_values():
    assert se.fisher_two_sided(0, 10, 0, 10) == pytest.approx(1.0)
    assert se.fisher_two_sided(8, 2, 1, 9) == pytest.approx(0.0055, abs=0.0015)
    assert 0 < se.fisher_two_sided(3, 7, 1, 9) <= 1

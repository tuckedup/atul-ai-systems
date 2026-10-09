"""The MiniCheck equivalence-only check, exercised with fake checkers (no GPU, no model, no download)."""
from __future__ import annotations

import json
import math
import shutil

import pytest
from evalops import judge_v19_equivalence as eq
from evalops import judge_v19_minicheck as mc
from evalops import judge_v19_phase1 as jv


class FromCache:
    """A checker that reproduces the cached scores exactly (looked up by claim text) or perturbed by `offset`."""

    def __init__(self, work, offset=0.0):
        self.offset = offset
        self.table = {}
        cached = json.loads((work / "minicheck_cached_routebench.json").read_text(encoding="utf8"))
        from evalops.dataset import CaseRecord

        for line in (work / "cases_routebench.jsonl").read_text(encoding="utf8").splitlines():
            c = CaseRecord.model_validate_json(line)
            ref = cached[c.case_id]
            self.table[c.candidate_output] = (ref["full"], ref["sent"])
            for i, value in enumerate(ref["sent"]):
                self.table[f"{c.candidate_output}\0{i}"] = (value, None)

    def score(self, docs, claims):
        return [min(1.0, self.table[c][0] + self.offset) for c in claims]


@pytest.fixture(scope="module")
def frozen_copy(tmp_path_factory):
    src = jv.WORK
    if not (src / "protocol.json").exists():
        pytest.skip("frozen judge-v19 protocol not present")
    dst = tmp_path_factory.mktemp("eq") / "w"
    dst.mkdir()
    for n in ("protocol.json", *jv.INPUT_FILES):
        shutil.copy(src / n, dst / n)
    return dst, jv._sha(dst / "protocol.json")


def split_like_cached(work):
    """A splitter that returns exactly as many sentences as the cached features hold, so only the scores are compared."""
    from evalops.dataset import CaseRecord

    cached = json.loads((work / "minicheck_cached_routebench.json").read_text(encoding="utf8"))
    counts = {}
    for line in (work / "cases_routebench.jsonl").read_text(encoding="utf8").splitlines():
        c = CaseRecord.model_validate_json(line)
        counts[c.candidate_output] = len(cached[c.case_id]["sent"])

    def splitter(text):
        n = counts.get(text, 1)
        return [text] if n == 1 else [f"{text}\0{i}" for i in range(n)]

    return splitter


def test_a_checker_that_reproduces_the_cache_passes(frozen_copy):
    work, sha = frozen_copy
    out = eq.check(work, sha, FromCache(work), split_like_cached(work))
    assert out["passed"] and out["rows_checked"] == 24 and out["worst_abs_difference"] == pytest.approx(0, abs=1e-12) and not out["sentence_split_mismatches"]


def test_a_shifted_checker_fails_the_tolerance(frozen_copy):
    work, sha = frozen_copy
    out = eq.check(work, sha, FromCache(work, offset=0.05), split_like_cached(work))
    assert not out["passed"] and out["worst_abs_difference"] > out["tolerance"]


def test_a_different_sentence_split_fails_and_is_reported(frozen_copy):
    work, sha = frozen_copy
    out = eq.check(work, sha, FromCache(work), lambda t: [t, t, t, t, t, t, t])  # never matches the cached sentence counts
    assert not out["passed"] and out["sentence_split_mismatches"]


def test_run_needs_the_exact_confirmation_refuses_low_memory_and_writes_once(frozen_copy, tmp_path, monkeypatch):
    work, sha = frozen_copy
    dst = tmp_path / "w"
    shutil.copytree(work, dst)
    with pytest.raises(eq.EquivalenceError, match="confirm"):
        eq.run(dst, sha, "yes", checker=FromCache(dst))
    monkeypatch.setattr(eq, "free_ram_mb", lambda: 800.0)
    with pytest.raises(eq.EquivalenceError, match="RAM is free"):
        eq.run(dst, sha, eq.CONFIRM)  # no checker injected: the memory floor is enforced BEFORE any model load
    assert not (dst / eq.REPORT).exists()
    out = eq.run(dst, sha, eq.CONFIRM, checker=FromCache(dst), splitter=split_like_cached(dst))
    assert out["passed"] and "no RAGTruth row scored" in out["scope"] and (dst / eq.REPORT).exists()
    assert not (dst / mc.OUTPUT).exists()  # the frozen feature file is never produced by this check
    with pytest.raises(eq.EquivalenceError, match="written once"):
        eq.run(dst, sha, eq.CONFIRM, checker=FromCache(dst))


def test_a_tampered_input_is_refused(frozen_copy, tmp_path):
    work, sha = frozen_copy
    dst = tmp_path / "w"
    shutil.copytree(work, dst)
    (dst / "minicheck_cached_routebench.json").write_text("{}", encoding="utf8")
    with pytest.raises(mc.MiniCheckJobError, match="changed after the freeze"):
        eq.check(dst, sha, FromCache(work))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.1, 1.1])
def test_invalid_model_probabilities_never_produce_a_pass(frozen_copy, monkeypatch, value):
    work, sha = frozen_copy
    monkeypatch.setattr(mc, "minicheck_pair", lambda *_a: (value, [0.5]))
    with pytest.raises(eq.EquivalenceError, match="finite probability"):
        eq.check(work, sha, object())


def test_nonfinite_sentence_probability_is_rejected(frozen_copy, monkeypatch):
    work, sha = frozen_copy
    monkeypatch.setattr(mc, "minicheck_pair", lambda *_a: (0.5, [math.nan]))
    with pytest.raises(eq.EquivalenceError, match="finite probability"):
        eq.check(work, sha, object())


@pytest.mark.parametrize("rows,tolerance", [(0, None), (-1, None), (True, None), (9999, None),
                                           (None, math.nan), (None, math.inf), (None, -0.1)])
def test_invalid_check_controls_fail_closed(frozen_copy, rows, tolerance):
    work, sha = frozen_copy
    with pytest.raises(eq.EquivalenceError):
        eq.check(work, sha, FromCache(work), rows=rows, tolerance=tolerance)


@pytest.mark.parametrize("floor", [math.nan, math.inf, -1, 0, 2999])
def test_ram_floor_cannot_be_bypassed(frozen_copy, floor):
    work, sha = frozen_copy
    with pytest.raises(eq.EquivalenceError, match="RAM floor"):
        eq.run(work, sha, eq.CONFIRM, min_free_mb=floor)


def test_new_reports_keep_sentence_vectors_and_test_execution_separate(frozen_copy, tmp_path):
    work, sha = frozen_copy
    dst = tmp_path / "w"
    shutil.copytree(work, dst)
    out = eq.run(dst, sha, eq.CONFIRM, checker=FromCache(dst), splitter=split_like_cached(dst))
    assert out["schema_version"] == 2 and out["execution"] == "injected_checker"
    assert out["peak_gpu_memory_mb"] is None
    assert out["producer_sha256"] == mc._sha(eq.Path(eq.__file__))
    assert all(row["cached_sent"] == row["recomputed_sent"] for row in out["details"])


def test_frozen_input_drift_refuses_before_real_model_memory_or_load(frozen_copy, tmp_path, monkeypatch):
    work, sha = frozen_copy
    dst = tmp_path / "w"
    shutil.copytree(work, dst)
    (dst / "minicheck_cached_routebench.json").write_text("{}", encoding="utf8")
    monkeypatch.setattr(eq, "free_ram_mb", lambda: pytest.fail("memory check ran before frozen-input check"))
    with pytest.raises(mc.MiniCheckJobError, match="changed after"):
        eq.run(dst, sha, eq.CONFIRM)

"""The recovered pre-2026-09-28 `config_hash` formula, and the rule that stops it being abused.

Context, because this file exists to correct a wrong claim. An earlier revision of
`docs/KAPPA_DESIGN.md` §10.1 argued: every stored `JudgeConfig` field round-trips exactly, yet the
hash still differs, and `config_hash` has only two inputs the record does not store; the exemplars
hash is the constant empty-path value; therefore the prompts changed. An independent review
disputed that and reconstructed all eight stored hashes under an older field-only scheme.

The review was right. An exhaustive search over 131,072 candidate formulas (every subset of the 14
`JudgeConfig` fields crossed with each way of including the exemplars hash and the prompt-template
hash) finds exactly ONE that reproduces all eight recorded hashes, and it has no `prompt_template`
key at all. The reasoning above was unsound at its last step: a formula change explains the
mismatch completely, and the legacy hash could not have detected a prompt change in the first
place.

What that leaves is genuine uncertainty rather than a known loss, and uncertainty is easy to
launder. So these tests pin both halves: the formula reproduces the record, AND nothing may promote
a legacy judgment to current provenance.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from evalops.dataset import CaseRecord, Judgment
from evalops.experiments import (
    CURRENT_PROMPT_PROVENANCE,
    LEGACY_PROMPT_PROVENANCE,
    JudgmentCache,
    _cache_key,
    _legacy_cache_key,
)
from evalops.judge import _LEGACY_CONFIG_HASH_KEYS, JudgeConfig, prompt_template_hash

# Immutable pre-rejudge record: the live dev record now uses the current hash formula.
RECORD = (Path(__file__).parent.parent / "evalops" / "data"
          / "dev_experiments.previous.20260928T180211361238Z.json")


def _saved_configs() -> dict[str, dict]:
    assert RECORD.exists(), "the historical hash regression requires its archived record"
    experiments = json.loads(RECORD.read_text(encoding="utf-8"))
    out = {}
    for summary in experiments.get("run_summaries", []):
        cfg = summary.get("config") or {}
        if cfg.get("variant_id") and cfg.get("config_hash"):
            out[cfg["variant_id"]] = cfg
    if not out:
        pytest.skip("dev record carries no stored configs")
    return out


# ---------------------------------------------------------------- the recovered formula


def test_legacy_hash_reproduces_every_recorded_hash():
    saved = _saved_configs()
    assert len(saved) == 8, sorted(saved)
    for variant, cfg in sorted(saved.items()):
        rebuilt = JudgeConfig.from_dict(cfg).legacy_config_hash
        assert rebuilt == cfg["config_hash"], variant


def test_the_legacy_formula_omits_the_prompt_template():
    # The load-bearing fact. A prompt edit was invisible to this hash, so a hash mismatch says
    # nothing about the prompts either way.
    assert "prompt_template" not in _LEGACY_CONFIG_HASH_KEYS
    assert "prompt" not in " ".join(_LEGACY_CONFIG_HASH_KEYS)


def test_the_legacy_formula_differs_from_the_current_one_in_known_ways():
    fields = set(asdict(JudgeConfig(variant_id="x", model="m")))
    legacy = set(_LEGACY_CONFIG_HASH_KEYS)
    # Labels, plus the two fields added when the v8-v11 variants were declared.
    assert fields - legacy == {
        "variant_id", "notes", "restrict_tasks", "samples", "reasoning_effort",
        "context_path"}
    # `concurrency` and the literal `exemplars_path` were in the legacy payload; the current one
    # drops concurrency (it cannot change a verdict) and hashes exemplar CONTENTS instead.
    assert "concurrency" in legacy
    assert "exemplars_path" in legacy


def test_the_recovered_formula_is_the_only_one_that_fits():
    """Re-runs the search, narrowed, so the claim "exactly one formula matches" stays checked.

    A single matching formula is what makes the reconstruction evidence rather than a coincidence:
    if several candidates fitted eight data points, picking one would be storytelling.
    """
    saved = _saved_configs()
    cfgs = {v: JudgeConfig.from_dict(c) for v, c in saved.items()}
    target = {v: c["config_hash"] for v, c in saved.items()}
    fields = sorted(asdict(next(iter(cfgs.values()))))

    def h(payload: dict) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    extras = (
        ("exemplars", lambda c: c.exemplars_hash),
        ("prompt_template", lambda c: prompt_template_hash(c.mode)),
    )
    # Search the neighbourhood of the recovered answer: it plus/minus any one field, crossed with
    # every extras subset. The full 2^14 sweep belongs in a one-off script, not a unit test.
    recovered = set(_LEGACY_CONFIG_HASH_KEYS)
    neighbourhood = [recovered]
    for f in fields:
        neighbourhood.append(recovered ^ {f})

    matches = []
    for keep in neighbourhood:
        for n in range(len(extras) + 1):
            for chosen in itertools.combinations(extras, n):
                ok = True
                for variant, cfg in cfgs.items():
                    d = asdict(cfg)
                    payload = {k: d[k] for k in sorted(keep)}
                    for name, fn in chosen:
                        payload[name] = fn(cfg)
                    if h(payload) != target[variant]:
                        ok = False
                        break
                if ok:
                    matches.append((tuple(sorted(keep)), tuple(n for n, _ in chosen)))
    assert matches == [(tuple(sorted(recovered)), ())], matches


def test_legacy_and_current_hashes_disagree_for_the_recorded_configs():
    # If they agreed there would be nothing to reconcile, and this whole file would be dead code.
    for variant, cfg in sorted(_saved_configs().items()):
        c = JudgeConfig.from_dict(cfg)
        assert c.legacy_config_hash != c.config_hash, variant


def test_legacy_hash_is_deterministic():
    c = JudgeConfig(variant_id="v", model="m")
    assert c.legacy_config_hash == c.legacy_config_hash


def test_variant_id_and_notes_do_not_change_the_legacy_hash():
    a = JudgeConfig(variant_id="a", model="m", notes="one")
    b = JudgeConfig(variant_id="b", model="m", notes="two")
    assert a.legacy_config_hash == b.legacy_config_hash


def test_concurrency_does_change_the_legacy_hash_but_not_the_current_one():
    a = JudgeConfig(variant_id="v", model="m", concurrency=0)
    b = JudgeConfig(variant_id="v", model="m", concurrency=3)
    assert a.legacy_config_hash != b.legacy_config_hash
    assert a.config_hash == b.config_hash


# ---------------------------------------------------------------- provenance, and not laundering it


def _case():
    return CaseRecord(case_id="c1", group_id="g1", task_class="code", task_input="i",
                      context="", reference="r", candidate_output="o")


def _judgment(config_hash: str, model: str = "gpt-4o-mini") -> Judgment:
    return Judgment(case_id="c1", variant_id="v", model_requested=model, model_served=model,
                    config_hash=config_hash, raw_score=0.8, status="ok", cost_usd=0.0001)


def test_legacy_lookup_finds_a_judgment_written_under_the_old_key(tmp_path):
    cache = JudgmentCache(tmp_path / "cache.jsonl")
    cfg = JudgeConfig(variant_id="v", model="m")
    case = _case()
    cache.put(_legacy_cache_key(case, cfg), _judgment(cfg.legacy_config_hash))

    found, provenance = cache.legacy_lookup(case, cfg)
    assert found is not None
    assert provenance == LEGACY_PROMPT_PROVENANCE


def test_legacy_lookup_prefers_a_re_judged_current_entry(tmp_path):
    # Once a case is re-judged under pinned prompts, the verified judgment must win. Otherwise
    # lookup order would decide which evidence backs a bundle.
    cache = JudgmentCache(tmp_path / "cache.jsonl")
    cfg = JudgeConfig(variant_id="v", model="m")
    case = _case()
    cache.put(_legacy_cache_key(case, cfg), _judgment("legacy"))
    cache.put(_cache_key(case, cfg), _judgment(cfg.config_hash))

    found, provenance = cache.legacy_lookup(case, cfg)
    assert provenance == CURRENT_PROMPT_PROVENANCE
    assert found is not None and found.config_hash == cfg.config_hash


def test_legacy_lookup_reports_nothing_when_neither_key_is_present(tmp_path):
    cache = JudgmentCache(tmp_path / "cache.jsonl")
    found, provenance = cache.legacy_lookup(_case(), JudgeConfig(variant_id="v", model="m"))
    assert found is None
    assert provenance == LEGACY_PROMPT_PROVENANCE


def test_the_two_cache_keys_are_different(tmp_path):
    cfg = JudgeConfig(variant_id="v", model="m")
    case = _case()
    assert _cache_key(case, cfg) != _legacy_cache_key(case, cfg)


def test_run_variant_does_not_treat_a_legacy_entry_as_a_hit(tmp_path, monkeypatch):
    """A legacy judgment must not be silently reused as though it were current.

    This is the relabelling the review warned against, expressed as a test: if `run_variant`
    accepted the legacy key as a cache hit, a re-judge would be skipped and the old judgment would
    be stamped with today's `config_hash` by the surrounding report.
    """
    from evalops import experiments as ex

    cache = JudgmentCache(tmp_path / "cache.jsonl")
    cfg = JudgeConfig(variant_id="v", model="gpt-4o-mini", max_attempts=1)
    case = _case()
    cache.put(_legacy_cache_key(case, cfg), _judgment(cfg.legacy_config_hash))

    called: list[str] = []

    def fake_judge_case(c, config):
        called.append(c.case_id)
        return _judgment(config.config_hash)

    monkeypatch.setattr(ex, "judge_case", fake_judge_case)
    meter = ex.SpendMeter(cap_usd=1.0)
    judgments, summary = ex.run_variant([case], cfg, cache=cache, meter=meter, concurrency=1)

    assert called == ["c1"], "the legacy entry must not have satisfied the run"
    assert summary.n_cached == 0
    assert judgments[0].config_hash == cfg.config_hash


def test_rubric_versions_in_the_committed_cache_match_this_tree():
    """The positive half of the provenance finding, kept honest by a test.

    The rubric text is the bulk of a rubric-mode prompt, and every `rubric_version` recorded in the
    cached judgments matches a rubric in this tree. That is real evidence the judgments are not
    from some wholly different judge — which is why "the prompts are lost" was too strong. It is
    also not sufficient: the scaffolding around the rubric is covered by nothing.
    """
    from evalops.rubrics import load_all

    cache_path = Path(__file__).parent.parent / "evalops" / "data" / "judgment_cache.jsonl"
    if not cache_path.exists():
        pytest.skip("no judgment cache in this checkout")
    current = {f"{r.id}@{r.version}+{r.hash}" for r in load_all().values()}
    saved = _saved_configs()
    wanted = {c["config_hash"] for c in saved.values()}
    seen = set()
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line).get("judgment") or {}
        if raw.get("config_hash") in wanted and raw.get("rubric_version"):
            seen.add(raw["rubric_version"])
    assert seen, "no cached judgments matched the recorded config hashes"
    assert seen <= current, f"cached rubric versions absent from this tree: {sorted(seen - current)}"

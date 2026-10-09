"""Opt-in `claim_fidelity` judge variant (offline preparation; nothing here has been run against a provider).

v18 (`v18-reasoning-o4mini`) is untouched: same model, config hash `12fc8b550d954359`, rubric hash `d72496fbb2991071`, prompt-template hash `ef895d791c0bd860`. This module defines ONE new variant that differs from v18
in exactly one respect, its rubric directory (`rubrics_claim_fidelity_v1/`: the original four criteria, pass definition and boundary examples unchanged, plus one added criterion, `claim_fidelity`).

What the criterion is: an explicit attention mechanism. The existing rubric already prohibits modal strengthening and stronger generalisations (`claims_entailed` fails "a generalisation stronger than the
document supports", and a boundary example pins "will expand" versus "is considering expanding"); the judge sometimes fails to apply those existing instructions. `claim_fidelity` gives the same distinctions
their own yes/no verdict so they are checked on every response rather than folded into one broad criterion. It is not a previously missing rule.

How it is defined: by the assertion's EVIDENTIAL STATUS in the document, not by the speaker's authority. Opinion stays attributed; predictions, plans, possibilities and concerns stay qualified; disputed or
unconfirmed statements keep that qualification; tense and material qualifications are kept; verbs and quantifiers are no stronger than the document's. Faithful paraphrase passes, and so does an ordinary
unqualified factual report that the document itself states plainly, whoever the speaker is. No benchmark sentence, source or published expert explanation appears in the prompt.

Decision rule is unchanged: pass iff the score is >= 1.0, i.e. every criterion (now five) is answered yes; a missing criterion in the judge's output is a parse error (never a pass).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from . import rubrics
from .judge import JudgeConfig, prompt_template_hash
from .run_calibration import GRID

BASE_VARIANT = "v18-reasoning-o4mini"
VARIANT_ID = "v24-claim-fidelity-o4mini"
RUBRIC_DIR = Path(__file__).parent / "rubrics_claim_fidelity_v1"
CRITERION_ID = "claim_fidelity"
PASS_THRESHOLD = 1.0  # unchanged original all-criteria-pass threshold

#: Identity of the preserved baseline. A change to any of these means v18 was edited.
V18_IDENTITY = {"config_hash": "12fc8b550d954359", "rubric_hash": "d72496fbb2991071", "prompt_template_hash": "ef895d791c0bd860"}


def base_config() -> JudgeConfig:
    return next(c for c in GRID if c.variant_id == BASE_VARIANT)


def config() -> JudgeConfig:
    """v18's config with only the rubric directory (and the labels `variant_id` / `notes`) changed."""
    return dataclasses.replace(
        base_config(), variant_id=VARIANT_ID, rubric_dir=str(RUBRIC_DIR),
        notes="v18 plus the claim_fidelity criterion (explicit attention to evidential status); opt-in, offline-prepared, not yet measured")


def rubric() -> rubrics.Rubric:
    return rubrics.for_task("summarize", str(RUBRIC_DIR))


def identity() -> dict[str, str]:
    """Hashes that identify the new variant and prove the baseline is unchanged."""
    v18, new = base_config(), config()
    base_rubric = rubrics.for_task("summarize", v18.rubric_dir)
    return {"v18_config_hash": v18.config_hash, "v18_rubric_hash": base_rubric.hash, "v18_prompt_template_hash": prompt_template_hash(v18.mode),
            "claim_fidelity_config_hash": new.config_hash, "claim_fidelity_rubric_hash": rubric().hash, "claim_fidelity_rubric_version": f"{rubric().id}@{rubric().version}",
            "claim_fidelity_prompt_template_hash": prompt_template_hash(new.mode)}


def decision(score: float) -> int:
    """Pass (1) iff every criterion was answered yes; same rule as v18."""
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
        raise ValueError(f"invalid score {score!r}")
    return int(score >= PASS_THRESHOLD)

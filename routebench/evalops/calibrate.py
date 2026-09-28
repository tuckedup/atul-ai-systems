"""Judge calibration: choose a threshold on dev, freeze a bundle, measure it once on test.

Replaces the previous module, which read `human_labels/labels.csv` -- a file whose `human_score`
and `judge_score` columns were written from the same variable by `generate_data.py` -- and
reported kappa 1.0. Two defects were structural rather than cosmetic:

*   The gate was `if report["kappa"] < 0.6: raise`. For a single-class table
    `cohen_kappa_score` returns `nan`, and `nan < 0.6` is False, so an undefined kappa passed.
    Here the gate goes through `Agreement.passes()`, which requires `defined is True`.

*   Thresholds were tuned against the same numbers that defined the target. A threshold is now
    chosen on the DEV split only, written into a frozen bundle, and the TEST split is scored
    exactly once with that frozen value.

The separation of `select_threshold` (dev) from `evaluate` (test) is the whole point of the
file. `evaluate` takes a `CalibrationBundle` and refuses to modify it.
"""
from __future__ import annotations

import argparse
import getpass
import json
import math
import platform
import socket
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import (
    HEADLINE_PROVENANCE,
    Annotation,
    CaseRecord,
    Judgment,
    LabelProvenance,
)
from .metrics import Agreement, MetricInputError, agreement, binarize
from .rubrics import bundle_hash

#: RouteBench release policy. Deliberately a module constant rather than a default buried in a
#: function signature, and deliberately not applied to the other projects that share
#: `aisys.evals.calibrate` (they keep the core default).
ROUTEBENCH_MIN_KAPPA = 0.74

#: Bounded, declared-in-advance threshold grid. An unbounded search over a continuous parameter
#: on a few hundred dev items is a good way to fit noise.
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.05 * n, 2) for n in range(1, 20))


class CalibrationError(RuntimeError):
    pass


@dataclass
class Paired:
    """Judgments joined to labels. Built only after a bundle is frozen."""

    case_ids: list[str]
    human: list[int]
    judge_scores: list[float]
    groups: list[str]
    tasks: list[str]
    provenance: list[str]
    #: Which split these items came from ("train" / "dev" / "test"), or "" when the caller did
    #: not say. The module docstring claimed the dev -> freeze -> test ordering was "enforced by
    #: the CLI, not by documentation", but nothing in `select_threshold` or `evaluate` could see
    #: which split it had been handed -- so the ordering was enforced by neither. Carrying the
    #: split here lets both functions refuse the wrong one by themselves.
    split: str = ""
    #: `(variant_id, config_hash) -> count` over the judgments that produced `judge_scores`.
    #: `evaluate` checks this against the frozen bundle. Without it, scoring a frozen bundle with
    #: some *other* judge's scores -- or with two variants' scores mixed together -- was
    #: undetectable, because `Paired` kept no record of where its numbers came from.
    judges: dict[tuple[str, str], int] = field(default_factory=dict)
    #: Cases that had a label but no usable judgment, kept so completion can be reported.
    unjudged: list[str] = field(default_factory=list)
    #: Provenance of each entry in `unjudged`, parallel by index. Added so a `restrict()`ed
    #: (per-track) view can attribute unjudged cases to the right track instead of charging
    #: every track with the full unjudged count -- without this, per-track completion for a
    #: small track would be dragged down (or up) by cases that never belonged to it.
    unjudged_provenance: list[str] = field(default_factory=list)
    #: Cases whose judgment errored, by status.
    errors: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.case_ids)

    def restrict(self, allowed: frozenset[LabelProvenance]) -> Paired:
        allowed_values = {a.value for a in allowed}
        keep = [i for i, p in enumerate(self.provenance) if p in allowed_values]
        keep_u = [i for i, p in enumerate(self.unjudged_provenance) if p in allowed_values]
        return Paired(
            case_ids=[self.case_ids[i] for i in keep],
            human=[self.human[i] for i in keep],
            judge_scores=[self.judge_scores[i] for i in keep],
            groups=[self.groups[i] for i in keep],
            tasks=[self.tasks[i] for i in keep],
            provenance=[self.provenance[i] for i in keep],
            split=self.split,
            judges=dict(self.judges),
            unjudged=[self.unjudged[i] for i in keep_u],
            unjudged_provenance=[self.unjudged_provenance[i] for i in keep_u],
            errors=dict(self.errors),
        )

    @property
    def completion(self) -> float:
        total = len(self.case_ids) + len(self.unjudged)
        return len(self.case_ids) / total if total else 0.0


def pair(
    cases: Sequence[CaseRecord],
    labels: dict[str, Annotation],
    judgments: Sequence[Judgment],
    *,
    split: str = "",
) -> Paired:
    """Join judgments to labels by case_id. This is the ONLY place the two meet.

    Fixture-provenance labels are refused outright: they are the artefact this rewrite exists
    to remove, and a run that silently included them would reproduce the old kappa 1.0.
    """
    by_case = {c.case_id: c for c in cases}
    best: dict[str, Judgment] = {}
    errors: dict[str, int] = {}
    for j in judgments:
        if j.usable:
            best[j.case_id] = j
        else:
            errors[j.status] = errors.get(j.status, 0) + 1

    out = Paired([], [], [], [], [], [], split=split)
    for case_id, annotation in sorted(labels.items()):
        case = by_case.get(case_id)
        if case is None:
            continue
        if annotation.provenance is LabelProvenance.FIXTURE:
            raise CalibrationError(
                f"case {case_id} carries synthetic_fixture label provenance. Fixture labels are "
                "not admissible calibration evidence; they are the defect this module replaces."
            )
        judged = best.get(case_id)
        if judged is None or judged.raw_score is None:
            out.unjudged.append(case_id)
            out.unjudged_provenance.append(annotation.provenance.value)
            continue
        out.case_ids.append(case_id)
        out.human.append(int(annotation.label))
        out.judge_scores.append(float(judged.raw_score))
        out.groups.append(case.group_id)
        out.tasks.append(case.task_class)
        out.provenance.append(annotation.provenance.value)
        key = (judged.variant_id, judged.config_hash)
        out.judges[key] = out.judges.get(key, 0) + 1
    out.errors = errors
    return out


# ---------------------------------------------------------------- dev: choose a threshold


@dataclass
class ThresholdChoice:
    threshold: float
    dev_kappa: float
    grid: list[dict[str, Any]]
    rule: str = (
        "argmax dev kappa over THRESHOLD_GRID; ties broken toward the threshold closest to "
        "0.5, then the smaller value, so the choice is deterministic and not an artefact of "
        "dict ordering"
    )


def select_threshold(dev: Paired, *, grid: Sequence[float] = THRESHOLD_GRID) -> ThresholdChoice:
    """Pick the judge threshold that maximises kappa on DEV. Never call this with test data.

    "Never call this with test data" used to be a sentence in a docstring. It is now a check: a
    `Paired` tagged `split="test"` is refused here, so the one thing the whole dev -> freeze ->
    test protocol exists to prevent cannot be done by calling this function directly.
    """
    if dev.split == "test":
        raise CalibrationError(
            "select_threshold was handed the TEST split. Selecting a threshold against held-out "
            "data is the exact failure the freeze protocol exists to prevent: the number it "
            "produces is a fit, not a measurement. Select on dev, freeze, then measure once."
        )
    if len(dev) < 30:
        raise CalibrationError(
            f"threshold selection needs a usable dev set; got {len(dev)} paired items. "
            "Tuning on fewer would fit noise."
        )
    rows: list[dict[str, Any]] = []
    for t in grid:
        try:
            predicted = binarize(dev.judge_scores, t)
        except MetricInputError as e:
            raise CalibrationError(f"dev judge scores are invalid: {e}") from e
        a = agreement(dev.human, predicted, groups=dev.groups, bootstrap=0)
        rows.append({
            "threshold": t, "kappa": a.kappa, "defined": a.defined,
            "agreement": a.observed_agreement, "judge_prevalence": a.judge_prevalence,
            "confusion": a.confusion.as_dict(),
        })
    usable = [r for r in rows if r["defined"] and r["kappa"] is not None]
    if not usable:
        raise CalibrationError(
            "no threshold produced a defined kappa on dev. The judge's scores are constant, or "
            "the dev labels are single-class; neither can calibrate anything."
        )
    best = max(usable, key=lambda r: (r["kappa"], -abs(r["threshold"] - 0.5), -r["threshold"]))
    return ThresholdChoice(float(best["threshold"]), float(best["kappa"]), rows)


# ---------------------------------------------------------------- the frozen bundle


@dataclass
class CalibrationBundle:
    """Everything needed to reproduce the accepted judge, and nothing that could leak test data."""

    bundle_id: str
    created_at: str
    variant_id: str
    judge_config: dict[str, Any]
    threshold: float
    min_kappa: float
    rubric_bundle_hash: str
    dataset_hash: str
    split_seed: int
    dev_kappa: float
    dev_n: int
    selection_rule: str
    threshold_grid: list[dict[str, Any]]
    label_provenance_counts: dict[str, int]
    #: Digest of the LABELS the threshold was selected against. `dataset_hash` covers case
    #: contents only, so without this a bundle is bound to the questions and not the answers, and
    #: a relabelled corpus passes the gate untouched.
    annotations_hash: str = ""
    #: Digest of the actual group-to-split assignment. `split_seed` binds the recipe; this binds
    #: the result, which is what `of_case` actually reads.
    split_membership_hash: str = ""
    #: Filled in by `evaluate`; absent until the frozen bundle has faced the test split.
    test_result: dict[str, Any] | None = None
    software: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> CalibrationBundle:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @property
    def frozen(self) -> bool:
        return self.test_result is not None


def identity_hash(
    *,
    variant_id: str,
    threshold: float,
    judge_config: dict[str, Any],
    rubric_bundle_hash: str,
    dataset_hash: str,
    split_seed: int,
    annotations_hash: str = "",
    split_membership_hash: str = "",
) -> str:
    """Hash the fields that define a bundle's identity. The one place `bundle_id` is computed.

    Factored out so the identity can be RE-derived from a bundle on disk. Before this, `freeze`
    computed `bundle_id` inline and nothing ever recomputed it, which made the id decorative:
    editing `threshold` or `judge_config` in `calibration_bundle.json` by hand left a stale id
    that no check compared against anything, and the artifact passed the release gate describing
    a judge and a cutoff that were never measured.
    """
    from .dataset import content_hash

    payload: dict[str, Any] = {
        "variant": variant_id, "threshold": threshold, "config": judge_config,
        "rubrics": rubric_bundle_hash, "dataset": dataset_hash, "seed": split_seed,
    }
    # Added only when present, so a bundle frozen before these bindings existed keeps its id and
    # stays verifiable. A migration that silently changed every historical id would make the
    # integrity check reject exactly the artifacts it is supposed to vouch for.
    if annotations_hash:
        payload["annotations"] = annotations_hash
    if split_membership_hash:
        payload["split_membership"] = split_membership_hash
    return content_hash(payload)


def expected_bundle_id(bundle: CalibrationBundle) -> str:
    """Recompute what `bundle.bundle_id` must be, from the bundle's own recorded fields."""
    return identity_hash(
        variant_id=bundle.variant_id,
        threshold=bundle.threshold,
        judge_config=bundle.judge_config,
        rubric_bundle_hash=bundle.rubric_bundle_hash,
        dataset_hash=bundle.dataset_hash,
        split_seed=bundle.split_seed,
        annotations_hash=bundle.annotations_hash,
        split_membership_hash=bundle.split_membership_hash,
    )


def freeze(
    choice: ThresholdChoice,
    *,
    variant_id: str,
    judge_config: dict[str, Any],
    dataset_hash: str,
    split_seed: int,
    dev: Paired,
    min_kappa: float = ROUTEBENCH_MIN_KAPPA,
    annotations_hash: str = "",
    split_membership_hash: str = "",
    notes: str = "",
) -> CalibrationBundle:
    counts: dict[str, int] = {}
    for p in dev.provenance:
        counts[p] = counts.get(p, 0) + 1
    return CalibrationBundle(
        bundle_id=identity_hash(
            variant_id=variant_id, threshold=choice.threshold, judge_config=judge_config,
            rubric_bundle_hash=bundle_hash(), dataset_hash=dataset_hash, split_seed=split_seed,
            annotations_hash=annotations_hash, split_membership_hash=split_membership_hash,
        ),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        variant_id=variant_id,
        judge_config=judge_config,
        threshold=choice.threshold,
        min_kappa=min_kappa,
        rubric_bundle_hash=bundle_hash(),
        dataset_hash=dataset_hash,
        split_seed=split_seed,
        dev_kappa=choice.dev_kappa,
        dev_n=len(dev),
        selection_rule=choice.rule,
        threshold_grid=choice.grid,
        label_provenance_counts=counts,
        annotations_hash=annotations_hash,
        split_membership_hash=split_membership_hash,
        software={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "host": socket.gethostname(),
            "user": getpass.getuser(),
        },
        notes=notes,
    )


# ---------------------------------------------------------------- test: measure once


def evaluate(
    bundle: CalibrationBundle,
    test: Paired,
    *,
    bootstrap: int = 2000,
) -> dict[str, Any]:
    """Score the frozen bundle on the held-out split. Reports every track separately.

    Three things are checked before anything is measured, because `evaluate` previously took the
    bundle's threshold on trust and the `Paired` on faith:

    *   the items came from the test split, not from dev (which would report a fitted number as a
        held-out one);
    *   every judgment came from the variant the bundle names, so the measurement describes the
        judge that was frozen;
    *   that variant's `config_hash` matches the bundle's, so a rubric or prompt edit after the
        freeze cannot be measured as if it were the frozen judge.
    """
    if not test.case_ids:
        raise CalibrationError("test split has no paired items; nothing to measure")
    if test.split and test.split != "test":
        raise CalibrationError(
            f"evaluate() was handed the {test.split!r} split. A frozen bundle is measured on the "
            "held-out split; scoring it on dev reports the data the threshold was fitted to and "
            "would overstate agreement."
        )
    if test.judges:
        wrong = sorted(v for (v, _) in test.judges if v != bundle.variant_id)
        if wrong:
            raise CalibrationError(
                f"bundle {bundle.variant_id!r} is being measured with judgments from "
                f"{wrong}. The reported kappa would describe a judge the bundle does not name."
            )
        frozen_hash = str(bundle.judge_config.get("config_hash", ""))
        if frozen_hash:
            drifted = sorted(h for (_, h) in test.judges if h and h != frozen_hash)
            if drifted:
                raise CalibrationError(
                    f"judgments carry config_hash {drifted}, but the frozen bundle records "
                    f"{frozen_hash!r}. Something the judge depends on changed after the freeze, "
                    "so these judgments are not the frozen judge's output."
                )

    def track(p: Paired, name: str) -> dict[str, Any] | None:
        #: How many labelled test items existed for this track (judged-and-paired plus
        #: unjudged-but-labelled), and what fraction of them got an automated decision. This is
        #: `Paired.completion` restated under the names the release gate checks for, so a track
        #: that is badly covered cannot hide behind a healthy overall completion number.
        n_labelled = len(p) + len(p.unjudged)
        track_completion = p.completion
        if len(p) < 20:
            return {
                "track": name, "n": len(p), "insufficient": True,
                "note": f"only {len(p)} items; too few to report a kappa for this track",
                "n_labelled": n_labelled, "track_completion": track_completion,
            }
        predicted = binarize(p.judge_scores, bundle.threshold)
        a: Agreement = agreement(
            p.human, predicted, groups=p.groups, tasks=p.tasks, bootstrap=bootstrap
        )
        d = a.as_dict()
        d.update({
            "track": name, "insufficient": False,
            "passes_min_kappa": a.passes(bundle.min_kappa),
            "provenance_counts": {
                v: p.provenance.count(v) for v in sorted(set(p.provenance))
            },
            "n_labelled": n_labelled, "track_completion": track_completion,
        })
        return d

    overall = track(test, "all_admissible_labels")
    headline = track(test.restrict(HEADLINE_PROVENANCE), "human_judgment_of_response")
    oracle = track(test.restrict(frozenset({LabelProvenance.GOLD_ORACLE})), "gold_oracle")

    return {
        "bundle_id": bundle.bundle_id,
        "variant_id": bundle.variant_id,
        "threshold": bundle.threshold,
        "min_kappa": bundle.min_kappa,
        "rubric_bundle_hash": bundle.rubric_bundle_hash,
        "dataset_hash": bundle.dataset_hash,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completion": {
            "paired": len(test), "unjudged": len(test.unjudged),
            "completion_rate": test.completion, "judge_errors": test.errors,
        },
        "tracks": {t["track"]: t for t in (overall, headline, oracle) if t is not None},
    }


# ---------------------------------------------------------------- promotion gate


def _to_finite_float(value: Any) -> float | None:
    """Coerce `value` to a finite float, or `None` if it is not genuinely one.

    Deliberately stricter than `float(value)`: booleans and numeric-looking strings (e.g.
    `"1.0"`) are rejected by type, and NaN / +-inf are rejected by `math.isfinite`. A gate that
    accepted `"1.0"` or `True` as a completion rate would be trusting a value that never went
    through the arithmetic that produces a real one.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def validate_bundle(
    path: str | Path,
    *,
    min_kappa: float = ROUTEBENCH_MIN_KAPPA,
    require_headline: bool = True,
    min_completion: float = 1.0,
    expected_dataset_hash: str | None = None,
    expected_annotations_hash: str | None = None,
    expected_split_membership_hash: str | None = None,
    require_data_bindings: bool = True,
) -> tuple[bool, list[str]]:
    """Decide whether a calibration artifact may back a routing/promotion decision.

    Returns `(ok, reasons)`. `reasons` is non-empty whenever `ok` is False and is written into
    the audit event, so a blocked promotion always says why. Never raises on a malformed
    artifact -- a structurally nonsense bundle is a rejection, not a crash.

    `min_completion` defaults to 1.0: the project's acceptance contract requires an automated
    decision for every labelled test item. A caller may pass a lower value explicitly to accept
    a documented, deliberate relaxation; nothing here lowers it silently.

    Every field pulled from the artifact is treated as untrusted. In particular `defined` and
    `passes_min_kappa` are self-reported by whatever produced the file, so the gate recomputes
    both from `kappa` and `min_kappa` rather than trusting them -- and a track that *claims*
    `passes_min_kappa: True` while the recomputed decision is False is rejected with that
    contradiction named explicitly, since a self-inconsistent artifact is evidence of tampering
    or a bug, not something to pass on the strength of its own say-so.
    """
    reasons: list[str] = []
    p = Path(path)
    if not p.exists():
        return False, [f"no calibration artifact at {p}"]
    try:
        b = CalibrationBundle.load(p)
    except Exception as e:  # noqa: BLE001 - a corrupt artifact must block, not crash the gate
        return False, [f"calibration artifact is unreadable: {type(e).__name__}: {e}"]

    # Internal integrity first. `bundle_id` is a hash over the variant, threshold, judge config,
    # rubric hash, dataset hash and split seed, so recomputing it detects any post-freeze edit to
    # those fields -- which is what makes the id load-bearing rather than ornamental.
    try:
        recomputed = expected_bundle_id(b)
    except Exception as e:  # noqa: BLE001 - an unhashable config is a rejection, not a crash
        reasons.append(f"cannot recompute bundle_id: {type(e).__name__}: {e}")
    else:
        if b.bundle_id != recomputed:
            reasons.append(
                f"bundle_id mismatch: artifact says {b.bundle_id!r}, its own recorded fields hash "
                f"to {recomputed!r}. The variant, threshold, judge config, rubric hash, dataset "
                "hash or split seed was changed after the freeze; the measurement in this file "
                "does not describe the configuration it now claims."
            )

    if not b.frozen:
        reasons.append("artifact has no test_result: the bundle was never measured on held-out data")
    if expected_dataset_hash is not None and b.dataset_hash != expected_dataset_hash:
        reasons.append(
            f"dataset hash mismatch: artifact {b.dataset_hash}, corpus {expected_dataset_hash}. "
            "The corpus changed since calibration, so this kappa was measured on different data "
            "than the judge is now being trusted for."
        )
    if require_data_bindings and not b.annotations_hash:
        reasons.append(
            "artifact records no annotations_hash: it is bound to the case contents but not to the "
            "LABELS the threshold was selected against, so a relabelled corpus would pass. Re-freeze "
            "with the label digest, or pass require_data_bindings=False to accept a pre-binding "
            "artifact deliberately."
        )
    if require_data_bindings and not b.split_membership_hash:
        reasons.append(
            "artifact records no split_membership_hash: it is bound to the split SEED but not to "
            "the assignment the seed produced, and `of_case` reads the assignment. Re-freeze with "
            "the membership digest, or pass require_data_bindings=False deliberately."
        )
    if expected_annotations_hash is not None and b.annotations_hash and \
            b.annotations_hash != expected_annotations_hash:
        reasons.append(
            f"annotation hash mismatch: artifact {b.annotations_hash}, corpus "
            f"{expected_annotations_hash}. The human labels changed since calibration, so this "
            "kappa was measured against different ground truth than the judge is now trusted for. "
            "Editing labels after seeing judge output is the first thing the protocol forbids."
        )
    if expected_split_membership_hash is not None and b.split_membership_hash and \
            b.split_membership_hash != expected_split_membership_hash:
        reasons.append(
            f"split membership mismatch: artifact {b.split_membership_hash}, plan "
            f"{expected_split_membership_hash}. Cases moved between splits after the freeze, so "
            "the held-out measurement no longer covers the items it reports on."
        )
    if b.rubric_bundle_hash != bundle_hash():
        reasons.append(
            f"rubric hash mismatch: artifact {b.rubric_bundle_hash}, working tree {bundle_hash()}. "
            "The rubrics changed since calibration, so the measurement no longer describes this judge."
        )
    if b.min_kappa < min_kappa:
        reasons.append(f"artifact was accepted at min_kappa={b.min_kappa} below policy {min_kappa}")

    test_result: Any = b.test_result
    if test_result is not None and not isinstance(test_result, dict):
        reasons.append(f"artifact 'test_result' is not a mapping: {type(test_result).__name__}")
        test_result = {}
    test_result = test_result or {}

    tracks: Any = test_result.get("tracks", {})
    if not isinstance(tracks, dict):
        reasons.append(f"artifact 'tracks' is not a mapping: {type(tracks).__name__}")
        tracks = {}
    if not tracks:
        reasons.append("artifact reports no measurement tracks")

    name = "human_judgment_of_response" if require_headline else "all_admissible_labels"
    t: Any = tracks.get(name)
    if t is None:
        reasons.append(f"artifact has no '{name}' track")
    elif not isinstance(t, dict):
        reasons.append(f"track '{name}' is not a mapping: {type(t).__name__}")
    elif t.get("insufficient"):
        reasons.append(f"track '{name}': {t.get('note', 'insufficient data')}")
    else:
        defined = t.get("defined")
        kappa_raw = t.get("kappa")
        kappa = _to_finite_float(kappa_raw)

        if defined is not True:
            reasons.append(
                f"track '{name}' kappa is undefined (defined={defined!r}); an undefined kappa "
                "never passes the gate, regardless of the reported kappa value"
            )
        elif kappa is None:
            reasons.append(
                f"track '{name}' kappa is {kappa_raw!r}, not a finite number; a missing, NaN, "
                "or infinite kappa never passes the gate"
            )
        elif not (-1.0 <= kappa <= 1.0):
            reasons.append(
                f"track '{name}' kappa {kappa} is outside the valid range [-1.0, 1.0] and is "
                "not a kappa at all"
            )
        else:
            recomputed_passes = kappa >= min_kappa
            if not recomputed_passes:
                reasons.append(f"track '{name}' kappa {kappa} is below the {min_kappa} policy")
            if t.get("passes_min_kappa") is True and not recomputed_passes:
                reasons.append(
                    f"track '{name}' claims passes_min_kappa=True but the recomputed decision "
                    f"from kappa {kappa} against min_kappa {min_kappa} is False; the artifact is "
                    "self-inconsistent"
                )

        if "track_completion" not in t:
            reasons.append(
                f"track '{name}' reports no per-track coverage (n_labelled/track_completion); "
                "coverage cannot be verified for this track"
            )
        else:
            tc_raw = t.get("track_completion")
            tc = _to_finite_float(tc_raw)
            if tc is None or not (0.0 <= tc <= 1.0):
                reasons.append(
                    f"track '{name}' track_completion is {tc_raw!r}; expected a finite number "
                    "in [0.0, 1.0]"
                )
            elif tc < min_completion:
                reasons.append(
                    f"track '{name}' coverage is {tc:.1%} of labelled items for this track, "
                    f"below the required {min_completion:.1%}; overall coverage can be healthy "
                    "while the primary track is not"
                )

    comp: Any = test_result.get("completion")
    if not isinstance(comp, dict) or not comp:
        reasons.append("no completion metadata; coverage cannot be verified")
    else:
        rate_raw = comp.get("completion_rate")
        rate = _to_finite_float(rate_raw)
        if rate is None or not (0.0 <= rate <= 1.0):
            reasons.append(
                f"completion_rate is {rate_raw!r}; expected a finite number in [0.0, 1.0]"
            )
        else:
            paired = comp.get("paired")
            unjudged = comp.get("unjudged")
            if (
                isinstance(paired, int) and not isinstance(paired, bool)
                and isinstance(unjudged, int) and not isinstance(unjudged, bool)
                and paired >= 0 and unjudged >= 0
            ):
                total = paired + unjudged
                if total == 0:
                    reasons.append(
                        "paired + unjudged is 0; there are no labelled test items to measure "
                        "coverage over"
                    )
                else:
                    expected = paired / total
                    if abs(expected - rate) > 1e-6:
                        reasons.append(
                            f"completion_rate {rate} is inconsistent with paired={paired} and "
                            f"unjudged={unjudged} (expected {expected})"
                        )
            if rate < min_completion:
                reasons.append(
                    f"only {rate:.1%} of labelled test cases received an automated decision; "
                    f"the policy requires at least {min_completion:.1%} coverage"
                )
    return (not reasons), reasons


def corpus_bindings(data_dir: str | Path) -> dict[str, str | None]:
    """The three data digests a bundle should be bound to, or Nones if the corpus is absent.

    Returned together because binding one and not the others is the hole this closes: the gate
    used to check case contents while the labels and the split assignment went unchecked.
    """
    out: dict[str, str | None] = {
        "dataset": None, "annotations": None, "split_membership": None,
    }
    try:
        from .dataset import Corpus

        corpus = Corpus.load(data_dir)
        if corpus.cases:
            out["dataset"] = corpus.cases_hash
            out["annotations"] = corpus.annotations_hash
    except Exception as e:  # noqa: BLE001 - absence of a corpus is not a gate failure
        # Deliberately swallowed and recorded, not logged: `validate_bundle` treats an absent
        # binding as "unverified" and says so, which is the honest outcome on a machine that does
        # not carry the corpus. Raising here would make validating an artifact impossible there.
        out["dataset_error"] = f"{type(e).__name__}: {e}"
    try:
        from .splits import SplitPlan

        out["split_membership"] = SplitPlan.load(Path(data_dir) / "splits.json").membership_hash
    except Exception as e:  # noqa: BLE001 - same reasoning as above
        out["split_membership_error"] = f"{type(e).__name__}: {e}"
    return out


def corpus_dataset_hash(data_dir: str | Path) -> str | None:
    """The dataset hash of the corpus on disk, or None if it cannot be loaded.

    Used by the CLI so `calibrate-check` (which is what CI runs) binds the artifact to the corpus
    in the tree, not just to the rubrics. Returns None rather than raising: validating an artifact
    on a machine without the corpus is a legitimate thing to do, and is reported as unverified
    rather than as a rejection.
    """
    try:
        from .dataset import Corpus, content_hash

        corpus = Corpus.load(data_dir)
        if not corpus.cases:
            return None
        return content_hash(sorted(c.fingerprint for c in corpus.cases))
    except Exception:  # noqa: BLE001 - absence of a corpus is not a gate failure
        return None


def _cli() -> int:
    ap = argparse.ArgumentParser(description="Validate a RouteBench calibration artifact.")
    ap.add_argument("artifact", nargs="?", default="routebench/evalops/data/calibration_bundle.json")
    ap.add_argument("--min-kappa", type=float, default=ROUTEBENCH_MIN_KAPPA)
    ap.add_argument("--allow-oracle-track", action="store_true",
                    help="accept the gold-oracle track instead of requiring human judgments")
    ap.add_argument("--no-dataset-check", action="store_true",
                    help="skip binding the artifact to the corpus on disk (for validating an "
                         "artifact on a machine that does not carry the corpus)")
    args = ap.parse_args()
    bindings: dict[str, str | None] = {
        "dataset": None, "annotations": None, "split_membership": None,
    }
    if not args.no_dataset_check:
        bindings = corpus_bindings(Path(args.artifact).parent)
        missing = [k for k, v in bindings.items() if v is None]
        if missing:
            print(f"note: corpus bindings NOT verified for {missing} (no corpus beside the "
                  "artifact)", file=sys.stderr)
    ok, reasons = validate_bundle(
        args.artifact, min_kappa=args.min_kappa, require_headline=not args.allow_oracle_track,
        expected_dataset_hash=bindings["dataset"],
        expected_annotations_hash=bindings["annotations"],
        expected_split_membership_hash=bindings["split_membership"],
    )
    if ok:
        b = CalibrationBundle.load(args.artifact)
        t = (b.test_result or {}).get("tracks", {})
        name = "all_admissible_labels" if args.allow_oracle_track else "human_judgment_of_response"
        row = t.get(name, {})
        ci = row.get("interval", {})
        print(f"PASS  {b.variant_id}  threshold={b.threshold}  n={row.get('n')}")
        print(f"      kappa={row.get('kappa'):.4f}  95% CI "
              f"[{ci.get('low')}, {ci.get('high')}]  track={name}")
        return 0
    print("FAIL  calibration artifact rejected:", file=sys.stderr)
    for r in reasons:
        print(f"  - {r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())

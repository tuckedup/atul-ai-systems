"""Render the human-readable calibration report from the frozen artifacts.

Kept separate from `calibrate.py` on purpose: reporting reads artifacts and writes prose, and
should never be in a position to recompute or adjust a number. Everything printed here comes
from `calibration_bundle.json` and `dev_experiments.json` as they sit on disk.

The report deliberately prints the things that make a kappa interpretable and are usually left
out: the confusion matrix, class prevalence on both sides, the completion rate, per-task
breakdown, the full threshold grid, the ablation table including variants that lost, and the
measured cost. A bare point estimate is not a result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DATA = Path(__file__).parent / "data"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "undefined"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _track_table(tracks: dict[str, Any], min_kappa: float) -> list[str]:
    lines = [
        "| track | n | groups | kappa | 95% CI | raw agreement | human pass rate | judge pass rate | >= policy |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for name, row in sorted(tracks.items()):
        if row.get("insufficient"):
            lines.append(f"| `{name}` | {row.get('n')} | - | - | - | - | - | - | insufficient data |")
            continue
        ci = row.get("interval", {})
        bounds = (
            f"[{_fmt(ci.get('low'), '.3f')}, {_fmt(ci.get('high'), '.3f')}]"
            if ci.get("usable") else f"unavailable ({ci.get('note', '')[:40]})"
        )
        lines.append(
            f"| `{name}` | {row['n']} | {row.get('n_groups', '-')} | **{_fmt(row.get('kappa'))}** | "
            f"{bounds} | {_fmt(row.get('observed_agreement'), '.3f')} | "
            f"{_fmt(row.get('human_prevalence'), '.3f')} | {_fmt(row.get('judge_prevalence'), '.3f')} | "
            f"{'yes' if row.get('passes_min_kappa') else 'NO'} |"
        )
    lines.append("")
    lines.append(f"Policy: kappa >= {min_kappa} as a point estimate on the frozen test split.")
    return lines


def render(bundle: dict[str, Any], dev: dict[str, Any] | None) -> str:
    result = bundle.get("test_result") or {}
    tracks = result.get("tracks", {})
    out: list[str] = [
        "# RouteBench judge calibration report",
        "",
        f"- Bundle: `{bundle.get('bundle_id')}`  ·  frozen {bundle.get('created_at')}",
        (
            f"- Judge variant: `{bundle.get('variant_id')}`  ·  model "
            f"`{(bundle.get('judge_config') or {}).get('model')}`"
        ),
        (
            f"- Decision threshold: **{bundle.get('threshold')}** (selected on dev, n="
            f"{bundle.get('dev_n')}, dev kappa {_fmt(bundle.get('dev_kappa'))})"
        ),
        f"- Rubric bundle hash: `{bundle.get('rubric_bundle_hash')}`",
        (
            f"- Dataset hash: `{bundle.get('dataset_hash')}`  ·  split seed "
            f"`{bundle.get('split_seed')}`"
        ),
        f"- Measured on the held-out split at {result.get('measured_at', 'not measured')}",
        "",
    ]

    if not tracks:
        out += [
            "## Result",
            "",
            "**Not measured.** The bundle is frozen but has not faced the test split. Run",
            "`python -m evalops.run_calibration test`.",
            "",
        ]
        return "\n".join(out)

    out += ["## Held-out result", ""]
    out += _track_table(tracks, float(bundle.get("min_kappa", 0.74)))
    out += [""]

    comp = result.get("completion", {})
    out += [
        "### Completion",
        "",
        f"- Labelled test cases with an automated decision: **{comp.get('paired')}**",
        f"- Labelled test cases with no usable judgment: {comp.get('unjudged')}",
        f"- Completion rate: **{_fmt(comp.get('completion_rate'), '.1%')}**",
        f"- Judge errors by status: `{comp.get('judge_errors') or 'none'}`",
        "",
        "Errored judgments are excluded from the metric rather than counted as a fail. The gate",
        "requires >=98% completion so coverage cannot be traded for agreement.",
        "",
    ]

    for name, row in sorted(tracks.items()):
        if row.get("insufficient") or not row.get("confusion"):
            continue
        c = row["confusion"]
        out += [
            f"### Confusion — `{name}`",
            "",
            "| | judge: pass | judge: fail |",
            "|---|---:|---:|",
            f"| **human: pass** | {c['tp']} | {c['fn']} |",
            f"| **human: fail** | {c['fp']} | {c['tn']} |",
            "",
        ]
        per_task = row.get("per_task") or {}
        if per_task:
            out += ["| task | n | kappa | raw agreement | note |", "|---|---:|---:|---:|---|"]
            for task, t in sorted(per_task.items()):
                out.append(
                    f"| `{task}` | {t['n']} | {_fmt(t.get('kappa'))} | "
                    f"{_fmt(t.get('observed_agreement'), '.3f')} | {t.get('reason', '')[:60]} |"
                )
            out.append("")

    if dev:
        out += [
            "## Variant grid (selected on dev — the test split played no part)",
            "",
            "| variant | model | mode | n | completion | kappa @0.5 | best dev kappa | best t | notes |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in dev.get("variants", []):
            if row.get("skipped"):
                out.append(
                    f"| `{row['variant_id']}` | - | - | - | - | - | - | - | "
                    f"SKIPPED: {row['skipped']} |"
                )
                continue
            at05 = (row.get("at_0.5") or {}).get("kappa")
            out.append(
                f"| `{row['variant_id']}` | `{row.get('model')}` | {row.get('mode')} | "
                f"{row.get('n_paired')} | {_fmt(row.get('completion'), '.1%')} | "
                f"{_fmt(at05)} | {_fmt(row.get('best_dev_kappa'))} | "
                f"{row.get('best_threshold')} | {row.get('notes', '')[:70]} |"
            )
        out += [
            "",
            (
                f"Dev spend: ${_fmt((dev.get('spend') or {}).get('spent_usd'), '.4f')} over "
                f"{(dev.get('spend') or {}).get('calls')} judged calls."
            ),
            "",
        ]
        chosen = bundle.get("variant_id")
        out += [
            f"The frozen variant is `{chosen}`. Losing variants are retained above on purpose: a",
            "grid that only reports its winner hides how much of the result is selection.",
            "",
            "### Threshold grid for the frozen variant (dev only)",
            "",
            "| threshold | dev kappa | dev agreement | judge pass rate |",
            "|---:|---:|---:|---:|",
        ]
        for row in bundle.get("threshold_grid", []):
            out.append(
                f"| {row['threshold']} | {_fmt(row.get('kappa'))} | "
                f"{_fmt(row.get('agreement'), '.3f')} | {_fmt(row.get('judge_prevalence'), '.3f')} |"
            )
        out += ["", f"Selection rule: {bundle.get('selection_rule')}", ""]

    run = result.get("run_summary") or {}
    out += [
        "## Provenance and cost",
        "",
        f"- Label provenance on the selection split: `{bundle.get('label_provenance_counts')}`",
        f"- Models actually served during the test run: `{run.get('models_served')}`",
        f"- Test-run spend: ${_fmt((run.get('spend') or {}).get('spent_usd'), '.4f')}",
        f"- Software: `{bundle.get('software')}`",
        "",
        "## What this number is not",
        "",
        "- It is a **point estimate** of kappa on the frozen test split. A claim that the",
        f"  population kappa exceeds {bundle.get('min_kappa')} would need the lower CI bound above",
        "  it, which is a stronger claim and needs a larger test set. Both are printed above.",
        "- The `human_gold_reference_oracle` track is a human-authored ground truth applied",
        "  mechanically, not a person reading the response. It is reported separately from the",
        "  human-judgment track for exactly that reason.",
        "- Any further tuning after this measurement requires a new confirmation set. Re-tuning",
        "  against this test split and re-reporting would make the number meaningless.",
        "",
        "Generated by `evalops/report.py` from `calibration_bundle.json` and",
        "`dev_experiments.json`. No number here is recomputed or adjusted at render time.",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default=str(DATA / "calibration_bundle.json"))
    ap.add_argument("--dev", default=str(DATA / "dev_experiments.json"))
    ap.add_argument("--out", default=str(DATA / "CALIBRATION_REPORT.md"))
    args = ap.parse_args(argv)

    bundle = _load(Path(args.bundle))
    if bundle is None:
        print(f"no calibration bundle at {args.bundle}")
        return 1
    text = render(bundle, _load(Path(args.dev)))
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

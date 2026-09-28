"""`make bench` -- print the quality x cost x latency matrix from MEASURED artifacts only.

The previous version of this file was:

    QUALITY = {"provider-small": 0.78, "provider-frontier": 0.93, "self-hosted": 0.81}

Three hardcoded constants printed under the heading "quality". They were illustrative
configuration for exercising the router, but `make bench` is listed in the definition of done as
the command that prints the measured matrix, so printing them there made invented numbers look
like results.

This version reads the artifacts on disk and, for anything absent, prints `not measured` with
the command that would produce it. An empty table is a better answer than a plausible one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DATA = Path(__file__).parent / "data"
QUALITY_MATRIX = DATA / "quality_matrix.json"
CALIBRATION = DATA / "calibration_bundle.json"
INFERENCE = Path(__file__).parent.parent / "inference" / "RESULTS.md"

#: Backend price/latency configuration. These are inputs to the routing policy, not
#: measurements, and are labelled as such in the output.
BACKEND_CONFIG: dict[str, dict[str, Any]] = {
    "provider-small": {"provider": "openai", "input_per_1m": 0.15, "output_per_1m": 0.60},
    "provider-frontier": {"provider": "anthropic", "input_per_1m": 3.00, "output_per_1m": 15.00},
    "self-hosted": {"provider": "vllm", "input_per_1m": 0.02, "output_per_1m": 0.02},
}

MISSING = "not measured"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _judge_line() -> list[str]:
    bundle = _load(CALIBRATION)
    if bundle is None:
        return [
            f"judge calibration:   {MISSING}",
            "  produce it with:   python -m evalops.run_calibration dev && ... freeze && ... test",
        ]
    result = bundle.get("test_result") or {}
    tracks = result.get("tracks", {})
    lines = [
        (
            f"judge calibration:   variant={bundle.get('variant_id')} "
            f"threshold={bundle.get('threshold')} bundle={bundle.get('bundle_id')}"
        ),
        f"  rubric hash:       {bundle.get('rubric_bundle_hash')}",
        f"  dev kappa:         {bundle.get('dev_kappa')} (n={bundle.get('dev_n')}, selection split)",
    ]
    if not tracks:
        lines.append(f"  held-out kappa:    {MISSING} -- bundle frozen but never tested")
        return lines
    for name, row in sorted(tracks.items()):
        if row.get("insufficient"):
            lines.append(f"  {name:30s} n={row.get('n')}  insufficient data")
            continue
        ci = row.get("interval", {})
        bounds = (
            f"[{ci['low']:.3f}, {ci['high']:.3f}]" if ci.get("usable") else "CI unavailable"
        )
        lines.append(
            f"  {name:30s} n={row['n']:4d}  kappa={row['kappa']:.4f}  95% {bounds}  "
            f"agreement={row['observed_agreement']:.3f}  "
            f"pass@{bundle.get('min_kappa')}={row.get('passes_min_kappa')}"
        )
    comp = result.get("completion", {})
    if comp:
        lines.append(
            f"  completion:        {comp.get('completion_rate', 0):.1%} "
            f"({comp.get('paired')} paired, {comp.get('unjudged')} unjudged)"
        )
    return lines


def _quality_table() -> list[str]:
    matrix = _load(QUALITY_MATRIX)
    header = f"{'model':<22}{'provider':<12}{'in $/1m':>10}{'out $/1m':>10}{'quality by task':>20}"
    lines = [header, "-" * len(header)]
    cells = (matrix or {}).get("cells", {})
    for model, cfg in BACKEND_CONFIG.items():
        measured = cells.get(model, {})
        if measured:
            body = " ".join(
                f"{task}={cell['quality']:.3f}(n={cell['n_graded']})"
                for task, cell in sorted(measured.items())
            )
        else:
            body = MISSING
        lines.append(
            f"{model:<22}{cfg['provider']:<12}{cfg['input_per_1m']:>10.2f}"
            f"{cfg['output_per_1m']:>10.2f}  {body}"
        )
    if matrix is None:
        lines += [
            "",
            f"quality matrix:      {MISSING}",
            "  produce it with:   python -m evalops.offline (needs live backends + a valid bundle)",
        ]
    else:
        lines += [
            "",
            f"quality measured at: {matrix.get('measured_at')}",
            f"  judge variant:     {matrix.get('judge_variant')}",
            f"  bundle:            {matrix.get('calibration_bundle_id')}",
        ]
        if matrix.get("notes"):
            lines.append(f"  notes:             {matrix['notes']}")
    return lines


def _latency_line() -> list[str]:
    if INFERENCE.exists():
        return [f"inference results:   {INFERENCE}"]
    return [
        f"p95 TTFT / TPOT / throughput: {MISSING}",
        "  requires the vLLM load-test sweep; see BLOCKERS.md (needs a container runtime).",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit the raw artifacts instead of a table")
    args = ap.parse_args(argv)

    if args.json:
        print(json.dumps(
            {"quality_matrix": _load(QUALITY_MATRIX), "calibration": _load(CALIBRATION),
             "backend_config": BACKEND_CONFIG},
            indent=2, sort_keys=True))
        return 0

    print("RouteBench measured matrix")
    print("=" * 78)
    print("\n".join(_quality_table()))
    print()
    print("\n".join(_judge_line()))
    print()
    print("\n".join(_latency_line()))
    print()
    print("Price and latency columns are routing-policy configuration, not measurements.")
    print(f"Anything marked '{MISSING}' has no artifact on disk; no placeholder is substituted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

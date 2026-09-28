"""Regenerate the synthetic SMOKE FIXTURES. This script does not produce evaluation data.

It used to. The previous version advertised itself as generating "the committed 200-case eval
suite and 50-label calibration set", and the calibration set it wrote was:

    for number in range(50):
        value = number % 2
        labels.append(f"label-{number + 1:02d},{value},{value}")
        #                                       human  judge

One variable in both columns. Cohen's kappa on that file is 1.0 by construction, and RouteBench
reported it as the judge's measured agreement with humans. The 200 "eval cases" were equally
hollow: `input: "coding case 7"`, `expected: "ok"`, grader `exact`.

So this script now writes only into `fixtures/`, names its outputs for what they are, and no
longer emits anything called a human label. The real corpus is built by `evalops/build.py` from
public benchmark sources with recorded provenance; real labels carry an explicit
`LabelProvenance` and the calibration gate refuses `synthetic_fixture`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from .taxonomy import MEASURED

FIXTURES = Path(__file__).parent / "fixtures"


def write_router_smoke_suite(n: int = 200) -> Path:
    """A content-free suite for exercising the loader and the router, not for measuring quality."""
    cases = [
        {
            "id": f"smoke-{index + 1:03d}",
            "input": f"{MEASURED[index % len(MEASURED)]} smoke case {index + 1}",
            "expected": "ok",
            "tags": [MEASURED[index % len(MEASURED)]],
            "grader": "exact",
        }
        for index in range(n)
    ]
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / "generated_router_smoke.yaml"
    path.write_text(yaml.safe_dump(cases, sort_keys=False), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, default=200)
    args = ap.parse_args(argv)
    path = write_router_smoke_suite(args.cases)
    print(f"wrote {args.cases} smoke-fixture cases to {path}")
    print(
        "NOTE: this is a fixture, not evaluation data, and no human labels were written.\n"
        "      Build the real corpus with:  python -m evalops.build all"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

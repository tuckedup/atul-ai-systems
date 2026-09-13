"""Calibrate judge decisions against committed human labels."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from aisys.evals import calibrate


def from_csv(path: str | Path) -> dict[str, Any]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 50:
        raise ValueError("judge calibration requires at least 50 human labels")
    report = calibrate([float(row["judge_score"]) for row in rows], [float(row["human_score"]) for row in rows])
    if report["kappa"] < 0.6:
        raise ValueError(f"judge calibration failed: kappa={report['kappa']:.3f}")
    return report


if __name__ == "__main__":
    print(from_csv(Path(__file__).parent / "human_labels" / "labels.csv"))


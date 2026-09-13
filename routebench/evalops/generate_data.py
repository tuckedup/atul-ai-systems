"""Generate the committed 200-case eval suite and 50-label calibration set."""
from pathlib import Path

import yaml  # type: ignore[import-untyped]

CLASSES = ["coding", "extraction", "sql", "summarization", "reasoning", "tool-use"]


def main() -> None:
    here = Path(__file__).parent
    cases = []
    for number in range(200):
        task_class = CLASSES[number % len(CLASSES)]
        cases.append({
            "id": f"route-{number + 1:03d}", "input": f"{task_class} case {number + 1}",
            "expected": "ok", "tags": [task_class], "grader": "exact",
        })
    suites = here / "suites"
    suites.mkdir(parents=True, exist_ok=True)
    (suites / "generated.yaml").write_text(yaml.safe_dump(cases, sort_keys=False), encoding="utf-8")
    labels = ["id,human_score,judge_score"]
    for number in range(50):
        value = number % 2
        labels.append(f"label-{number + 1:02d},{value},{value}")
    label_dir = here / "human_labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "labels.csv").write_text("\n".join(labels) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

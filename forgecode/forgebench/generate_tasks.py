"""Generate the deterministic 100-case ForgeBench manifest."""
from pathlib import Path

import yaml

TAGS = ["bugfix", "feature", "tests", "refactor", "api-migration", "dep-error", "multi-file", "ambiguous"]
TASKS = [
    ("py-auth", "refresh tokens intermittently fail exactly on expiry; find and fix the boundary", "auth.py"),
    ("ts-slug", "slug consecutive spaces create repeated dashes; normalize the separator", "src/slug.ts"),
]


def main() -> None:
    cases = []
    for number in range(100):
        fixture, task, expected_file = TASKS[number % len(TASKS)]
        cases.append({
            "id": f"forge-{number + 1:03d}",
            "input": {"fixture": fixture, "task": task},
            "expected": expected_file,
            "tags": [TAGS[number % len(TAGS)]],
            "grader": "contains",
        })
    target = Path(__file__).parent / "tasks" / "generated.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(cases, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()


from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "eval-gate.yml"


def test_eval_gate_runs_supported_deterministic_comparison():
    text = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)

    assert "forgebench-regression" in workflow["jobs"]
    assert "forgecode/forgebench/run.py --output .local/forgebench-candidate.json" in text
    assert "aisys evals-compare" in text
    assert "--max-drop 0.02" in text
    assert "--subset" not in text
    assert "--out " not in text
    assert "postgres" not in text.lower()
    assert "OPENAI_API_KEY" not in text

from pathlib import Path

from forgecode.context.pack import build

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "forgecode" / "forgebench" / "fixtures"


def test_context_never_exceeds_budget_and_retrieves_bug_files():
    cases = [
        (FIXTURES / "py-auth", "refresh tokens intermittently fail on expiry", "auth.py"),
        (FIXTURES / "ts-slug", "slug consecutive spaces create repeated dashes", "src\\slug.ts"),
    ]
    hits = 0
    for fixture, task, expected in cases:
        pack = build(fixture, task, budget_tokens=300)
        assert pack.tokens <= 300
        normalized = [name.replace("/", "\\") for name in pack.files]
        hits += expected in normalized
    assert hits / len(cases) >= 0.8

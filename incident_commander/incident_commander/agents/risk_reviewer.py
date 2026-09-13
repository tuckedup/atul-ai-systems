from .scoping import prompt

NAME = "risk_reviewer"


def render_prompt(context: dict[str, object]) -> str:
    return prompt(NAME, context)

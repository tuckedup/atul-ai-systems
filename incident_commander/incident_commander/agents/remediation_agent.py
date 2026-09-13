from .scoping import prompt

NAME = "remediation_agent"


def render_prompt(context: dict[str, object]) -> str:
    return prompt(NAME, context)


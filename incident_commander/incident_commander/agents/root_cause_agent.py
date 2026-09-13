from .scoping import prompt

NAME = "root_cause_agent"


def render_prompt(context: dict[str, object]) -> str:
    return prompt(NAME, context)


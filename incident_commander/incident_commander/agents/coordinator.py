from .scoping import prompt

NAME = "coordinator"


def render_prompt(context: dict[str, object]) -> str:
    return prompt(NAME, context)


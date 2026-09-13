from .scoping import prompt

NAME = "logs_agent"


def render_prompt(context: dict[str, object]) -> str:
    return prompt(NAME, context)


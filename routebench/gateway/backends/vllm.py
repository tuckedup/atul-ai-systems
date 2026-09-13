from .base import OpenAICompatibleBackend


class VLLMBackend(OpenAICompatibleBackend):
    """Self-hosted OpenAI-wire adapter; live verification is Docker-blocked on this host."""


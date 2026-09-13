"""Incident Commander settings — Degraded Mode: SQLite + mock Prometheus."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database — SQLite by default, Postgres when available
    database_url: str = "sqlite:///.local/ic.db"

    # aisys-core LLM settings (inherited)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    default_model: str = "gpt-4o-mini"

    # Prometheus — mock in Degraded Mode
    prometheus_url: str = "mock://prometheus"
    prometheus_mode: str = "mock"  # "real" or "mock"

    # GitHub integration
    github_token: str = ""
    github_repo: str = ""

    # Service catalog
    service_catalog_path: str = "demo/scenarios/services.yaml"


settings = Settings()

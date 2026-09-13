from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AISYS_", extra="ignore")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("AISYS_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
    )  # point at RouteBench: http://localhost:8080/v1
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("AISYS_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    default_model: str = "gpt-4o-mini"
    fallback_models: list[str] = []
    database_url: str = "sqlite:///.local/aisys.db"
    cache_backend: str = "memory"
    trace_jsonl_path: str = ".local/traces.jsonl"
    service_name: str = "aisys"
    request_timeout_s: float = 60.0
    max_retries: int = 3
    operator_sre_token: str = "local-sre-token"
    operator_viewer_token: str = "local-viewer-token"


settings = Settings()

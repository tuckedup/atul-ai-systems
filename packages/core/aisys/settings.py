from pydantic_settings import BaseSettings, SettingsConfigDict

from .redis_dict import RedisDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- single source of truth for the DB DSN ----
    # Tomorrow's swap to Postgres is a one-line change in .env (or here as fallback).
    # The default is SQLite; nowhere else hardcodes sqlite.
    database_url: str = "sqlite:///.local/aisys.db"

    # ---- Redis substitute: in-process dict behind the same interface ----
    redis_url: str = "redis://localhost:6379/0"  # informational; real impl ignores it
    _redis_cache: RedisDict | None = None

    @property
    def redis(self) -> RedisDict:
        if self._redis_cache is None:
            self._redis_cache = RedisDict()
        return self._redis_cache

    # ---- OpenAI / RouteBench ----
    openai_base_url: str = "https://api.openai.com/v1"  # RouteBench: http://localhost:8080/v1
    openai_api_key: str = ""
    default_model: str = "gpt-4o-mini"
    fallback_models: list[str] = []

    # ---- tracing ----
    # When Phoenix / OTel collector is available, OTLP_ENDPOINT points at it and
    # init_tracing(exporter=...) is called with an OTLPSpanExporter. Otherwise the
    # default path (ConsoleSpanExporter + JSONL) is used.
    otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "aisys"

    request_timeout_s: float = 60.0
    max_retries: int = 3


settings = Settings()

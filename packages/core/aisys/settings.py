from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_base_url: str = "https://api.openai.com/v1"   # point at RouteBench: http://localhost:8080/v1
    openai_api_key: str = ""
    default_model: str = "gpt-4o-mini"
    fallback_models: list[str] = []
    # One-line swap to Postgres: set DATABASE_URL=postgresql://aisys:aisys@localhost:5432/aisys
    database_url: str = "sqlite:///.local/aisys.db"
    # One-line swap to Redis: set REDIS_URL=redis://localhost:6379/0
    redis_url: str = "local://dict"
    # OTel: ConsoleSpanExporter + JSONL file. Swap to OTLP: set OTEL_ENDPOINT=http://localhost:4317
    otlp_endpoint: str = "console+jsonl://.local/traces.jsonl"
    service_name: str = "aisys"
    request_timeout_s: float = 60.0
    max_retries: int = 3


settings = Settings()

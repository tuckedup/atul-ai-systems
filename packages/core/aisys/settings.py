from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_base_url: str = "https://api.openai.com/v1"   # point at RouteBench: http://localhost:8080/v1
    openai_api_key: str = ""
    default_model: str = "gpt-4o-mini"
    fallback_models: list[str] = []
    database_url: str = "postgresql://aisys:aisys@localhost:5432/aisys"
    redis_url: str = "redis://localhost:6379/0"
    otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "aisys"
    request_timeout_s: float = 60.0
    max_retries: int = 3


settings = Settings()

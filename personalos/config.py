"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Database
    database_url: str = "postgresql://user:password@localhost:5432/personalos"
    database_echo: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_workers: int = 4

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Observability
    log_level: str = "INFO"
    jaeger_enabled: bool = False
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831

    # MCP Servers
    mcp_files_enabled: bool = True
    mcp_google_enabled: bool = False
    mcp_jobs_enabled: bool = True

    # Google Integration
    google_api_key: str = ""
    google_search_engine_id: str = ""

    # Application
    app_name: str = "PersonalOS"
    app_env: str = "development"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()

"""Application configuration."""

from pydantic import field_validator
from pydantic_settings import BaseSettings

#: The PostgreSQL DBAPI this project actually installs (`psycopg2-binary`, see
#: `pyproject.toml`).
#:
#: A `postgresql://` URL that does not name a driver is resolved by SQLAlchemy
#: against a built-in default, and that default moved: 2.0 resolved it to
#: psycopg2, 2.1 resolves it to psycopg (v3). Since `personalos.persistence`
#: calls `create_engine` at import time, and `create_engine` imports the DBAPI
#: to load the dialect, a bare URL under SQLAlchemy 2.1 fails the import of the
#: whole persistence package with `ModuleNotFoundError: No module named
#: 'psycopg'` -- taking every module that transitively imports it down with it.
#:
#: So the URL names its driver rather than inheriting one. That keeps a given
#: URL meaning the same thing across SQLAlchemy versions, which an upper pin on
#: `sqlalchemy` would not: a pin would only defer the same break to whenever it
#: was next raised.
POSTGRES_DRIVER = "psycopg2"


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Database
    database_url: str = f"postgresql+{POSTGRES_DRIVER}://user:password@localhost:5432/personalos"
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

    # Credentials. These name *where* secrets are, never the secrets: refresh
    # tokens, API keys and the OAuth client secret live in the OS keychain
    # under `secret_store_service`, filed by credential reference.
    secret_store_service: str = "personalos"
    google_oauth_client_id: str = ""
    google_oauth_client_secret_ref: str = "cred://google-oauth-client/default"

    # Application
    app_name: str = "PersonalOS"
    app_env: str = "development"

    @field_validator("database_url")
    @classmethod
    def _name_the_postgres_driver(cls, value: str) -> str:
        """Rewrite a bare `postgresql://` URL to name `POSTGRES_DRIVER`.

        Applied to the configured value, not just the default, because the URL
        usually arrives from `DATABASE_URL` -- a deployment env var, a CI
        service block, a copied `.env.example` -- and those are conventionally
        written in the bare form. Normalizing here means every caller
        (`persistence.database`, `migrations/env.py`) gets a URL whose driver is
        installed, without each of them having to know about it.

        A URL that already names a driver (`postgresql+asyncpg://`, say) is left
        exactly as written: naming one is an explicit choice, and overriding it
        would be the more surprising behaviour.
        """
        prefix = "postgresql://"
        if value.startswith(prefix):
            return f"postgresql+{POSTGRES_DRIVER}://{value[len(prefix):]}"
        return value

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()

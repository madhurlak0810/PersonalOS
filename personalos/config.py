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
    # `json` (one object per line, for shipping) or `text` (for a terminal).
    # See `personalos.observability.structured_logging`.
    log_format: str = "json"
    # Tracing (OpenTelemetry). Off unless enabled; with no endpoint, spans are
    # created and propagated but exported nowhere. The endpoint is an OTLP/HTTP
    # collector, e.g. http://localhost:4318 for Jaeger's all-in-one image.
    # See `personalos.observability.tracing_setup`.
    tracing_enabled: bool = False
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = ""

    # Models. A `ModelConfig` JSON file naming the provider and model for each
    # model-backed port (see `personalos.models.gateway` and
    # `evals/model_benchmarks/configs/`). Empty runs every port on the
    # deterministic rules. Evaluate a change with `python -m evals.harness
    # --config <file>` before pointing this at it.
    llm_gateway_config: str = ""

    # MCP Servers
    mcp_files_enabled: bool = True
    mcp_google_enabled: bool = False
    mcp_jobs_enabled: bool = True

    # File tools. The only directories the files MCP server may read or write,
    # as absolute paths; the first is the one relative paths resolve against.
    # Empty means the file tools reject every path. Set from the environment
    # as a JSON list: FILES_ALLOWED_ROOTS=["/home/me/job-search"]
    files_allowed_roots: list[str] = []

    # Job providers. Greenhouse boards to watch, as board token -> employer
    # name; a board belongs to one company, so there is no global search.
    # Set as JSON: GREENHOUSE_BOARDS={"acme": "Acme"}
    greenhouse_boards: dict[str, str] = {}

    # Application lifecycle. An application with no activity for this many
    # days is moved to STALLED by `apps.worker.stall_monitor`, which a
    # scheduler runs every `application_stall_check_interval_seconds`.
    application_stall_window_days: int = 14
    application_stall_check_interval_seconds: int = 3600

    # Worker. How often the worker polls for queued workflow commands and
    # outbox events when it finds none, and how often it sweeps for
    # follow-up checkpoints and approvals that have come due. These are
    # polling cadences: what is due is read from the database each time.
    # `worker_graphs` names the factory that composes the graphs the worker
    # runs, as `module:callable`; see `apps.worker.main.WorkerGraphs`.
    worker_idle_seconds: float = 1.0
    worker_checkpoint_sweep_seconds: int = 60
    worker_approval_expiry_sweep_seconds: int = 300
    worker_graphs: str = ""

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
            return f"postgresql+{POSTGRES_DRIVER}://{value[len(prefix) :]}"
        return value

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()

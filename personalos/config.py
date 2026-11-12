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

    # API. Loopback by default. Every route but `/health` needs a bearer token
    # (see `apps.api.auth`); the API refuses to bind a non-loopback address
    # with API_AUTH_ENABLED=false.
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_workers: int = 4
    # API authentication. On by default; turning it off makes every request
    # the API_ACTOR and is only allowed on a loopback address.
    api_auth_enabled: bool = True
    # Where the local API token lives: `file` (a 0600 file at API_TOKEN_FILE)
    # or `keychain` (the OS keychain under SECRET_STORE_SERVICE). It is created
    # on first run; `personalos.cli auth token` prints it, `--rotate` replaces it.
    api_token_store: str = "file"
    # Empty means $XDG_CONFIG_HOME/personalos/api-token (~/.config/...).
    api_token_file: str = ""
    # The actor the local token authenticates as: whose profile, evidence and
    # approvals a request touches. Matches the CLI's default --actor.
    api_actor: str = "api"
    # How long a stream ticket (`POST /v1/auth/stream-tickets`) can be used to
    # open a workflow's event stream.
    api_stream_ticket_seconds: int = 60
    # Browser origins allowed to call the API cross-origin (CORS), as exact
    # `scheme://host[:port]` values in a JSON list. The default is only the
    # local dev UI. A wildcard is refused at startup; see `apps.api.security`.
    #   API_CORS_ORIGINS=["http://localhost:5173"]
    api_cors_origins: list[str] = ["http://localhost:5173"]
    # The largest request body the API reads, in bytes; bigger is a 413.
    api_max_body_bytes: int = 1024 * 1024

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

    # Embeddings. Resume ingestion stores evidence vectors with this embedder,
    # and artifact prep embeds postings with it. It is one setting because
    # vectors from different embedders are not comparable
    # (`personalos.models.embedding`).
    #   ollama   a local embedding model served by Ollama (the default; needs
    #            `ollama pull nomic-embed-text` and the `local` extra)
    #   hashing  the deterministic feature-hashing embedder: offline, no download
    #   openai   the hosted OpenAI embeddings API; refused unless
    #            CLOUD_INFERENCE_ALLOWED is true
    # Empty leaves evidence retrieval unconfigured. An `embedding` block in the
    # LLM_GATEWAY_CONFIG file takes precedence over these settings. Changing the
    # embedder means re-embedding what is stored: `personalos.cli reembed`.
    embedding_provider: str = "ollama"
    # The provider's model; empty uses the provider's default
    # (`nomic-embed-text`, `text-embedding-3-small`).
    embedding_model: str = ""
    # Stored as `embedding_version` on every vector row. Empty uses the
    # model's tag (`latest` when the name has none).
    embedding_model_version: str = ""
    # Server URL for `ollama` or `openai`; empty uses the client's default.
    embedding_base_url: str = ""
    # Environment variable holding the `openai` API key.
    embedding_api_key_env: str = "OPENAI_API_KEY"

    # Privacy. False (the default) keeps resume and email text on this
    # machine: a hosted embedder cannot be selected. Set it to true to
    # allow hosted inference.
    cloud_inference_allowed: bool = False

    # MCP Servers
    mcp_files_enabled: bool = True
    # The Google (Gmail and Calendar) MCP server. Off until a Google account
    # is connected (`personalos.cli auth google`); it is registered
    # in-process, and can also be served over stdio (`personalos.cli
    # mcp-google`). It never listens on a port.
    mcp_google_enabled: bool = False
    # Which connected Google account the Google tools act on, by address.
    # Empty means the most recently connected active account.
    google_account: str = ""
    # The IANA zone (e.g. "America/New_York") calendar events are written in
    # and all-day events are read in. Required for the calendar to be wired:
    # times are never interpreted in the server's local zone.
    calendar_time_zone: str = ""
    # Which calendar interview events and prep blocks go on.
    calendar_id: str = "primary"

    # Gmail recruiter inbox (`personalos.executor.gmail_inbox`). The mailbox
    # scope: only messages carrying one of these labels, or matching one of
    # these Gmail search queries, are ever looked at; everything else is
    # ignored and nothing of it is stored. Both empty (the default) leaves the
    # poller off. Set as JSON lists:
    #   GMAIL_RECRUITER_LABELS=["Recruiting"]
    #   GMAIL_RECRUITER_QUERIES=["from:greenhouse.io", "subject:interview"]
    gmail_recruiter_labels: list[str] = []
    gmail_recruiter_queries: list[str] = []
    # How often the worker polls the Gmail history for new in-scope mail.
    gmail_poll_interval_seconds: float = 300.0
    # How far back a first poll (or a resync after a scope change or an
    # expired cursor) searches for in-scope mail. 0 means only new mail.
    gmail_backfill_days: int = 14
    # Whose applications inbound mail is matched against: the same actor the
    # API token authenticates as (API_ACTOR) and the CLI takes as --actor.
    gmail_inbox_actor: str = "api"

    # File tools. The only directories the files MCP server may read or write,
    # as absolute paths; the first is the one relative paths resolve against.
    # Empty means the file tools reject every path. Set from the environment
    # as a JSON list: FILES_ALLOWED_ROOTS=["/home/me/job-search"]
    files_allowed_roots: list[str] = []

    # Job providers. Greenhouse boards to watch, as board token -> employer
    # name; a board belongs to one company, so there is no global search.
    # Set as JSON: GREENHOUSE_BOARDS={"acme": "Acme"}
    greenhouse_boards: dict[str, str] = {}
    # Lever sites (the slug in jobs.lever.co/<site>; prefix `eu/` for an
    # EU-hosted site) and Ashby boards (jobs.ashbyhq.com/<board>), each to the
    # employer's display name. Both are public, documented read-only APIs.
    #   LEVER_SITES={"acme": "Acme", "eu/globex": "Globex"}
    #   ASHBY_BOARDS={"initech": "Initech"}
    lever_sites: dict[str, str] = {}
    ashby_boards: dict[str, str] = {}
    # Workday career sites, as "<tenant>.wd<N>.myworkdayjobs.com/<site>" ->
    # employer name. Opt-in per employer: this reads the career page's own
    # undocumented endpoint, not a published API (see personalos/providers/
    # workday.py and "Job board coverage" in docs/ARCHITECTURE_BOUNDARIES.md).
    #   WORKDAY_SITES={"acme.wd5.myworkdayjobs.com/AcmeCareers": "Acme"}
    workday_sites: dict[str, str] = {}

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
    # runs, as `module:callable`; see `apps.worker.main.WorkerGraphs`. The
    # default runs the Supervisor and Job Search; empty runs no graphs.
    worker_idle_seconds: float = 1.0
    worker_checkpoint_sweep_seconds: int = 60
    worker_approval_expiry_sweep_seconds: int = 300
    # How often the worker checks every integration's health (Gmail, Calendar,
    # the Google MCP server, job boards, the embedder, local model servers).
    integration_health_check_seconds: float = 300.0
    # How long work held back by an open circuit -- it needs an integration
    # that is down or awaiting a reconnect -- waits before it is looked at again.
    integration_circuit_retry_seconds: float = 60.0
    worker_graphs: str = "apps.worker.graphs:build_worker_graphs"

    # Credentials. These name *where* secrets are, never the secrets: refresh
    # tokens, API keys and the OAuth client secret live in the OS keychain
    # under `secret_store_service`, filed by credential reference.
    secret_store_service: str = "personalos"
    google_oauth_client_id: str = ""
    google_oauth_client_secret_ref: str = "cred://google-oauth-client/default"
    # `testing` (the default) or `production`: the OAuth client's publishing
    # status in the Google Cloud console. An unverified app in Testing is
    # limited to its listed test users and gets 7-day refresh tokens, so a
    # connection made through it is recorded as expiring after a week.
    google_oauth_publishing_status: str = "testing"

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

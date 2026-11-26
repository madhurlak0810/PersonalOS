# syntax=docker/dockerfile:1.7
#
# PersonalOS API and worker images. One Dockerfile, several targets:
#
#   api                the FastAPI server                       (docker-compose.yml)
#   worker             the worker: workflow runs, outbox, sweeps (docker-compose.yml)
#   worker-quickstart  the worker plus the fake job board and the sample resume
#                      and profile from docs/quickstart          (compose.quickstart.yml)
#
# `api` and `worker` carry no tests, no test fakes, no dev dependencies and
# no secrets: the build context is an allow-list (.dockerignore), the
# dependency install refuses the `dev` and `test` extras, and the two fake
# modules are deleted. Secrets reach a container at run time, as Docker
# secrets or files on a volume (docs/CONTAINERS.md), never at build time.
#
# Both run as the unprivileged user `personalos` (uid 10001), with the source
# owned by root and read-only to them, so the root filesystem can be mounted
# read-only (docker-compose.yml does). The only writable paths are the
# volume at /var/lib/personalos and a tmpfs /tmp.

# Pinned by tag and digest: a rebuild gets the same base until this changes.
ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

# --- Dependencies -------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build
COPY pyproject.toml ./

# Optional extras to install, comma-separated. `local` backs the Ollama
# embedder and model providers (the compose `ollama` profile); `llm` the
# Anthropic intent classifier. `dev` and `test` are refused.
ARG PERSONALOS_EXTRAS=local

# Only the project's runtime dependencies, read from pyproject.toml. The
# project itself is not pip-installed: its source is copied below, so what
# runs is exactly what is in the repository.
RUN python - <<'EOF'
import os
import tomllib

project = tomllib.load(open("pyproject.toml", "rb"))["project"]
requirements = list(project["dependencies"])
for extra in filter(None, (e.strip() for e in os.environ.get("PERSONALOS_EXTRAS", "").split(","))):
    if extra in ("dev", "test"):
        raise SystemExit(f"the '{extra}' extra is not allowed in an image")
    requirements += project["optional-dependencies"][extra]
open("requirements.txt", "w").write("\n".join(requirements) + "\n")
EOF
RUN pip install -r requirements.txt

# --- Shared runtime -----------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="PersonalOS" \
      org.opencontainers.image.licenses="Apache-2.0"

# Container defaults. Secrets are files (docs/CONTAINERS.md), not the OS
# keychain, which a container does not have.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/var/lib/personalos \
    SECRET_STORE_BACKEND=file \
    SECRET_STORE_DIR=/var/lib/personalos/secrets \
    SECRET_STORE_READ_ONLY_DIRS='["/run/secrets"]' \
    API_TOKEN_STORE=file \
    API_TOKEN_FILE=/var/lib/personalos/secrets/api-token \
    LOG_FORMAT=json

# /var/lib/personalos/secrets is the writable secret store (0700).
# /var/lib/personalos/db is where the compose stack's `db-password` service
# writes the generated database password; 0755 so a fresh volume mounted there
# inherits it and the Postgres container's own user can read the password.
RUN groupadd --system --gid 10001 personalos \
    && useradd --system --uid 10001 --gid personalos \
        --home-dir /var/lib/personalos --no-create-home --shell /usr/sbin/nologin personalos \
    && install -d -o personalos -g personalos -m 0700 \
        /var/lib/personalos /var/lib/personalos/secrets \
    && install -d -o personalos -g personalos -m 0755 /var/lib/personalos/db

COPY --from=deps /opt/venv /opt/venv

WORKDIR /app
COPY LICENSE NOTICE alembic.ini ./
COPY migrations ./migrations
COPY personalos ./personalos
COPY apps ./apps
COPY mcp_servers ./mcp_servers

# Test fakes stay out of the production images; `worker-quickstart` adds
# them back. Bytecode is compiled now, since the read-only root filesystem
# means it cannot be written at run time.
RUN rm -f apps/worker/fake_providers.py personalos/providers/fake.py \
    && python -m compileall -q /app

USER 10001:10001

# --- API ----------------------------------------------------------------------
FROM runtime AS api

# Inside the container the API listens on every interface so the published
# port can reach it; docker-compose.yml publishes that port on 127.0.0.1
# only. The API refuses a non-loopback bind unless authentication is on.
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=1
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import os, sys, urllib.request; sys.exit(0 if urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('API_PORT', '8000')}/health\", timeout=4).status == 200 else 1)"]

CMD ["python", "-m", "personalos.cli", "api"]

# --- Worker -------------------------------------------------------------------
FROM runtime AS worker

ENV WORKER_HEARTBEAT_FILE=/tmp/personalos-worker.heartbeat \
    WORKER_HEARTBEAT_SECONDS=10

HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=4 \
    CMD ["python", "-m", "personalos.cli", "worker-health"]

CMD ["python", "-m", "personalos.cli", "worker"]

# --- Worker for the quick start -----------------------------------------------
# The worker with the fake job board and the extractive draft writer
# (apps/worker/fake_providers.py) and the sample resume and profile, for the
# containerized quick start and the CI compose smoke test. Not for real use.
FROM worker AS worker-quickstart

COPY personalos/providers/fake.py ./personalos/providers/fake.py
COPY apps/worker/fake_providers.py ./apps/worker/fake_providers.py
COPY docs/quickstart ./docs/quickstart
USER root
RUN python -m compileall -q personalos/providers/fake.py apps/worker/fake_providers.py
USER 10001:10001

ENV WORKER_GRAPHS=apps.worker.fake_providers:build_worker_graphs \
    FILES_ALLOWED_ROOTS='["/app/docs/quickstart"]' \
    EMBEDDING_PROVIDER=hashing

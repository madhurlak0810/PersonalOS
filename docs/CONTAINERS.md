# Containers

PersonalOS ships a `Dockerfile` for the API and the worker, and a
`docker-compose.yml` that runs them with PostgreSQL (with pgvector), the
migrations and an OpenTelemetry collector. Ollama, the operator UI and the
monitoring stack (Prometheus, Alertmanager, Grafana) are optional, behind
compose profiles. CI builds the images and runs the stack end
to end (`scripts/smoke_compose.py`).

The local, non-container path in
[IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md#local-quick-start) is
unchanged and still keeps secrets in the OS keychain.

- [Start the stack](#start-the-stack)
- [What runs](#what-runs)
- [Security defaults](#security-defaults)
- [Secrets in containers](#secrets-in-containers)
- [Configuration](#configuration)
- [Optional services](#optional-services)
- [State, volumes and reset](#state-volumes-and-reset)
- [What CI checks](#what-ci-checks)
- [Known limits](#known-limits)

## Start the stack

You need Docker with Compose v2.24 or newer. Nothing else is needed: no Python,
no PostgreSQL and no API keys.

```bash
docker compose up -d --build
docker compose ps                        # postgres, api and worker: (healthy)
export TOKEN=$(docker compose exec -T api python -m personalos.cli auth token)
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/v1/integrations | jq
```

On the first `up`, the stack generates a database password, initializes
PostgreSQL, runs `alembic upgrade head` and then starts the API and the worker.
The API creates its token on first start, and `auth token` prints it.

With no job board configured, the worker runs on the real wiring with every
board unbound. To run a workflow with nothing configured, use the quick start
override, which swaps in the fake job board. The guide's
[containerized quick start](../IMPLEMENTATION_GUIDE.md#containerized-quick-start)
walks through it.

```bash
docker compose -f docker-compose.yml -f compose.quickstart.yml up -d --build
```

## What runs

| Service | Image | Network | Host port | Health check |
| --- | --- | --- | --- | --- |
| `db-password` | `Dockerfile` target `api` | none | none | one-shot; exits 0 |
| `postgres` | `pgvector/pgvector:0.8.1-pg16` | `backend` | **none** | `pg_isready` over TCP |
| `migrate` | `Dockerfile` target `api` | `backend` | none | one-shot; exits 0 |
| `api` | `personalos-api` (`Dockerfile` target `api`) | `backend`, `egress` | `127.0.0.1:8000` | `GET /health` |
| `worker` | `personalos-worker` (target `worker`) | `backend`, `egress` | none | heartbeat file younger than 30 s (`personalos.cli worker-health`) |
| `otel-collector` | `otel/opentelemetry-collector-contrib:0.139.0` | `backend` | none | see [Known limits](#known-limits) |
| `ollama` (profile) | `ollama/ollama:0.12.10` | `backend`, `egress` | none | `ollama list` |
| `ui` (profile) | `personalos-ui` (`ui/Dockerfile`) | `egress` | `127.0.0.1:5173` | `GET /` |
| `prometheus` (profile `monitoring`) | `prom/prometheus:v3.7.3` | `backend` | none | `GET /-/ready` |
| `alertmanager` (profile `monitoring`) | `prom/alertmanager:v0.29.0` | `backend` | none | `GET /-/ready` |
| `alert-receiver` (profile `monitoring`) | `Dockerfile` target `api` | `backend` | none | `GET /health` |
| `grafana` (profile `monitoring`) | `grafana/grafana:12.2.1` | `backend`, `egress` | `127.0.0.1:3000` | `GET /api/health` |

Startup order follows readiness, not just start order. `postgres` waits for
`db-password` to exit 0. `migrate` waits for `postgres` to be healthy. `api` and
`worker` wait for `migrate` to exit 0 and for `postgres` to be healthy. `ui`
waits for `api` to be healthy. If a migration fails, the API and worker don't
start, and `docker compose up` reports it.

`backend` is an `internal` network with no route out of the host. Only the
API, the worker and Ollama are also on `egress`, because they need it for job
boards, Google APIs and model downloads, and because a published port needs a
non-internal network.

Every base image is pinned by tag and digest, in `Dockerfile`, `ui/Dockerfile`
and `docker-compose.yml`. To upgrade one, change both the tag and the digest
together.

## Security defaults

- **Only loopback is published.** The API is published on `127.0.0.1:8000`,
  the UI on `127.0.0.1:5173` and, with the `monitoring` profile, Grafana on
  `127.0.0.1:3000`. Inside its container the API listens on `0.0.0.0` so that
  the port mapping can reach it. It refuses that bind unless
  `API_AUTH_ENABLED=true`, which is the default. Grafana serves read-only
  dashboards to anonymous viewers, with no login form and no admin over HTTP.
  The dashboards hold only metrics, which carry ids and code-defined names and
  never user content (METRICS.md).
- **PostgreSQL has no host port.** It is reachable only on the internal
  network. To inspect it, run
  `docker compose exec postgres psql -U personalos`.
- **No default database password.** `db-password` writes a random one into the
  `db-password` volume on first run. PostgreSQL reads it with
  `POSTGRES_PASSWORD_FILE` and the app reads it with `DATABASE_PASSWORD_FILE`.
  No compose file, env file or image contains it.
- **Non-root, read-only.** The API and worker run as `personalos` (uid 10001).
  Their source is owned by root, and their root filesystem is mounted
  read-only. The only writable paths are the volumes and a 64 MB tmpfs `/tmp`.
  All capabilities are dropped and `no-new-privileges` is set. The UI runs as
  nginx's unprivileged user (uid 101) under the same restrictions.
- **Nothing secret is in an image.** The build context is an allow-list
  (`.dockerignore` starts with `*`), so a local `.env`, an API token or a
  virtualenv never reaches the builder. The dependency install refuses the
  `dev` and `test` extras. The fake job board (`personalos/providers/fake.py`)
  and the quick-start graph factory (`apps/worker/fake_providers.py`) are
  deleted from the `api` and `worker` images. Only `worker-quickstart` has
  them.

## Secrets in containers

A container has no OS keychain. The containers use `FileSecretStore`
(`personalos/secrets/store.py`) instead, through the same `SecretStore`
interface the keychain store implements. The reference model doesn't change:
the database stores only `cred://provider/name` references, and the store turns
a reference into a value when the credential broker asks.

The image sets `SECRET_STORE_BACKEND=file`. Each secret is a file named after
its reference, as `<provider>.<name>`:

| Reference | File |
| --- | --- |
| `cred://google-oauth-client/default` | `google-oauth-client.default` |
| `cred://google/alice@example.com` | `google.alice@example.com` |
| `cred://greenhouse/default` | `greenhouse.default` |

The store looks in two places:

1. **`SECRET_STORE_DIR`** (`/var/lib/personalos/secrets`, on the
   `personalos-secrets` volume) is writable, and it is checked first. Secrets
   that PersonalOS writes itself go here, such as a refresh token after an
   OAuth connect. Files are written `0600` and atomically. A file there that
   group or others can read is refused.
2. **`SECRET_STORE_READ_ONLY_DIRS`** (`["/run/secrets"]`) are only read. This
   is where Docker secrets are mounted. PersonalOS never writes or deletes
   anything here.

### Provide a secret with Docker secrets

Put the value in a file outside the repository, and add a
`compose.override.yml`. Compose picks this file up automatically.

```yaml
# compose.override.yml
services:
  api:
    secrets:
      - source: google-oauth-client
        target: google-oauth-client.default   # = cred://google-oauth-client/default
  worker:
    secrets:
      - source: google-oauth-client
        target: google-oauth-client.default
secrets:
  google-oauth-client:
    file: ~/.config/personalos/google-oauth-client-secret
```

Without Swarm, Compose bind-mounts the file and keeps its host owner and mode,
so uid 10001 must be able to read it. Either
`sudo chown 10001 <file> && chmod 0400 <file>`, or `chmod 0444 <file>` on a
single-user machine.

### Provide a secret as a mounted directory

To manage secrets with the CLI on the host, point the CLI at a directory
instead of the keychain, then mount that directory:

```bash
SECRET_STORE_BACKEND=file SECRET_STORE_DIR=$HOME/.config/personalos/container-secrets \
  python -m personalos.cli auth google-client client_secret.json
```

```yaml
# compose.override.yml
services:
  api:    { volumes: ["~/.config/personalos/container-secrets:/run/secrets:ro"] }
  worker: { volumes: ["~/.config/personalos/container-secrets:/run/secrets:ro"] }
```

The CLI writes those files as `0600` and owned by you, so uid 10001 has to be
given read access, the same as for a Docker secret. This works for any
command that only writes to the secret store, such as `auth google-client`.
It doesn't work for a command that also records something in the database,
such as `auth google`, because the stack's database has no host port.

### The API token

The API token is a `0600` file on the same volume
(`API_TOKEN_FILE=/var/lib/personalos/secrets/api-token`). To print it or
rotate it:

```bash
docker compose exec -T api python -m personalos.cli auth token
docker compose exec -T api python -m personalos.cli auth token --rotate
docker compose restart api
```

### The local CLI is unchanged

Outside a container, `SECRET_STORE_BACKEND` defaults to `keychain`, and
`build_secret_store()` uses the OS keychain as before. The file store is never
a fallback. It is used only when it is configured explicitly, so a machine
whose keychain is broken still fails to start rather than writing tokens to
disk.

## Configuration

Containers read settings in this order, with later sources winning:

1. the image's own defaults (the secret store, the API bind);
2. `deploy/compose.defaults.env` (tracked; the hashing embedder, tracing on,
   the UI's CORS origins);
3. `.env.compose` at the repository root, if it exists (gitignored). Put your
   own settings here: `GREENHOUSE_BOARDS`, `LLM_GATEWAY_CONFIG`,
   `CALENDAR_TIME_ZONE`, and so on;
4. the stack wiring in `docker-compose.yml`: `DATABASE_URL`,
   `DATABASE_PASSWORD_FILE`, the secret store, the API token file and the
   collector endpoint. These can't be overridden from an env file.

The repository's `.env`, which the local quick start writes, is not passed to
the containers. Compose reads it only to fill in `${...}` in the compose file,
which is why the compose file's own variables are prefixed
`PERSONALOS_`:

| Variable | Default | Effect |
| --- | --- | --- |
| `PERSONALOS_API_PORT` | `8000` | Host port for the API (still on 127.0.0.1). |
| `PERSONALOS_UI_PORT` | `5173` | Host port for the UI. If you change it, add the new origin to `API_CORS_ORIGINS`. |
| `PERSONALOS_IMAGE_TAG` | `local` | Tag of the images built and run. |

To install the `llm` extra (the Anthropic intent classifier), build with
`--build-arg PERSONALOS_EXTRAS=local,llm`. The default is `local`, which
covers the Ollama and OpenAI-compatible clients.

## Optional services

**Ollama**, for local models and embeddings:

```bash
cat >> .env.compose <<'EOF'
EMBEDDING_PROVIDER=ollama
EMBEDDING_BASE_URL=http://ollama:11434
EOF
docker compose --profile ollama up -d
docker compose exec ollama ollama pull nomic-embed-text
```

Ollama has no host port. Models are kept in the `ollama-models` volume.

**The operator UI**:

```bash
docker compose --profile ui up -d --build
```

Then open http://localhost:5173 and paste the API token. The API's address is
compiled into the bundle and into its Content-Security-Policy, so changing
`PERSONALOS_API_PORT` means rebuilding the UI.

**Dashboards and alerts**: the `monitoring` profile adds Prometheus, which
scrapes the collector and evaluates the alert rules; Alertmanager, which sends
every alert to `alert-receiver` as a log line; and Grafana on
http://127.0.0.1:3000 with the dashboards read-only. Nothing outside the stack
is needed. See [METRICS.md](METRICS.md) and [ALERTS.md](ALERTS.md).

```bash
docker compose --profile monitoring up -d --build
docker compose logs -f alert-receiver
```

| Variable | Default | Effect |
| --- | --- | --- |
| `PERSONALOS_GRAFANA_PORT` | `3000` | Host port for Grafana (still on 127.0.0.1). |
| `PERSONALOS_ALERTMANAGER_CONFIG` | `./deploy/alertmanager/alertmanager.yml` | An Alertmanager config of your own, to add an external channel ([ALERTS.md](ALERTS.md#sending-alerts-elsewhere)). |

**Traces**: the collector prints a summary of each batch to its log
(`docker compose logs -f otel-collector`). To send traces to Jaeger, Tempo
or a vendor, add an exporter in `deploy/otel/collector.yaml`, and add the
`egress` network to the collector if the backend is outside the stack.

## State, volumes and reset

| Volume | Holds |
| --- | --- |
| `personalos_postgres-data` | the database |
| `personalos_db-password` | the generated database password |
| `personalos_personalos-secrets` | the API token and secrets PersonalOS writes (refresh tokens) |
| `personalos_ollama-models` | downloaded Ollama models |
| `personalos_prometheus-data` | metrics history (15 days) |
| `personalos_alertmanager-data` | silences and notification state |
| `personalos_grafana-data` | Grafana's own database (the dashboards themselves are files) |

```bash
docker compose down                     # stop; keep all state
docker compose down -v                  # stop and delete every volume: a fresh install
docker volume rm personalos_postgres-data personalos_db-password   # just the database
docker volume rm personalos_ollama-models                          # just the model cache
```

Delete `postgres-data` and `db-password` together. PostgreSQL reads the
password file only when it initializes an empty data directory. If you delete
just one of them, the database and the app disagree about the password.

Deleting `personalos-secrets` deletes the API token and any connected
accounts' refresh tokens. The references in the database then point at
nothing, and those integrations show as needing a reconnect.

## What CI checks

The `containers` job in `.github/workflows/ci.yml`:

1. builds the `api`, `worker`, `worker-quickstart` and `ui` images;
2. runs `scripts/check_image.py` inside the `api` and `worker` images. It checks
   that the image runs as a non-root user, contains no test fakes, no `tests/`,
   no dev or test packages and no `.env`, has an empty secret directory, and
   still imports the API and the worker;
3. scans every image for secrets with gitleaks: the files the build copied in,
   plus the image config and layer history, which is where an `ENV`, `ARG` or
   `RUN` would leave one;
4. runs `scripts/smoke_compose.py`. It brings up the quick-start stack, waits
   for health, checks that the API is published on 127.0.0.1 only, that
   PostgreSQL has no host port and nothing answers on the host's 5432, that
   the API and worker run as non-root on read-only root filesystems, and that
   the collector is healthy. It then runs the quick start's workflow through
   approval and tears the stack down with `down -v`;
5. runs `promtool` (the rule unit tests and `check config`) and `amtool
   check-config` inside the pinned Prometheus and Alertmanager images;
6. runs `scripts/smoke_observability.py`. It brings the quick-start stack up
   with `--profile monitoring`, checks that every dashboard panel has data,
   and then stalls a workflow to check that `WorkflowStalled` fires, reaches
   the alert receiver, and clears.

`tests/unit/test_container_config.py` checks the same posture statically on
every test run: published ports, networks, pinned images, health checks,
readiness conditions and the build context allow-list.

## Known limits

- **The collector has no container health check.** Its image has no shell or
  HTTP client to run one. The API and worker depend on it being started, not
  healthy. That is safe because an unreachable collector only means spans are
  dropped. Its `health_check` extension answers on `otel-collector:13133`
  inside the network, and CI probes it from the API container.
- **You can't connect a Google account to the stack yet.** The consent flow
  (`auth google`, `POST /v1/integrations/google/connect`) needs the browser to
  reach a loopback redirect listener in the same process. In a container, that
  listener is on the container's own loopback interface, which the browser
  can't reach. The OAuth client secret and API keys can be provided as
  described above. Connecting an account needs the local, non-container setup
  for now.
- **One API process.** The image sets `API_WORKERS=1`. Rate limits are kept
  per process (docs/LIMITS.md), so more workers would multiply them.

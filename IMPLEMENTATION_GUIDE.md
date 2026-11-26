# PersonalOS Implementation Guide

How PersonalOS is put together, how to run it locally, and what is still to be
built.

## Table of Contents

1. [Local Quick Start](#local-quick-start)
2. [Security Posture](#security-posture)
3. [Project Overview](#project-overview)
4. [Project Structure](#project-structure)
5. [Architecture](#architecture)
6. [Built Components](#built-components)
7. [MCP Framework Guide](#mcp-framework-guide)
8. [Testing](#testing)
9. [Next Steps](#next-steps)
10. [Configuration](#configuration)

---

## Local Quick Start

From a fresh clone to a job search waiting on your approval, with no API keys
and no network beyond localhost. The job board is a fake one that serves two
sample postings, and drafts are written by restating your resume verbatim
instead of by a model, so every step below behaves the same on every machine.
`scripts/smoke_quickstart.py` runs these same steps in CI.

Run every command from the repository root: the processes read `.env` from the
current directory.

### Prerequisites

- Python 3.10 or newer
- Docker, for PostgreSQL with the `vector` extension. Any PostgreSQL 16 with
  pgvector works; the `pgvector/pgvector:pg16` image is the one CI uses.
- `curl` and `jq`, for the API calls below

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,test]"
```

### 2. Start PostgreSQL with pgvector

```bash
docker run -d --name personalos-db \
  -e POSTGRES_USER=personalos -e POSTGRES_PASSWORD=personalos -e POSTGRES_DB=personalos \
  -p 127.0.0.1:5432:5432 pgvector/pgvector:pg16
```

The port is published on `127.0.0.1` only. Wait until
`docker exec personalos-db pg_isready -U personalos` reports
`accepting connections`.

### 3. Set the minimum environment

```bash
cat > .env <<EOF
DATABASE_URL=postgresql://personalos:personalos@127.0.0.1:5432/personalos
FILES_ALLOWED_ROOTS=["$PWD/docs/quickstart"]
WORKER_GRAPHS=apps.worker.fake_providers:build_worker_graphs
EMBEDDING_PROVIDER=hashing
LOG_FORMAT=text
EOF
```

| Variable | Why |
| --- | --- |
| `DATABASE_URL` | The database from step 2. |
| `FILES_ALLOWED_ROOTS` | The only directories the file tools may read. The sample resume lives in `docs/quickstart/`. |
| `WORKER_GRAPHS` | Runs the worker on the fake job board and the extractive draft writer (`apps/worker/fake_providers.py`). Leave it unset to use the real wiring, which needs at least one job board (`GREENHOUSE_BOARDS`, `LEVER_SITES`, `ASHBY_BOARDS` or `WORKDAY_SITES`) and `LLM_GATEWAY_CONFIG`. |
| `EMBEDDING_PROVIDER` | The offline `hashing` embedder, so the quick start needs no Ollama. The default is a local Ollama model. |
| `LOG_FORMAT` | Readable logs in a terminal. The default is JSON. |

Everything else keeps its default. See [Configuration](#configuration) for the
full list.

### 4. Run the migrations

```bash
alembic upgrade head
```

### 5. Ingest a resume

```bash
python -m personalos.cli ingest-resume resume.md
```

A relative path resolves against the first allowed root, so this reads
`docs/quickstart/resume.md`. It splits the resume into evidence chunks and
embeds them with the `hashing` embedder set in step 3. Job matching and drafts can
cite only these chunks. The output ends with `"added": 7`. Running it again on
the unchanged file changes nothing.

### 6. Create a search profile

```bash
python -m personalos.cli profile create docs/quickstart/profile.json
```

The sample profile looks for remote backend roles using Python and Postgres.
Both commands act as the actor `api`, the same actor the API token
authenticates as (`API_ACTOR`). That is how the search in step 8 finds this
resume and this profile.

### 7. Start the API and the worker

In two more terminals, each with the virtualenv activated:

```bash
python -m personalos.cli api        # serves on http://127.0.0.1:8000
```

```bash
python -m personalos.cli worker
```

The worker logs `wired graphs from 'apps.worker.fake_providers:build_worker_graphs'`
and then a warning listing its unbound ports (`application_store`,
`application_submitter`, `calendar`, `mailbox`, `reply_drafter`). Those are
the adapters still to be built (see [Next Steps](#next-steps)).

The first `api` run creates the local API token (a `0600` file at
`~/.config/personalos/api-token`). Every route but `/health` needs it, so
put it in the shell you run the next steps from:

```bash
export TOKEN=$(python -m personalos.cli auth token)
export AUTH="Authorization: Bearer $TOKEN"
```

The interactive API docs are at http://127.0.0.1:8000/docs; use
**Authorize** with the token to call routes from there.

### 8. Start a job search

```bash
WORKFLOW_ID=$(curl -s -X POST http://127.0.0.1:8000/v1/chat -H "$AUTH" \
  -H "Content-Type: application/json" \
  -d '{"message": "find me remote python backend jobs"}' | jq -r .workflow_id)
echo $WORKFLOW_ID
```

The API answers `202 Accepted` right away. The worker then runs the
Supervisor and the Job Search subgraph: search the fake board, normalize,
dedupe, filter out the on-site posting, score, check evidence, shortlist, and
draft a resume and cover letter for the top match.

### 9. Reach the approval interrupt

Within a few seconds the workflow stops before the only step with an outside
effect, submitting the application:

```bash
curl -s -H "$AUTH" http://127.0.0.1:8000/v1/workflows/$WORKFLOW_ID | jq '{status, current_step}'
```

```json
{
  "status": "waiting",
  "current_step": {
    "thread_id": "job_search:…",
    "step": "approval_checkpoint_for_external_submission"
  }
}
```

The approval queue shows what approving would do: the recipient, a diff of
the documents, and their evidence citations.

```bash
curl -s -H "$AUTH" http://127.0.0.1:8000/v1/approvals | jq '.approvals[] | {id, action_type, summary, action_hash}'
python -m personalos.cli approvals    # the same queue, rendered for a terminal
```

### 10. Approve the pending action

An approval must quote the `action_hash` it was listed with. If the action
changes after it was listed, the hash no longer matches and the approval is
refused with a `409`.

```bash
APPROVAL=$(curl -s -H "$AUTH" http://127.0.0.1:8000/v1/approvals | jq -c '.approvals[0]')
curl -s -X POST -H "$AUTH" "http://127.0.0.1:8000/v1/approvals/$(jq -r .id <<<"$APPROVAL")/decision" \
  -H "Content-Type: application/json" \
  -d "{\"approved\": true, \"action_hash\": $(jq .action_hash <<<"$APPROVAL")}" | jq
```

The response has `"status": "approved"` and `"resumed": true`. The worker
resumes the run, and the executor runs the approved submission under the
recorded decision. This build has no submitter adapter yet, so the step
fails as a retryable `PortNotConfigured`, the same way it would in a
deployment that has not configured one:

```bash
curl -s -H "$AUTH" http://127.0.0.1:8000/v1/workflows/$WORKFLOW_ID/failures \
  | jq '.failures[] | {step, error_type, last_error}'
curl -s -H "$AUTH" http://127.0.0.1:8000/v1/workflows/$WORKFLOW_ID/audit | jq   # the decision trail
```

That is the end of the walkthrough. To start over, drop the database
(`docker rm -f personalos-db`) and go back to step 2. A second search against
the same database stops at an `IdempotencyConflict` instead, because the
submission's idempotency key is derived from the posting.

### Optional: review it in the browser

Steps 9 and 10 can be done from the approvals inbox in `ui/` instead of with
`curl`. It needs Node.js 20.19 or newer. With the API and worker from step 7
still running and the workflow parked at step 9:

```bash
cd ui
npm ci
npm run dev                         # serves on http://localhost:5173
```

Open http://localhost:5173 and paste the token from
`python -m personalos.cli auth token`. The inbox lists the pending
`submit_application` approval labelled `AWAITING_APPROVAL`, with its target,
risk level, requested scopes, expiry, the documents it would send and the
action's full payload. **Approve** sends the `action_hash` the approval was
listed with; if the action changed or expired since the page loaded, the API
answers `409` and the UI asks you to refresh and review it again rather than
resubmitting.

The API's CORS allow-list (`API_CORS_ORIGINS`) admits `http://localhost:5173`
by default. To point the UI at another API, set `VITE_API_BASE_URL` (see
`ui/.env.example`) and add the UI's origin to `API_CORS_ORIGINS`.

### The same thing in one command

```bash
python scripts/smoke_quickstart.py
```

This runs steps 4–10 against `DATABASE_URL`, which defaults to the database
from step 2 and must be fresh. It starts the API and the worker as real
processes, checks each response, and exits non-zero with both processes' logs
if anything differs from what this guide says. To run it without PostgreSQL,
set `DATABASE_URL=sqlite:////tmp/smoke.db`. That skips Alembic and the
pgvector index, so CI runs it on PostgreSQL.

### Containerized quick start

The same walkthrough with Docker only: no Python, no virtualenv, no `.env`.
See [docs/CONTAINERS.md](docs/CONTAINERS.md) for what the stack runs and why.
`compose.quickstart.yml` swaps in a worker image that has the fake job board,
the extractive draft writer, and the sample resume and profile. The production
`worker` image has none of them.

```bash
docker compose -f docker-compose.yml -f compose.quickstart.yml up -d --build
docker compose ps        # wait until postgres, api and worker are (healthy)
```

This replaces steps 1–4 and 7: the stack generates a database password, runs
the migrations, then starts the API on `127.0.0.1:8000` and the worker. Steps
5 and 6 run inside the worker container. The resume path resolves against
`/app/docs/quickstart`, as it does against `docs/quickstart` locally:

```bash
docker compose exec -T worker python -m personalos.cli ingest-resume resume.md
docker compose exec -T worker python -m personalos.cli profile create /app/docs/quickstart/profile.json
export TOKEN=$(docker compose exec -T api python -m personalos.cli auth token)
export AUTH="Authorization: Bearer $TOKEN"
```

Steps 8–10 are then exactly as written above, against
`http://127.0.0.1:8000`. To review the approval in the browser instead, add
`--profile ui` to the `up` command and open http://localhost:5173. To start
over, run `docker compose -f docker-compose.yml -f compose.quickstart.yml down -v`,
which deletes the database and the token.

`python scripts/smoke_compose.py` does all of this, checks the stack's
security posture along the way, and tears the stack down. CI runs it.

### Optional: connect a Google account

The walkthrough above needs no Google account. Gmail and Calendar adapters
(Phase J2 onwards) need one, and this is how to connect it. You need a
desktop browser on the same machine, because Google redirects to
`127.0.0.1`.

1. In the [Google Cloud console](https://console.cloud.google.com/), create a
   project, enable the Gmail API and the Google Calendar API, and configure
   the OAuth consent screen as **External**, in **Testing** status. Add your
   own Google account under **Test users**.
2. Create an OAuth client ID of type **Desktop app** and download its JSON.
3. Store the client secret in the OS keychain, and put the client ID (which
   is not secret) in `.env`:

   ```bash
   python -m personalos.cli auth google-client --from-json ~/Downloads/client_secret_*.json
   echo 'GOOGLE_OAUTH_CLIENT_ID=<the client id it printed>' >> .env
   ```

4. Connect. This opens Google's consent page, waits for the redirect on a
   `127.0.0.1` port, and stores the refresh token in the OS keychain:

   ```bash
   python -m personalos.cli auth google            # read-only: gmail.readonly, calendar.readonly
   python -m personalos.cli auth google --status   # account, granted scopes, consent expiry
   ```

   The database holds only the reference `cred://google/<you>@gmail.com`,
   never the token.

5. Write access is a separate consent, asked for only once the read-only
   connection exists, and only for the capabilities you name:

   ```bash
   python -m personalos.cli auth google --enable gmail.send     # or gmail.compose, calendar.write
   ```

6. To disconnect, revoke the grant at Google and delete the keychain entry.
   Any later use of the connection fails with a *reconnect required* error
   until you connect again:

   ```bash
   python -m personalos.cli auth google --revoke
   ```

7. Turn on the Gmail and Calendar tools. `MCP_GOOGLE_ENABLED=true` registers
   the Google MCP server in the API and worker processes. The tools act on
   the most recently connected account, or on `GOOGLE_ACCOUNT` if you set it.
   The server opens no port. To run it as a separate process instead, a
   parent spawns `python -m personalos.cli mcp-google` and talks to it over
   stdin/stdout (`bootstrap.connect_google_mcp_stdio`).

8. For interview scheduling on Google Calendar, enable `calendar.write`
   (step 5) and set `CALENDAR_TIME_ZONE` to an IANA zone such as
   `America/New_York`. With both that and `MCP_GOOGLE_ENABLED` set, the
   worker binds `GoogleCalendar` as the calendar reader and the calendar
   executor's client (`bootstrap.build_google_calendar`). Reads and
   free/busy need no approval. Creating, moving or cancelling an event is
   always held for approval, and PersonalOS only changes events it created
   (the ones that carry its `personalos_idempotency_key` stamp).
   `bootstrap.build_calendar_change_scanner` finds interviews that were moved
   in the calendar UI and publishes `calendar.event_changed`, which queues a
   re-plan of their prep blocks.

9. To feed recruiter mail into classification, name the mail PersonalOS may
   look at. This is the whole of its view of your mailbox: a message that
   matches none of the rules is never fetched and nothing of it is stored.

   ```bash
   echo 'GMAIL_RECRUITER_LABELS=["Recruiting"]' >> .env           # Gmail label names
   echo 'GMAIL_RECRUITER_QUERIES=["from:greenhouse.io"]' >> .env  # Gmail search syntax
   python -m personalos.cli inbox status   # the rules, the account, and the cursor
   python -m personalos.cli inbox poll     # one poll, now
   ```

   With `MCP_GOOGLE_ENABLED` and a rule set, the worker polls every
   `GMAIL_POLL_INTERVAL_SECONDS` (`gmail_inbox` sweep). Each poll reads the
   Gmail history since a stored history id (`mailbox_cursors`); a label rule is
   matched from the label ids the history reports, a query rule by a Gmail
   search over new mail. Each in-scope message becomes one `email.received`
   event in the outbox, written in the same transaction as the new cursor, and
   the worker classifies it on a Job Search thread of its own. The first poll,
   or one after the rules change or Gmail's history expires (about a week),
   searches the last `GMAIL_BACKFILL_DAYS` instead. Your own sent mail and
   drafts are never in scope, even on a labelled thread.

The API does the same over HTTP: `POST /v1/integrations/google/connect`
returns an `authorization_url`, Google redirects to
`/v1/integrations/google/callback`, and `GET /v1/integrations` reports the
status. No response contains token material.

> **Unverified app limits.** `gmail.readonly`, `gmail.compose` and
> `gmail.send` are *restricted* scopes. To publish an app that uses them,
> Google requires verification and an annual security assessment. An
> unverified app in **Testing** status avoids that, but Google limits it in
> two ways:
>
> - **Only listed test users can consent** (at most 100). That is enough for
>   local-first, single-user use, but you cannot share the OAuth client with
>   anyone you have not added. Each person should create their own client.
> - **Refresh tokens expire after 7 days.** `auth google --status` shows the
>   expiry. Run `auth google` again before it passes. If you have moved the
>   app to **In production** status, set `GOOGLE_OAUTH_PUBLISHING_STATUS=production`
>   and the connection is no longer recorded as expiring.
>
> Google also shows an "unverified app" warning on the consent page. That is
> expected for your own client.

---

## Security Posture

**Every API route needs the local API token**, except `/health` and Google's
OAuth callback (`apps/api/auth.py`). The token is generated on first run and
kept outside the repository, in a `0600` file
(`~/.config/personalos/api-token`, or `API_TOKEN_FILE`) or, with
`API_TOKEN_STORE=keychain`, in the OS keychain.

- **Send it as a bearer token.** `python -m personalos.cli auth token`
  prints it; `--rotate` replaces it, and a running API stops accepting the
  old one within seconds. A file that other users can read is refused.
- **The actor comes from the token.** Routes record the actor the token
  verified as (`API_ACTOR`, `api` by default). `X-Actor-Id` is ignored, and
  a body that names an identity (`actor_id`, `user_id`, `decided_by`, ...) is
  a 422.
- **Each actor sees only its own.** Workflows, events, approvals,
  applications, artifacts, audit records and credential references are
  scoped by actor in the stores themselves. Another actor's object is a 404,
  exactly like a missing one. See [docs/AUTHORIZATION.md](docs/AUTHORIZATION.md).
- **Tokens carry scopes**: `read`, `write`, `approve` and `admin`
  (`API_TOKEN_SCOPES`, all four by default). Deciding an approval needs
  `approve`. The worker, the scheduler and model-driven nodes are system
  identities that can never hold it, so the agent cannot approve its own
  action. Every denial is audited, and no audit row carries the token.
- **The event stream authenticates too.** A fetch-based SSE client sends the
  header. A browser `EventSource`, which cannot, first calls
  `POST /v1/auth/stream-tickets` with `{"workflow_id": ...}` and opens the
  returned `stream_url`. The ticket names one workflow, expires after
  `API_STREAM_TICKET_SECONDS` (60), and is voided when the token rotates. The
  long-lived token is never accepted in a URL.
- **The token is redacted everywhere.** It is registered with the Phase D
  redactor when loaded, and its `pos_at_` shape (and the tickets' `pos_st_`)
  is matched on its own. Logs, spans and audit payloads never carry either.
- **The API binds to `127.0.0.1` by default.** `API_AUTH_ENABLED=false`
  turns authentication off for local experiments; the API then refuses to
  start on any non-loopback address.
- **Only the configured UI origin may call the API from a browser.**
  `API_CORS_ORIGINS` (a JSON list, default `["http://localhost:5173"]`) is
  the CORS allow-list; a wildcard, a path or a trailing slash is refused at
  startup. Credentials mode is off (the token is a header, not a cookie),
  and a preflight passes only for GET, POST and PUT with `Authorization`,
  `Content-Type` and `Last-Event-ID`. A refused origin gets no
  `Access-Control-*` header at all (`apps/api/security.py`).
- **Every response carries security headers**: a `Content-Security-Policy`
  that loads nothing and forbids framing (`/docs` and `/redoc` get one that
  admits their CDN), `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY` and `Referrer-Policy: no-referrer`.
- **Request bodies are capped** at `API_MAX_BODY_BYTES` (1 MiB). Resume
  ingestion is the one exception, at `API_MAX_RESUME_BODY_BYTES` (10 MiB). A
  bigger body, declared or chunked, is a 413 `payload_too_large` before
  authentication runs. JSON nested deeper than 32 levels, with a string over
  64 KiB, or not valid UTF-8 is a 422 before it is parsed.
  `tests/adversarial/test_http_layer.py` covers these, and malformed JSON.
- **Callers are rate-limited per actor.** Workflow starts, resumes, approval
  decisions, retries and dismissals, and evidence ingestion each have a
  per-minute budget. Each actor may have up to 5 active workflows and 5 open
  event streams. Repeated bad tokens from one address back off exponentially,
  and a single typo doesn't lock the user out. Every refusal is a 429 with
  `Retry-After`, and is counted at `GET /v1/failures/limits`. See
  [docs/LIMITS.md](docs/LIMITS.md).
- **The quick start's PostgreSQL publishes only on `127.0.0.1`.** Its
  password is a placeholder, so keep it off the network.
- **Approvals are the only control on outside effects.** The policy engine
  still denies anything off its allowlist, and every outward-facing action
  still waits for an approval that quotes its `action_hash`. Anyone holding
  a token with `approve` can approve one, so treat it like the user's own password.
- **Secrets stay out of `.env`.** Refresh tokens, API keys and the OAuth
  client secret live in the OS keychain (`SECRET_STORE_SERVICE`). The
  configuration only holds `cred://` references to them. Containers have no
  keychain, so they use files instead (`SECRET_STORE_BACKEND=file`), through the
  same `SecretStore` interface: Docker secrets, plus a `0700` directory on a
  volume. The database still holds only the references. See
  [docs/CONTAINERS.md](docs/CONTAINERS.md#secrets-in-containers).
- **The compose stack publishes only on `127.0.0.1`.** The API and UI are on
  loopback ports. PostgreSQL has no host port and sits on an internal network,
  with a password generated on first run. The API and worker run as non-root
  on read-only root filesystems.

---

## Project Overview

PersonalOS is a local-first agentic assistant for job search. A model only
proposes actions, as typed intents. A deterministic policy engine decides
whether each one may run, and an isolated executor runs the approved ones
through a single tool gateway. State is checkpointed to PostgreSQL, so a run
survives restarts and waits durably for human approval.

### Key Technologies

- **FastAPI** - HTTP API (`apps/api`)
- **LangGraph** - Supervisor and domain subgraphs, checkpointed to PostgreSQL
- **SQLAlchemy 2 + Alembic** - persistence and versioned migrations
- **PostgreSQL + pgvector** - relational state, the outbox, and evidence embeddings
- **Pydantic 2** - typed contracts at every boundary
- **OpenTelemetry** - one trace per workflow, off by default
- **MCP** - the files tool server, behind the tool gateway

### Core Principles

- **The model emits intent only.** It never calls a tool or touches a credential.
- **Default deny.** No code path reaches a tool without a recorded policy decision.
- **Approval before outside effects.** Anything that writes outside the system
  pauses for a human, and the approval is bound to a hash of the exact action.
- **Exactly once.** Every mutation carries an idempotency key and is journaled
  with the decision that authorized it.
- **Durable by default.** Requests return immediately. Work runs in the worker,
  and everything it needs to resume after a restart is in the database.
- **Enforced boundaries.** Layer imports are checked in CI
  (`scripts/check_boundaries.py`, `docs/ARCHITECTURE_BOUNDARIES.md`).

---

## Project Structure

Every directory down to two levels is listed. Entries marked `(planned …)`
exist only as placeholders. `scripts/check_guide_tree.py` runs in CI and fails
when this tree and the repository drift apart.

<!-- tree:begin -->
```
PersonalOS/
├── pyproject.toml                 # dependencies and tool config
├── alembic.ini                    # migrations read DATABASE_URL via migrations/env.py
├── .env.example                   # every setting, with defaults
├── Dockerfile                     # api, worker and worker-quickstart images
├── docker-compose.yml             # the stack: docker compose up (docs/CONTAINERS.md)
├── compose.quickstart.yml         # override: the worker on the fake job board
├── README.md
├── IMPLEMENTATION_GUIDE.md        # this file
│
├── personalos/                    # the application package
│   ├── config.py                  # settings from the environment and .env
│   ├── bootstrap.py               # composition root: wires every layer
│   ├── cli.py                     # api, worker, db-init, ingest-resume, profile, approvals, ...
│   ├── domain/                    # entities, typed intents, errors, workflow ids; imports nothing internal
│   ├── policy/                    # default-deny engine: allow / require_approval / deny
│   ├── tools/                     # the tool gateway (the only way to a tool) and registry
│   ├── executor/                  # runs approved intents: idempotency, retries, credential broker
│   ├── graphs/                    # LangGraph Supervisor, Job Search and File subgraphs, node contracts
│   ├── persistence/               # ORM, repositories, checkpointer, outbox, audit, leases
│   ├── events/                    # event bus and the in-memory event transport
│   ├── mcp/                       # MCP server base, manager, cache, gateway adapter
│   ├── providers/                 # job board adapters: Greenhouse, Lever, Ashby, Workday (opt-in), and a fake
│   ├── retrieval/                 # resume ingestion, embedder, evidence selection, job matching
│   ├── models/                    # model gateway: rules by default; Anthropic, Ollama, OpenAI-compatible
│   ├── secrets/                   # keychain and container file stores, short-lived token exchange
│   └── observability/             # structured logging, tracing, redaction
│
├── apps/                          # process entry points (composition layer)
│   ├── api/                       # FastAPI app
│   │   └── routes/                # chat/workflows, approvals, applications, audit, failures,
│   │                              # events (SSE), profile, evidence
│   └── worker/                    # workflow commands, outbox dispatch, scheduled sweeps
│       ├── graphs.py              # default graph factory (WORKER_GRAPHS)
│       ├── fake_providers.py      # quick-start factory: fake board, extractive drafts
│       ├── health_monitor.py      # scheduled integration health checks, circuit breakers
│       ├── heartbeat.py           # liveness file for the container health check
│       └── unconfigured.py        # stand-ins for ports with no adapter yet
│
├── ui/                            # operator UI: Vite + React + TypeScript (npm run dev)
│   ├── Dockerfile                 # the UI built and served by nginx (compose `ui` profile)
│   ├── docker/                    # the nginx config, with the same headers as `npm run preview`
│   ├── openapi.json               # the API schema, written by scripts/export_openapi.py
│   ├── src/                       # approvals, board, workflow progress, failures, audit; src/api/schema.d.ts is generated
│   └── tests/                     # Vitest + Testing Library component tests
│
├── mcp_servers/
│   ├── files/                     # sandboxed read_file / write_file
│   └── google/                    # Gmail and Calendar tools, in-process or over stdio
│
├── migrations/                    # Alembic environment
│   └── versions/                  # one file per schema change
│
├── evals/                         # evaluation harness: python -m evals.harness
│   ├── golden/                    # golden dataset: per-domain routing, mail, postings, scenarios
│   └── model_benchmarks/          # model configs, baselines, review ratings
│
├── deploy/                        # compose stack configuration
│   ├── compose.defaults.env       # container settings; override in .env.compose
│   └── otel/                      # the OpenTelemetry collector's config
│
├── docs/
│   ├── ARCHITECTURE_BOUNDARIES.md # layer ownership and enforced import rules
│   ├── CONTAINERS.md              # images, compose, secrets in containers, volumes, reset
│   ├── AUTHORIZATION.md           # scopes, system identities, actor-scoped stores
│   ├── LIMITS.md                  # rate limits, concurrency caps, request limits
│   ├── EVENT_TRANSPORT.md         # the outbox transport and how to swap it
│   ├── ADDING_A_DOMAIN.md         # classifier labels, golden cases, policy rows, subgraph wiring
│   └── quickstart/                # sample resume and profile for the quick start
│
├── scripts/
│   ├── check_boundaries.py        # layer import check (CI)
│   ├── check_guide_tree.py        # this tree against the repository (CI)
│   ├── check_image.py             # run inside a built image: non-root, no fakes or dev deps (CI)
│   ├── smoke_compose.py           # the containerized quick start, end to end (CI)
│   ├── export_openapi.py          # ui/openapi.json from the FastAPI app (--check in CI)
│   └── smoke_quickstart.py        # the quick start, end to end (CI)
│
└── tests/
    ├── architecture/              # layer boundaries and the intent-only boundary
    ├── contracts/                 # node contract coverage, event transport, offline guards
    ├── unit/                      # pure logic, routes, stores, worker wiring
    ├── graph_scenarios/           # whole-graph runs: approvals, resume, recovery, tracing
    ├── adversarial/               # prompt injection, permission escalation, hostile payloads
    ├── migrations/                # alembic upgrade/downgrade against PostgreSQL
    └── fixtures/                  # fakes, recorded model outputs, sample resumes
```
<!-- tree:end -->

---

## Architecture

Layer ownership and the enforced import rules live in
[docs/ARCHITECTURE_BOUNDARIES.md](docs/ARCHITECTURE_BOUNDARIES.md). The short
version: orchestration proposes intents, policy decides, executors run only
approved intents, adapters do the I/O.

### Data Flow

```
 client ──HTTP──▶ apps/api ──writes a command row, returns 202──▶ PostgreSQL
                     ▲                                               │
                     │ status, approvals, audit, SSE progress        │ outbox / command queue
                     │                                               ▼
                     └──────────── reads ◀────────────────────  apps/worker
                                                                     │ claims the command, takes the lease
                                                                     ▼
                                                   Supervisor graph ─▶ Job Search subgraph
                                                                     │ every step checkpointed
                                                                     ▼
                                                    ToolIntent / ActionIntent (proposals only)
                                                                     │
                                                                     ▼
                                                   Policy engine: allow │ require_approval │ deny
                                                                     │        │
                                                                     │        ▼
                                                                     │   approval interrupt ◀── POST /v1/approvals/{id}/decision
                                                                     ▼        │
                                                   Executor (idempotency ledger, retries, credential broker)
                                                                     │
                                                                     ▼
                                                   Tool gateway ─▶ MCP servers │ job providers │ (planned adapters)
                                                                     │
                                                                     ▼
                                                   receipts, audit events, outbox events ─▶ PostgreSQL
```

### Key Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Policy Gate** | `personalos/policy/engine.py` | Default-deny authorization of every intent |
| **Ports & Adapters** | `personalos/tools/gateway.py`, `personalos/graphs/job_search.py` | Graphs and executors depend on ports, not adapters |
| **Composition Root** | `personalos/bootstrap.py`, `apps/worker/graphs.py` | The only places that wire concrete adapters |
| **Durable Checkpoints** | `personalos/persistence/checkpointer.py` | Resume any workflow by id after a crash |
| **Transactional Outbox** | `personalos/persistence/event_outbox.py` | Events and commands committed with the state that caused them |
| **Idempotency Ledger** | `personalos/persistence/execution_ledger.py` | Each side effect runs once per key |
| **Repository** | `personalos/persistence/repositories.py` | Data access behind typed methods |
| **Unconfigured Stand-ins** | `apps/worker/unconfigured.py` | A missing adapter fails loudly and retryably, never silently |

---

## Built Components

### 1. Configuration (`personalos/config.py`)

One `Settings` object read from the environment and `.env`. `.env.example`
lists every setting with its default; [Configuration](#configuration) covers
the ones you are likely to change.

### 2. Domain (`personalos/domain/`)

Typed values every other layer shares, with no internal imports: intents and
action kinds, the error taxonomy (`errors.py`, which distinguishes retryable
from terminal failures), workflow and thread ids, job search values
(`SearchProfile`, `NormalizedPosting`, `ShortlistEntry`), artifacts and
evidence, recruiter events, and the application lifecycle state machine.

### 3. Policy (`personalos/policy/`)

`PolicyEngine` decides `allow`, `require_approval` or `deny` for every intent
from the tool allowlist, the tool's permission class (`read_local` through
`destructive`), and the intent's origin. Every decision is recorded in
`policy_decisions` before anything runs.

Google tools are also allowlisted per node. `NodeToolAllowlistRule` checks the
intent's `requested_by` (for example `graph:job_search#plan_interview_schedule`)
against `GOOGLE_NODE_TOOL_ALLOWLISTS`. A node that is not listed for a tool is
denied, even when the intent carries a matching human approval. By default only
`handle_recruiter_response` may send mail, and only `plan_interview_schedule`
and the Calendar subgraph's `plan_calendar_change` may write to (or delete
from) the calendar -- every such write still needs its approval, and the
calendar executor refuses any event without the PersonalOS stamp.

### 4. Tools and Executor (`personalos/tools/`, `personalos/executor/`)

`ToolGateway` is the only way to reach a tool, and it consults policy first.
`ToolExecutor` runs approved actions under an idempotency key, journals each
attempt, retries transient failures with backoff, and brokers short-lived
credentials so long-lived secrets never leave `personalos/secrets`.

### 5. Persistence (`personalos/persistence/`, `migrations/`)

PostgreSQL through SQLAlchemy 2, with the schema in Alembic migrations. It
holds workflows, checkpoints and leases; postings, profiles, applications and
their lifecycle; policy decisions, tool executions and audit events; the
outbox with retries and dead letters; and pgvector tables for evidence,
posting and message embeddings.

### 6. Retrieval and Models (`personalos/retrieval/`, `personalos/models/`)

Resume ingestion (`ingest-resume`, `POST /v1/evidence/documents`) chunks
PDF, DOCX, Markdown or text resumes into evidence. `HybridJobMatcher` scores
postings and grounds each claim in that evidence. The model gateway
(`LLM_GATEWAY_CONFIG`) assigns a provider to each role. With no config,
routing and recruiter-event extraction run on deterministic rules, semantic
assessment declines, and the draft writer is unbound.

The gateway also selects the embedder (`personalos/models/embedding.py`). The
default is a local Ollama model (`nomic-embed-text`), and `hashing` is the
offline fallback. A hosted embedder (`openai`) is refused unless
`CLOUD_INFERENCE_ALLOWED=true`. Every vector row records its embedding model
and version, and retrieval only compares vectors from one model. So after
changing the embedder, run `python -m personalos.cli reembed`. It re-embeds
evidence, posting and message vectors in batches. Evidence is overwritten in
place, so cited evidence ids do not change. Rows already on the new model are
skipped, so an interrupted run resumes when you run it again.

### 7. API (`apps/api/`)

Every route answers in milliseconds. Long-running work is queued for the
worker.

```
POST   /v1/chat                                   Start a workflow (202, returns workflow_id)
GET    /v1/workflows/{id}                         Status, plan, current step, pending approval
POST   /v1/workflows/{id}/resume                  Answer an interrupt directly
GET    /v1/workflows/{id}/events                  Progress as server-sent events
GET    /v1/workflows/{id}/failures                Failed steps, with retry/dismiss actions
POST   /v1/workflows/{id}/retry | /dismiss        Act on a failed step
GET    /v1/workflows/{id}/audit, /v1/audit        The decision and execution trail
GET    /v1/approvals                              Everything waiting for a human
POST   /v1/approvals/{id}/decision                Approve or reject, quoting action_hash
GET    /v1/applications[/{id}]                    The application board
POST   /v1/applications/{id}/transition           Request a lifecycle transition
POST   /v1/profile, GET/PUT /v1/profile           The search profile (versioned)
GET    /v1/profile/versions[/{n}]
POST   /v1/evidence/documents                     Ingest a resume
GET    /v1/integrations                           Connected accounts, scopes, consent expiry
POST   /v1/integrations/google/connect            Begin Google consent (read-only unless `enable`)
GET    /v1/integrations/google/callback           Google's redirect; stores the refresh token
POST   /v1/integrations/google/revoke             Revoke at Google, delete the keychain entry
GET    /v1/failures/events, POST .../{id}/retry   Parked outbox events
GET    /v1/auth/whoami                            The actor the token authenticates as
POST   /v1/auth/stream-tickets                    A short-lived ticket for one event stream
GET    /health                                    Liveness; the only route without a token
```

Every route needs `Authorization: Bearer <token>` except `/health` and
Google's OAuth callback, whose one-time `state` is its credential. The actor a
route records is the one the token verified as. See
[Security Posture](#security-posture).

### 8. Worker (`apps/worker/`)

One process runs three loops against the database: it claims queued workflow
commands and runs them under a lease, dispatches outbox events to their
consumers, and runs scheduled sweeps (due follow-ups, stalled applications,
expired approvals). It keeps nothing in memory that a restart would lose.
`WORKER_GRAPHS` names the graph factory: `apps.worker.graphs:build_worker_graphs`
by default, or `apps.worker.fake_providers:build_worker_graphs` for the quick
start. Ports without an adapter are bound to `apps/worker/unconfigured.py`
stand-ins, which the worker lists at startup.

### 9. Events and Observability (`personalos/events/`, `personalos/observability/`)

Producers and consumers depend on the transport interfaces in
`personalos/domain/event_transport.py`, so the Postgres outbox can be swapped
out (`docs/EVENT_TRANSPORT.md`). Logs are structured and carry
`workflow_id`/`correlation_id`. Each workflow is one OpenTelemetry trace, and
credentials are redacted from logs, traces, checkpoints and audit payloads.

### 10. CLI (`personalos/cli.py`)

```
python -m personalos.cli api [--host 127.0.0.1] [--port 8000] [--reload]
python -m personalos.cli worker [--graphs module:callable]
python -m personalos.cli db-init                  # create tables without Alembic (tests, SQLite)
python -m personalos.cli ingest-resume PATH [--actor api]
python -m personalos.cli profile create|update|show|versions [--actor api]
python -m personalos.cli approvals [--json]
python -m personalos.cli workflow-status WORKFLOW_ID [--json]
python -m personalos.cli stall-monitor [--once]
python -m personalos.cli auth google [--enable CAPABILITY] [--status] [--revoke] [--json]
python -m personalos.cli auth google-client [--from-json PATH]
python -m personalos.cli mcp-google               # serve the Google MCP server on stdio
```

### 11. Orchestration Graphs (`personalos/graphs/`)

**`supervisor.py` — the top-level graph.** Loads the actor's long-term facts,
classifies the request into a typed `RouteDecision`, and either asks for
clarification or plans a bounded `TaskDAG` and hands it to a domain subgraph. It
holds no tool and no repository: `IntentClassifier`, `JobSubgraphRunner`,
`FileSubgraphRunner`, `CommunicationsSubgraphRunner` and
`CalendarSubgraphRunner` are injected ports. All four domains (`JOB`, `FILE`,
`COMMUNICATIONS`, `CALENDAR`) are routable; a Supervisor built without one of
the optional subgraphs asks that domain back.

**Cross-domain plans** (K4; `domain/task_plan.py`, `models/task_planning.py`,
`graphs/plan_steps.py`). A request that spans domains, such as "reply to the
recruiter and hold Thursday at 2pm", is planned as a `TaskPlan`: a DAG of
steps, each with a domain, an instruction, inputs, outputs and dependencies.
Each step runs through that domain's existing subgraph. A request that is one
domain's work takes the routing path above, unchanged.

```
load_context -> plan_task -> classify_intent -> ...            (one domain)
                plan_task -> clarify                           (no valid plan after repairs)
                plan_task -> publish_plan -> run_plan_step     (one step per superstep)
                             run_plan_step -> await_plan_steps (interrupts)
                             -> collect_plan_steps -> ...
                             -> settle_plan -> run_plan_step   (a proposed release)
                             -> settle_plan -> END
```

| Property | How it holds |
| --- | --- |
| Validated before it runs | `validate_task_plan` checks the planner's draft against `PLAN_DRAFT_JSON_SCHEMA`, then checks the step cap (`SUPERVISOR_MAX_PLAN_STEPS`), cycles, unknown or unwired domains, unknown dependencies, and that each input is a declared output of a dependency, of a kind the consuming domain accepts. It reports every issue at once. `ResilientTaskPlanner` hands an invalid draft back with those issues up to `SUPERVISOR_PLAN_REPAIR_ATTEMPTS` times. If the draft is still invalid, the request is asked back and the refusal is audited (`plan.rejected`). A plan is never trimmed or guessed into shape. A planner that raises falls back to `RuleBasedTaskPlanner`. |
| One workflow, a thread per step | Each step runs on its own thread in the domain's namespace, keyed by the workflow, the plan and the step (`step_task_key`). The worker registers it under the Supervisor run's workflow, owned by the conversation's candidate (`apps/worker/plan_steps.py:WorkflowScopedStepThreads`). Actor scoping applies to every step, and the active-workflow cap counts the plan once. |
| Durable, step by step | Each step is its own superstep, so its `StepRecord` is checkpointed before the next step starts. After a crash, the plan resumes at the step that was running. A re-run step reads its thread rather than starting the subgraph again, so a finished step's side effect is never repeated. |
| One approval per effect | A plan has no approval of its own. Each step's subgraph raises its own `ApprovalRequest`, bound to its own `action_hash`, on the step's thread, and `GET /v1/approvals` lists them separately. The Supervisor parks on an interrupt with stage `plan_steps`, which carries no requests. After the worker runs a command that moves a step thread, `PlanStepResumer` queues one resume of the plan, deduplicated on the plan's checkpoint. |
| Refusals stop dependents only | A rejected, expired or failed step skips its dependents, or pauses them when the step asks for `on_upstream_failure: pause`. Independent steps keep running. The plan view (`result.plan`) lists the steps that ran, are awaiting approval, were rejected, failed, were skipped, paused or are still pending. |
| Partial failure | When a plan ends incomplete, `plan_compensations` lists every effect that did happen. A calendar step marked `propose_release` gets a release step, which proposes cancelling the event it made and waits for its own approval. Every other effect, such as a sent email, is left as is and listed. Nothing is removed without approval. |
| By reference | Steps pass `StepRef`s (a kind and a bare id, such as a thread id or an event id), never content. The ids are mapped onto the subgraph's structured fields, and each step's instruction is the person's own words for that step. A ref read from outside, such as a mail thread, is flagged `untrusted`. The planner sees only the request. |
| Shown and audited first | `publish_plan` records the whole plan (`plan.proposed`, `plan.step.planned`) and returns it as `result.plan` and `task_dag` before any step runs. Each step then records `plan.step.intent`, `plan.step.policy`, `plan.step.approval.<state>` and `plan.step.receipt`, in order, followed by `plan.effect.dangling` and `plan.finished` (`persistence/plan_audit.py`, with strictly increasing timestamps, recorded once per key). |

Tests: `tests/adversarial/test_cross_domain_plans.py` (two-domain DAG with
separate approvals, cyclic and over-length plans refused, a crash after step 1
resumes at step 2, a rejected reply leaves the hold dangling with a proposed
release, the audit order), `tests/graph_scenarios/test_cross_domain_plan_worker.py`
(the same through the approval queue and the worker) and
`tests/unit/test_task_plan.py`.

**`file_task.py` — the File subgraph** (K1). Finds, reads, summarizes and
organizes job-search documents (resumes, cover letters, application
documents):

```
discover_files -> plan_file_task -> read_documents
-> [needs approval] request_file_approval -> file_approval_checkpoint (interrupts)
-> apply_file_changes -> verify_file_changes -> END
```

| Property | How it holds |
| --- | --- |
| No raw filesystem access | The graph's only way out is the `FileOperations` port, bound to `executor/file_operations.py:GatewayFileOperations`. Every call is a `files.*` intent through the policy gateway to the sandboxed files MCP server, which resolves paths canonically (after `..` and symlinks) and refuses, and audits (`files.path_denied` in `audit_events`), anything outside `FILES_ALLOWED_ROOTS`. |
| Permission classes | `list_files`, `stat_file`, `read_file` are `READ_LOCAL`; `write_file` and `move_file` are `WRITE_REVERSIBLE` (a move returns its undo record); `delete_file` is `DESTRUCTIVE` and always needs approval. |
| Typed, validated plan | `plan_file_task` asks the `ResilientFilePlanner` (a model constrained to `FilePlan`, re-asked with the error, then the deterministic `RuleBasedFilePlanner`) and re-validates the result against `FILE_PLAN_JSON_SCHEMA` and the inventory. The planner sees metadata only, never content. The fallback never plans a delete. |
| Approval shows exact paths | A plan with a delete, an overwrite, or more changed files than `FILES_MAX_UNATTENDED` is proposed as one `APPLY_FILE_PLAN` action listing every step and path and pinning the plan's hash. It uses the same approval triple and inbox as job search; at redemption both the action hash and the plan hash are recomputed. Policy holds moves from larger plans (`FileBulkMutationRule`) and every delete (`DestructiveFileRule`) on its own. |
| Idempotent mutations | Each mutating step carries `step_idempotency_key(plan, step)` and the hash discovery saw. Before each write, move or delete the apply node checks the disk: a file already in its goal state (a crash after the write, before the checkpoint) is `RECONCILED`, not written again. |
| Documents are data | Content reaches only `read_documents`, whose `requested_by` may call `read_file`/`stat_file` and nothing else (`FILE_TASK_NODE_TOOL_ALLOWLISTS`); it goes to the summarizer as quoted data and comes back as a `DocumentSummary`, which has no field for steps or tool calls. Moves and deletes are reserved for File subgraph nodes. |
| Redaction and scoping | Read results are redacted by the gateway and again before they become state; only hashes, summaries and short excerpts are kept, and the durable checkpointer redacts every checkpoint. A Supervisor run's file task runs on a `file_task` thread of the same workflow, owned by the conversation's candidate. |

Tests: `tests/adversarial/test_file_task_boundaries.py` (traversal and
symlinks denied and audited, deletes and bulk moves refused without approval,
crash-after-write replay, prompt injection in a resume, Supervisor routing),
`tests/unit/test_file_task_nodes.py`, `tests/unit/test_file_planning.py`,
`tests/unit/test_files_server_tools.py`.

**`communications.py` — the Communications subgraph** (K2). Triages the
in-scope inbox, summarizes a thread, extracts its action items and drafts a
reply, over the J2 Google tools, the J3 mailbox scope and the J5 drafting and
send path. It adds no provider code.

```
parse_request -> triage_inbox -> END
parse_request -> load_thread -> link_application -> END             (summary, action items)
              -> load_thread -> link_application -> draft_reply
              -> request_send_approval -> send_approval_checkpoint (interrupts)
              -> execute_send -> END                                (draft a reply)
```

| Property | How it holds |
| --- | --- |
| Scoped reads | `executor/gmail_threads.py:GmailThreadReader` searches only the `MailboxScope` rules the inbox poller uses (`GMAIL_RECRUITER_LABELS`, `GMAIL_RECRUITER_QUERIES`). A thread with no message matching a rule is not read (`ThreadNotInScope`), and an empty scope reads nothing. Spam, trash and drafts are never read; the candidate's sent mail is read inside a thread but never triaged. |
| Permission classes | Listings and reads are `READ_EXTERNAL`. The Gmail draft and the `artifact_versions` row are `WRITE_REVERSIBLE`. `gmail_send_message` stays `WRITE_EXTERNAL` with `require_approval`, and no node of this subgraph may call it (`COMMUNICATIONS_NODE_TOOL_ALLOWLISTS`): an approved reply is sent by `GmailSendExecutor` as `GMAIL_SENDER`, journaled and reconciled against the Sent folder. A second `RequesterToolScopeRule` keeps these nodes off every other server. |
| Typed triage | `TriageResult`: category, urgency and suggested action from closed enums. A model answer that will not validate as `TriageProposal`, after the repair attempt, becomes `needs_review` (`ResilientTriager`); a provider error does too. Under `rules`, `RuleBasedTriager` answers. |
| Summaries cite messages | `ThreadSummary.message_ids` lists every message covered, and each key point and action item names the message ids it came from. `digest_from_proposal` refuses an id that is not in the thread, and the analyzer then falls back to the extractive digest. |
| No invented claims | `retrieval/thread_reply_drafting.py:ThreadReplyDrafter` writes a `ReplyProposal` from the candidate's evidence and checks it with `finalize_reply` (the Phase E validator). A rejected draft is recorded as a `DraftRejection` and none of it is kept. The neutral template reply is drafted instead. A thread tied to an application is grounded in that application's company, role and skills, and an application belonging to another user is refused. |
| The approval shows the message and is bound to it | One `SEND_EMAIL` intent. Its `ApprovalRequest.preview` has the recipient, subject, body diff, attachments and `in_reply_to` (a reply or a new thread). Recipient and subject are set by code from the message being answered. `execute_send` recomputes the hash, so an edited body or recipient is refused with `action_hash_mismatch`. Nothing is sent without a person's answer. |
| Mail is data | Bodies reach the triager, analyzer, extractor and writer as quoted JSON, and each one returns a closed schema. Routing reads only the parsed request and the nodes' conclusions. Bodies are not checkpointed: `load_thread` keeps the thread without them and `draft_reply` reads it again. Links are stored as inert strings (`EmailMessage.links`). No reachable tool fetches a URL or downloads an attachment. |
| Applications | Inbound messages are classified (`RecruiterEventExtractor`), correlated (`correlate_application`, deterministic) and triaged exactly as J3 mail is. `link_application` records matched mail through `SqlRecruiterEventStore`, writing one `communication_events` row and at most one lifecycle transition per message, through the lifecycle's rules. A refused transition or a weak match becomes a review. Reprocessing a thread records nothing new. |
| Actors and limits | A Supervisor run's task runs on a `communications` thread of the same workflow, owned by the conversation's candidate, so the send approval is that actor's. It is reached through `POST /v1/chat`, so the Phase L per-actor rate and active-workflow limits apply. Each run reads at most `MAX_TRIAGE_ITEMS` listings and `MAX_THREAD_MESSAGES` messages. |

Models: `message_triager`, `thread_analyzer` and `thread_reply_drafter`
roles in the model configuration. All three fall back to the rules (or the
template reply) under `rules`.

Tests: `tests/adversarial/test_communications_boundaries.py` (routing and a
typed summary, no send without approval, a hash mismatch after an edit, the
claim validator, prompt injection, recruiter events exactly once, scope and
actors) and `tests/unit/test_communications_nodes.py`.

**`calendar.py` — the Calendar subgraph** (K3). Answers availability
questions, proposes interview times from free/busy, and creates, moves or
cancels events, over the J4 `GoogleCalendar` reader and the J4 calendar
executor. It adds no provider code.

```
parse_request -> check_availability -> END                (when am I free, propose times)
parse_request -> plan_calendar_change -> END              (conflict, not found, not ours, ...)
              -> plan_calendar_change -> request_calendar_approval
              -> calendar_approval_checkpoint (interrupts)
              -> execute_calendar_changes -> END          (create, move, cancel)
```

| Property | How it holds |
| --- | --- |
| Permission classes | Free/busy, listings and single-event reads are `READ_EXTERNAL` and need no approval; they are made as `graph:calendar#read_calendar`, whose allowlist cannot write. Creates and moves are `WRITE_EXTERNAL` and cancels `DESTRUCTIVE`, all `require_approval`, made by the calendar executor as `graph:calendar#plan_calendar_change` after `execute_calendar_changes` has checked the approval (`CALENDAR_NODE_TOOL_ALLOWLISTS`; a `RequesterToolScopeRule` keeps these identities off every other server). |
| Explicit time zones | `parse_request` reads relative phrases ("next Tuesday afternoon", "Thursday at 2pm") against `CALENDAR_TIME_ZONE` and a pinned clock (`domain/calendar.py:resolve_when`). A DST gap moves the time forward, a DST overlap takes the first occurrence, and each reading -- which day "next Tuesday" was, which offset applied -- is recorded as a note. Availability windows are drawn per local date, so the working day stays 09:00-17:00 local across a DST change. The server's zone is never read. |
| The approval shows the event | Each write's payload, hashed into its `ApprovalRequest`, has the title, attendees, start and end in UTC and local time with the IANA zone, the phrase the time was read from and the resolution notes; a move adds the previous times and a `changes` list. `ActionPreview` shows the attendees and a body diff of old and new times. An edited payload fails the hash at `execute_calendar_changes`. |
| Exactly once | Each create carries a stable idempotency key (`calendar_create_key`), stamped by the executor in `extendedProperties.private`. Under `ToolExecutor`, an attempt with no recorded outcome is reconciled against the calendar by that stamp before any retry (`CalendarReconciler`), so a lost response becomes the found event, not a second insert. |
| Never someone else's event | A move or cancel of an event without the PersonalOS stamp ends as `CalendarOutcome.NOT_MANAGED` with no write proposed, and `CalendarActionExecutor` refuses it again if one were ever approved. An unstamped event is planned around, never moved. |
| Typed conflicts | A slot taken by an event (or by free/busy alone, e.g. another calendar) ends as `CalendarOutcome.CONFLICT` with a `CalendarConflict`: the requested slot, what is in the way, and up to three free alternatives of the same length. No intent is proposed and the run does not interrupt. Every outcome is a `CalendarOutcome`, never free text. |
| Applications and prep blocks | A create or move of an interview linked to one of the candidate's applications (by id, or by company name) runs through the Phase E planner (`plan_interview_schedule`), so the interview and its prep blocks are proposed in one approval and a changed interview time moves its prep blocks with it. Cancelling an interview cancels its upcoming prep blocks. An application that is not the candidate's (`ApplicationDirectory`) is refused. |
| Actors and limits | A Supervisor run's task runs on a `calendar` thread of the same workflow, owned by the conversation's candidate, so its approvals are that actor's. It is reached through `POST /v1/chat`, so the Phase L per-actor rate and active-workflow limits apply; approvals go through the rate-limited decision route. One run proposes at most `MAX_WRITES_PER_RUN` writes and `MAX_PROPOSALS` slots. |

Wiring: `bootstrap.build_calendar_reader` and `build_calendar_graph`. The worker
wires the subgraph only when `CALENDAR_TIME_ZONE` is set and the Google server
is enabled; until then a calendar request is asked back.

Tests: `tests/adversarial/test_calendar_boundaries.py` (availability from fake
free/busy in the configured zone, no create without approval, exactly one event
after a lost response, double-booking as a typed conflict, outside events never
changed, DST boundaries, prep-block reconciliation, scoping, routing),
`tests/unit/test_calendar_nodes.py` and `tests/unit/test_calendar_time.py`.

**`job_search.py` — the Job Search subgraph**:

```
load_search_profile -> search_providers -> normalize_jobs -> deduplicate
-> hard_filter -> score_candidates -> evidence_check -> rank -> shortlist
-> [optional] prepare_application_packet
-> request_approval_for_external_write
-> approval_checkpoint_for_external_submission     (interrupts here)
-> execute_approved_actions
-> persist_application -> emit application.created -> END
```

with a recruiter-response branch off the end (`handle_recruiter_response` ->
`create_follow_up_checkpoint`), which loops an owed reply back through the same
approval sequence. Recruiter handling and follow-up checkpoints live here
rather than in a Communications or Calendar subgraph: for a job-search-only
build they are steps in the application's own lifecycle.

There is a second way in. A run invoked with `fired_checkpoints` in its input
is a durable wait coming due, and enters at `draft_follow_up_for_triggered_checkpoint`
instead of at discovery — see [Pending checkpoints](#14-pending-checkpoints-durable-conditional-waits).

And a third: a run invoked with `inbound_messages` is recruiter mail not yet
tied to an application, and enters at `classify_recruiter_events` — see
[Recruiter events](#15-recruiter-events-classification-commitments-and-correlation).

And a fourth: a run invoked with `calendar_changes` is an interview event that
moved on the calendar, and enters at `plan_interview_schedule` -- see
[Interview scheduling](#16-conditional-follow-ups-and-interview-scheduling).

**Approval interrupts before external writes.** Every point at which the graph
is about to do something the outside world can see goes through those three
nodes, and only through them:

- `request_approval` raises an `ApprovalRequest` per pending action — the
  action's hash, its target, a human-readable summary, a risk level, the scopes
  it consumes and an expiry — and returns, so all of it is checkpointed.
- `approval_checkpoint` calls LangGraph's `interrupt()`. The run stops with its
  whole state durable; the answer may arrive hours or days later, from a process
  that did not exist when the request was raised. Resume it with
  `Command(resume=[ApprovalDecision(...)])`.
- `execute_approved_actions` runs a super-step later, re-reads the pending
  action from the checkpoint, **recomputes its hash**, and executes only if it
  still matches the hash the approval was granted against. Anything else is
  recorded as an `ApprovalRefusal` and not called.

That last check is what stops an approval of "apply to the backend role at
Acme" being spent on whatever the pending action says by the time it is
redeemed. Risk levels, requested scopes and per-kind approval TTLs are a table
in `personalos/domain/job_search.py` (`ACTION_RISK_PROFILES`), so a new kind of
side effect cannot reach a reviewer without someone deciding what reviewing it
costs.

Three properties hold across every node:

| Property | How |
| --- | --- |
| A node is a transformer, not an agent | Reads a narrow slice of `JobSearchState`, calls injected ports only, returns a typed partial update. The `graphs` layer cannot import `tools`, `mcp` or `persistence` at all. |
| Nothing executes a side effect without a human answer | A node returns an `ActionIntent` and has no executor and no edge to one. Only `execute_approved_actions` holds an `ActionExecutor`, and it acts only when `authorize_execution` clears the action against its checkpointed `ApprovalRequest` — right verdict, right request, unexpired, and a recomputed hash that still matches. |
| State is JSON, not objects | Every `JobSearchState` value round-trips through a `BaseCheckpointSaver`, which is what lets the run park at the interrupt and a human answer it later, from another process. |

The typed values passed between nodes are in
[`personalos/domain/job_search.py`](personalos/domain/job_search.py) — all
`frozen=True` and `extra="forbid"`, so a node cannot mutate a posting in place
and a provider cannot smuggle an unexpected field through normalization.

**Tests:** a full happy path against fake providers in
`tests/graph_scenarios/test_job_search_graph.py`, per-node input/output
contracts in `tests/unit/test_job_search_nodes.py` (including a guard that
fails if a node is added without a contract test), and one fake per port in
`tests/fixtures/job_search_fakes.py`. The approval interrupt has its own two
files: `tests/unit/test_approval_requests.py` for the request and the guard as
pure values, and `tests/graph_scenarios/test_approval_interrupts.py` for the
end-to-end cases — approving and resuming, mutating the pending action between
the approval and the resume (refused), and parking a run in one process,
approving it in a second and resuming it in a third.

**Not yet wired:** `personalos/bootstrap.py` does not build a
`JobSearchGraph` — the concrete adapters behind its ports (a repository-backed
`ApplicationStore`, a gateway-backed `ActionExecutor`, real job board
providers) are a follow-up. The graph is complete and exercised against fakes.

### 12. Durable Checkpointing and Resume

Both graphs default to `InMemorySaver`, which is a test fixture: a restart loses
every thread it held. The production checkpointer is
[`persistence/checkpointer.py`](personalos/persistence/checkpointer.py), backed
by the `checkpoints` and `checkpoint_writes` tables, and the composition root
passes it in — the graphs themselves are unchanged, because they already compile
against LangGraph's `BaseCheckpointSaver` port.

**Two identifiers, deliberately distinct**
([`domain/workflow.py`](personalos/domain/workflow.py)):

| | Names | Notes |
| --- | --- | --- |
| `thread_id` | one conversational/orchestration thread | *Derived*, via `derive_thread_id`, so a restarted process recomputes it instead of having to have kept it. A thread id minted per invocation checkpoints diligently and can never resume. |
| `workflow_id` | one long-running business process | May span several threads (a Supervisor conversation and the Job Search subgraph it delegated to). What an operator resumes, and what a lease is taken against. |

A thread is bound to its workflow by `WorkflowThreadRegistry.register`
(get-or-create on `workflows`/`workflow_runs`, so re-registering a derived id
rejoins its run). The saver refuses to checkpoint a thread that is not bound —
the alternative is a checkpoint under an invented `workflow_id`, which is a
checkpoint nobody resuming that workflow will find.

**Nothing is written twice.** Three layers, coarse to fine:

| Layer | Guarantees |
| --- | --- |
| Checkpoint per super-step | LangGraph writes the step's checkpoint *before* running its tasks, so a worker killed inside a node resumes at that node with everything before it intact. |
| `checkpoint_writes` | Tasks that finished inside the in-flight step are recorded, so a resume replays only what never landed. |
| [`action_journal.py`](personalos/persistence/action_journal.py) | Commits a claim on the action's `idempotency_key` immediately before the external call and the receipt immediately after. A resume that finds the receipt replays it; one that finds a claim with no receipt returns a not-ok receipt and does **not** call out again. |

A deployment binds [`ToolExecutor`](personalos/executor/tool_executor.py)
(`bootstrap.build_tool_executor`), which applies that rule and additionally
records each action against the `policy_decisions` row that authorized it, in
both `tool_executions` and `audit_events`, and can reconcile an unrecorded
outcome against the provider before deciding whether to execute again.

That last rule is the one with a cost: an action whose outcome is genuinely
unknown needs a human rather than a retry, unless a reconciler can settle it. It is the only rule consistent with
"a crash between those two points cannot produce a duplicate write on resume" —
a duplicate application cannot be withdrawn, while a missed one can be
resubmitted deliberately.

**Resuming** is [`apps/worker/workflow_runner.py`](apps/worker/workflow_runner.py):
`DurableWorkflowRunner.resume(workflow_id=...)` takes the workflow's lease,
checks the checkpoint says there is a step left, and invokes the graph with
`None` — with no input LangGraph continues the stored run, where passing the
inputs again is what *starting* looks like.

**Two workers cannot resume one workflow.**
[`leases.py`](personalos/persistence/leases.py) holds one row per workflow. The
exclusion rests on a unique constraint on `workflow_leases.workflow_id` plus a
token-guarded single-statement `UPDATE`, not on `SELECT ... FOR UPDATE` — that is
taken on Postgres but is a silent no-op on SQLite, so a guarantee depending on it
would hold in production and nowhere else. Leases expire (a `kill -9` never
releases one, and a workflow must not be stranded), and every takeover mints a
new fencing token so a stalled holder cannot release or renew its successor's
lease.

**Tests:**

- [`tests/graph_scenarios/test_durable_resume.py`](tests/graph_scenarios/test_durable_resume.py)
  — the acceptance criteria. A real subprocess runs the workflow and `SIGKILL`s
  itself at the approval step; a restarted worker resumes and the test asserts
  the job boards were searched exactly *once* across both processes. A second
  test kills the worker after the external submission but before its receipt and
  asserts the resumed run does not submit again. A third runs two resumes of one
  workflow at the same time and asserts exactly one proceeded.
- [`tests/unit/test_durable_checkpointer.py`](tests/unit/test_durable_checkpointer.py),
  [`test_workflow_lease.py`](tests/unit/test_workflow_lease.py),
  [`test_action_journal.py`](tests/unit/test_action_journal.py),
  [`test_workflow_identity.py`](tests/unit/test_workflow_identity.py) — each
  component's own contract.

A kill is a real `SIGKILL` from a real child process, not an exception raised in
a node: an exception unwinds, runs `finally` blocks and flushes buffers, which is
the graceful path a crash does not take. Testing against it would prove nothing.

### 13. Pending Checkpoints: Durable Conditional Waits

The requirement in one sentence: *"seven days after applying, if no recruiter
response exists, draft a follow-up."* Two things in it are easy to build wrongly.

**"Seven days" is longer than any process.** So a wait is a **row**, not a
timer: nothing sleeps, nothing is parked, nothing is awaited. `pending_checkpoints`
(migration `202609290002`) holds the condition, the `trigger_at`, an explicit
`expires_at`, and the `thread_id` to come back to — a *lookup key*, resolved when
the wait comes due, never a live handle. A deploy takes every task and connection
away; it does not take the row away.

**"If no recruiter response exists" is about the moment the follow-up would be
sent, not the moment the wait was set up.** So the condition is stored
declaratively, as a `CheckpointCondition` value rather than a closure or a
pre-computed flag, and re-asked at trigger time through the
`CheckpointConditionEvaluator` port. A condition evaluated at creation time would
be blind to exactly the thing the waiting is for: a condition that answers "no
reply yet" when the wait is created will fire even if the recruiter replies the
next day.

Three pieces, in three layers:

| Layer | Piece | Owns |
| --- | --- | --- |
| `domain` | [`checkpoints.py`](personalos/domain/checkpoints.py) | the shape of a wait and `decide_checkpoint`, the whole policy as a pure function of `(checkpoint, condition_met, now)` |
| `persistence` | [`pending_checkpoints.py`](personalos/persistence/pending_checkpoints.py) | the row: idempotent `schedule` (on `dedupe_key`), the `due` query, and a guarded `close` |
| `apps` | [`checkpoint_monitor.py`](apps/worker/checkpoint_monitor.py) | the sweep: re-ask the condition, apply the outcome, start the thread |

**The policy, in order.** `decide_checkpoint` is pure and takes an explicit
`now`, so every branch is testable without waiting for any of it:

1. **Condition met → resolve, silently.** No event, no draft, no run. The reason
   for the wait went away; a "your follow-up was cancelled" for a follow-up
   nobody ever saw is noise. Applies whether or not the trigger has passed.
2. **Past `expires_at` → expire.** Even when the trigger also passed — that is
   the monitor-was-down-for-a-week case, and a week-late "just checking in" is
   worse than nothing. This is why the expiry is stored as its own column rather
   than derived: it is also what stops a checkpoint nobody swept from sitting
   `pending` forever.
3. **Due → fire.** The monitor starts the wait's thread with
   `{"fired_checkpoints": [...]}`, which routes the run into
   `draft_follow_up_for_triggered_checkpoint`. That node proposes an
   `ActionIntent` and holds no executor, so a follow-up drafted at 3am by a
   sweep with nobody watching still parks at the approval interrupt.

`resolved` and `expired` are both silences and they mean opposite things, which
is why the status is a five-value enum (`pending`, `resolved`, `fired`,
`expired`, `cancelled`) and closed rows are kept: "why was no follow-up sent?"
is answerable only if the checkpoint that decided not to send one is still there,
with its `closed_reason`.

**Exclusion.** `close` is `UPDATE ... WHERE id = ? AND status = 'pending'` and
returns its affected-row count, so two monitors that both selected the same due
checkpoint produce exactly one winner — and only the winner starts the graph
path. The claim is taken *before* the invocation, because a claim taken after it
would exclude nobody. A thread already parked on an unanswered approval is
deferred rather than fired: stacking a second request on one a human has not
answered helps nobody, and the wait's own expiry settles it if the answer never
comes.

**Tests:**

- [`tests/graph_scenarios/test_pending_checkpoints.py`](tests/graph_scenarios/test_pending_checkpoints.py)
  — the three acceptance criteria, end to end on a real durable checkpointer:
  a wait whose condition resolves during the seven days is closed with no
  follow-up event; a wait still unmet at its trigger fires, resumes the
  follow-up path (not discovery) and parks at the approval checkpoint; and a
  wait nobody swept in time is marked `expired` rather than left pending.
- [`tests/unit/test_pending_checkpoints.py`](tests/unit/test_pending_checkpoints.py)
  — the value's invariants, every branch of `decide_checkpoint`, and the store's
  idempotent scheduling and guarded close.

Time is moved rather than waited for: every decision takes an explicit `now`, so
a seven-day wait is a seven-day test only in the fiction.

### 14. Recruiter Events: Classification, Commitments and Correlation

Inbound recruiter mail the poller finds is handled by two nodes of the Job
Search graph. A thread someone asks the Communications subgraph about is
recorded the same way, with the same store and dedupe key, by its
`link_application` node:

```
START -> classify_recruiter_events -> record_recruiter_events
      -> [a reply is owed] request_approval_for_external_write -> ...
```

Invoke the graph with `inbound_messages` (dumped `RecruiterMessage`s) and a
`user_id`, on `personalos.domain.workflow.recruiter_inbox_thread_id(user_id)`.

| Question | Answered by | How |
| --- | --- | --- |
| What is this message? | `RecruiterEventExtractor` port | `StructuredLLMRecruiterEventExtractor` binds a chat model to the `RecruiterEventExtraction` JSON schema and validates the result with Pydantic. Invalid output is retried up to `max_attempts` (default 3), then `RuleBasedRecruiterEventExtractor` — deterministic phrase rules — answers instead. The outcome records which one was used. |
| What does it commit anyone to? | the same extraction | `Commitment`: `actor` (`user` \| `external_person`), `action`, `due_at`, `condition`, `confidence`, `source_message_id`. The source id is stamped by code from the message, never taken from the model. Stored in `commitments`. |
| Which application is it about? | `correlate_application` | Deterministic identifiers only — a thread already on file, the posting's requisition id quoted in the text, a known sender, the sender's domain, the company name, the title — combined into a confidence. No model is asked. |
| What happens to it? | `triage` | Matched at or above `DEFAULT_CORRELATION_THRESHOLD` (0.75) and classified at or above `DEFAULT_MIN_CLASSIFICATION_CONFIDENCE` (0.6): recorded, and its transition proposed. Anything less: a `RecruiterEventReview` and an `application.recruiter_event_review_required` event, and no transition. Unrelated mail is dropped. |

Classifications are `RECRUITER_RESPONSE`, `INTERVIEW_INVITE`, `REJECTION`,
`OFFER`, `ACTION_REQUIRED`, `GENERAL_UPDATE` and `UNRELATED`
(`CommunicationEventClassification`).

**Exactly once.** `SqlRecruiterEventStore.record` writes the
`communication_events` row, its commitments, the application's transition and
the outbox rows in one transaction, keyed on a unique
`communication_events.dedupe_key` derived from the provider message id. A
redelivered message finds the first delivery's row and writes nothing.

**The lifecycle still decides.** A classification proposes a status; it goes
through `ApplicationRepository.update_status` like every other transition. An
offer email for an application that never reached `INTERVIEWING` is recorded,
left where it is, and raised for review.

**For the calendar step.** A confident interview invite stages
`application.interview_invite_received`, whose payload is
`InterviewInviteReceived` — application, thread, sender and the extracted
commitments with their times.

**Replies.** A reply the sender is waiting on is drafted automatically (the
`RecruiterReplyDrafter` port) and proposed as a `SEND_RECRUITER_MESSAGE`
intent. Sending it goes through the approval interrupt.

The drafter a worker binds is
[`retrieval/reply_drafting.py`](personalos/retrieval/reply_drafting.py)
(`GroundedReplyDrafter`, built by `bootstrap.build_reply_drafter`) once the
model configuration is not `rules`; under `rules` the port stays unbound.

```
event -> EvidenceReplyGrounding: the application (company, role, skills) and
         the candidate's evidence nearest the message
      -> ReplyWriter (ModelGateway.reply_writer(): the primary model at
         temperature 0.5, schema ReplyProposal -- segments citing evidence ids;
         no recipient, no subject, no tools)
      -> finalize_reply: the Phase E claim validator. Rejected -> nothing is
         kept and the template reply is drafted instead
      -> artifact_versions row (type recruiter_reply), evidence = the cited
         records + the message it answers            [WRITE_REVERSIBLE]
      -> Gmail draft in the thread (gmail_create_draft)    [WRITE_REVERSIBLE]
      -> ReplyDraft -> SEND_RECRUITER_MESSAGE intent, its payload carrying the
         evidence ids, the artifact version and the Gmail draft id
```

**Sending.** Approved `SEND_RECRUITER_MESSAGE` and `SEND_EMAIL` actions are
carried out by [`executor/gmail_outbound.py`](personalos/executor/gmail_outbound.py)
(`GmailSendExecutor`, inside `ToolExecutor`, with `GmailSentReconciler`).
`SEND_EMAIL` is the general-purpose kind (`domain.outbound_email.send_email_intent`):
`google.gmail_send_message`, `WRITE_EXTERNAL`, `require_approval`.

| Question | Answer |
| --- | --- |
| What is sent? | `OutboundEmail.from_intent`: recipient, subject, body, reply-to and thread, read from the approved payload. A payload key outside `EMAIL_PAYLOAD_KEYS` -- a credential, an account, `from`, `bcc`, headers -- refuses the send. Whose mailbox sends is the gateway's `ToolAuthorization`, never an argument. |
| Can it change after approval? | Not without a new approval: recipient, subject, body and attachments are in the hash (`ActionIntent.fingerprint`). `authorize_execution` refuses a changed action with `action_hash_mismatch`, and the executor, policy and gateway each refuse the old approval too. |
| Who may send? | The principal `executor:gmail_send` (`GMAIL_SENDER`), stamped only by the send executor after checking the approval. The drafting node's allowlist is `gmail_create_draft` alone. |
| Exactly once? | Every message carries a `Message-ID` derived from the action's idempotency key. The executor searches the Sent folder (`in:sent rfc822msgid:`) before every send; sends get no blind retry (`SEND`, 0 retries); and after a failure or a lost response, the next attempt reconciles against the Sent folder -- found: its receipt is recorded; missing twice, a moment apart: sent once. |
| Attachments? | Shown in the approval preview. The Gmail tool cannot carry them, so an approved email with attachments is refused rather than sent without them. |

Drafting and sending are tested in `tests/unit/test_gmail_outbound.py`, against
the fake Google API in-process and over MCP stdio: no send without an
approval, a mutated action refused on its hash, a lost send response
reconciled from the Sent folder and not sent twice, and an untraceable claim
rejected by the validator.

**A reviewer's answer** to a review request is passed back as
`confirmed_correlations: {provider_message_id: application_id}` alongside the
message; the id must be one of that user's applications.

**Evaluation.** `python -m evals.recruiter_event_classification` reports
`event_classification_f1` (macro F1) over the 35 synthetic emails in
`evals/golden/recruiter_emails.json`; `--llm` evaluates the Claude-backed
extractor instead of the rules. `tests/unit/test_recruiter_event_eval.py`
pins the rules' score.

**Wiring.** `bootstrap.build_recruiter_event_store()` is both the
`application_directory` and the `recruiter_event_recorder`; pair it with an
extractor from `personalos.models.recruiter_events`.

**Where the mail comes from.** Gmail, through
[`executor/gmail_inbox.py`](personalos/executor/gmail_inbox.py)
(`GmailRecruiterInbox`, built by `bootstrap.build_gmail_recruiter_inbox`). Its
`poll` is a worker sweep; its `read` is the graph's `InboundMessageReader`:

```
gmail history (since mailbox_cursors.history_id)
  -> in MailboxScope? (labels from the history; queries by Gmail search)
  -> email.received in the outbox + the new cursor, one transaction
  -> EmailReceivedHandler -> START on job_search_thread_id(user, "email:<dedupe_key>")
       with {"inbound_message_refs": ["gmail:<message id>"]}
  -> classify_recruiter_events reads the message (gmail_read_message), classifies it
```

| Question | Answer |
| --- | --- |
| What is in view? | `MailboxScope`: the labels in `GMAIL_RECRUITER_LABELS` and the searches in `GMAIL_RECRUITER_QUERIES`. Empty means nothing; there is no whole-inbox default. Sent mail, drafts, spam and trash never are. `cli inbox status` prints the rules. |
| What does the event carry? | A pointer. Type `email.received`, source `gmail`, `dedupe_key` `email.received:gmail:<message id>`, `raw_ref` `gmail:<message id>`, and a payload of `user_id`, `provider_message_id`, `thread_id`, `matched_rules` and `history_id`. No subject, sender or body. |
| Is a body kept? | No. **Decision (#113): provider ids, and fetch on demand.** The body is read when the classifier needs it and left out of the checkpoint, the outbox and the command; what is kept is the `communication_events` row and its commitments. A message deleted at Gmail before it is classified is skipped. `personalos.domain.mailbox` records the reasoning. |
| Exactly once? | Per message id, three times over: the poller stages its events and the cursor in one commit, so a crash before the cursor is written leaves neither; the outbox refuses a second `dedupe_key`, so a replayed page or a resync stages nothing new; and the command and the `communication_events` row are keyed from the same id. |
| What can a hostile email do? | Be classified. The poller's node (`worker:gmail_inbox_poller`) may list history, labels and search hits, and cannot read content or write. The read is made as `classify_recruiter_events`, whose allowlist is `gmail_search` and `gmail_read_message`. Text in a message cannot change either: allowlists are keyed on provenance the code sets. |

The per-application `RecruiterInbox` port (`fetch(application_id)`, the older
`handle_recruiter_response` branch) is still unbound: mail now arrives by
event, not by asking per application, and binding both would classify a
message twice.

**Tests:** `tests/unit/test_recruiter_events.py` (rules, extractors, store),
`tests/graph_scenarios/test_recruiter_events.py` (the compiled graph against a
real database), and the two node contract classes in
`tests/unit/test_job_search_nodes.py`. The Gmail path:
`tests/graph_scenarios/test_gmail_recruiter_inbox.py` (mailbox to recorded
transition, end to end: a message delivered twice, a poller crash before the
cursor is written, out-of-scope mail, a stored-body scan, and prompt
injection) and `tests/unit/test_gmail_inbox.py` (scope, backfill, resync,
paging, the poller's allowlist, the handler).

### 15. Conditional Follow-ups and Interview Scheduling

Interview scheduling driven by recruiter mail and calendar changes lives in
the Job Search graph: a follow-up, an interview and its prep blocks all exist
to move one application forward. A person asking for a change directly goes
to the Calendar subgraph (K3), which reuses the same planner.

#### Follow-ups that cancel themselves

The wait is a [pending checkpoint](#14-pending-checkpoints-durable-conditional-waits)
and its condition is a stored `CheckpointCondition` -- `kind`, `subject_id`,
`since`, `params`. What this phase adds is the thing that answers it.
[`persistence/checkpoint_conditions.py`](personalos/persistence/checkpoint_conditions.py)
(`SqlCheckpointConditionEvaluator`, built by
`bootstrap.build_checkpoint_condition_evaluator`) is the evaluator a deployment
hands `PendingCheckpointMonitor`:

| Condition | Met when |
| --- | --- |
| `RECRUITER_RESPONSE_RECEIVED` | a `communication_events` row for the application, not classified `unrelated`, with `occurred_at >= since` |
| `CANDIDATE_REPLY_SENT` | a completed `tool_executions` row keyed `reply-<application_id>-...`, updated since |
| `APPLICATION_CLOSED` | the application's status is terminal (or the application is gone) |

So "seven days after applying, if no recruiter response, draft a follow-up" is:
the search run schedules the wait; a reply arriving on day two is recorded by
the inbound-mail path on a different thread and touches no checkpoint; the
sweep on day seven asks the database, finds the reply, and closes the wait
`resolved` with no draft, no approval request and no event.

An `INTERVIEW_PREP` wait that fires is a reminder to the candidate, not a
message to a recruiter: `draft_follow_up_for_triggered_checkpoint` emits
`application.interview_reminder` for it and proposes nothing.

#### Interview scheduling

```
record_recruiter_events -> [invite with a time] plan_interview_schedule
                        -> request_approval_for_external_write -> ...
START -> [calendar_changes] plan_interview_schedule -> ...
```

| Piece | Where | Does |
| --- | --- | --- |
| Planner | [`domain/interview_scheduling.py`](personalos/domain/interview_scheduling.py) | `plan_interview_schedule(request, calendar_events, now=...)`: pure. Reconciles the interview and its prep blocks against the calendar and returns a create / update / keep per event, plus conflicts. |
| Node | `JobSearchGraph.plan_interview_schedule` | Reads the calendar through the `CalendarReader` port, plans, turns each write into a `CREATE_CALENDAR_EVENT` / `UPDATE_CALENDAR_EVENT` intent, emits status events, (re)schedules the reminder. Holds no executor. |
| Executor | [`executor/calendar.py`](personalos/executor/calendar.py) | `CalendarActionExecutor` performs approved writes through the `CalendarClient` port; `CalendarReconciler` tells `ToolExecutor` whether an unrecorded write landed. |

**An invite is schedulable only if it fixes a time.** `interview_time_from`
takes a commitment as the interview when its action reads like one and not
like arranging one. "Send your availability by Friday" schedules nothing; the
drafted reply handles it.

**The calendar is the record of the schedule.** Every event created carries
private extended properties: `personalos_application_id`, `personalos_key`
(`interview:<application>` or `prep:<application>:<slot>`), `personalos_role`,
`personalos_interview_key` on a prep block, and `personalos_idempotency_key`.
Nothing else stores which prep block belongs to which interview, so a changed
time needs no bookkeeping: the events are found by application, matched by
key, and moved. `CalendarReader.events_for_application` exists for that -- an
interview that moved two weeks leaves its old events outside any window around
the new time.

**Prep blocks** (`DEFAULT_PREP_BLOCKS`): 90 minutes ending at least 12 hours
before the interview, inside `WorkingHours`; and 30 minutes ending 15 minutes
before it. Each goes in the latest free slot that fits. An existing block is
left alone while it still fits and is moved when it does not -- the interview
moved, or something was booked over it. A block with nowhere to go, or
anything booked over the interview itself, is a `ScheduleConflict`.

**Writes need approval.** Both kinds map to `google.calendar_create_event` /
`google.calendar_update_event`, which are `WRITE_EXTERNAL`, and leave through
the approval triple with the requested scope `calendar:write`.

**Events, not only the calendar UI.**

| Event | When |
| --- | --- |
| `application.interview_schedule_proposed` | a plan has writes to make; payload is the `InterviewSchedulePlan` |
| `application.interview_schedule_conflict` | the interview overlaps something, or a prep block could not be placed |
| `application.interview_reminder` | the prep reminder came due |
| `action.succeeded` | each calendar write that took effect (from `ToolExecutor`) |

The reminder is a pending checkpoint keyed on the interview time, triggered at
the first prep block and expiring when the interview starts. A moved interview
schedules a new one through `PendingCheckpointScheduler.reschedule`, which
cancels the old.

**No duplicates on retry.** A calendar insert is not idempotent at the
provider. `CalendarActionExecutor` looks for an event stamped with the
intent's idempotency key before it inserts, and reads an event by id before it
moves it. Under `ToolExecutor` (`bootstrap.build_calendar_action_executor`),
an attempt with no recorded outcome is settled by `CalendarReconciler` from
the same lookups: found means the receipt is recorded and nothing is sent;
positively not found means it is sent; a calendar that cannot be asked means
nothing is sent.

**Wiring.** Pass a `CalendarReader` as `calendar_reader=` to turn scheduling
on, and `build_calendar_action_executor(client, fallback=...)` as
`action_executor=`. The Google MCP server (`mcp_servers/google`) provides the
calendar tools. Nothing implements `CalendarReader` or `CalendarClient` over
those tools yet, and nothing produces `calendar_changes`. That is J4.

**Not handled:** a cancelled interview (its events are left in place), and a
new booking over an existing prep block is only noticed the next time that
application's schedule is planned.

**Tests:**

- [`tests/graph_scenarios/test_conditional_follow_ups.py`](tests/graph_scenarios/test_conditional_follow_ups.py)
  -- a recruiter reply before `trigger_at` closes the follow-up without
  drafting anything, against the real evaluator; and the control where no
  reply arrives and it is drafted.
- [`tests/unit/test_calendar_executor.py`](tests/unit/test_calendar_executor.py)
  -- a create that succeeds with its response lost is reconciled on retry and
  not duplicated, through the real `ToolExecutor` and ledger.
- [`tests/graph_scenarios/test_interview_scheduling.py`](tests/graph_scenarios/test_interview_scheduling.py)
  -- a changed interview time moves the interview and its linked prep blocks,
  whether a message or the calendar moved it.
- [`tests/unit/test_interview_scheduling.py`](tests/unit/test_interview_scheduling.py)
  (the planner), `TestPlanInterviewSchedule` in
  `tests/unit/test_job_search_nodes.py`, and the evaluator and `supersede`
  cases in `tests/unit/test_pending_checkpoints.py`.

### 16. Evaluation Harness and the Model Gateway

Agent quality is measured against a version-controlled golden dataset, not
demoed. One command runs all of it against one model configuration:

```
python -m evals.harness                                    # the deterministic `rules` config
python -m evals.harness --config evals/model_benchmarks/configs/ollama-qwen2.5-7b.json
python -m evals.harness compare A/results.json B/results.json
```

**The model is configuration.** [`models/gateway.py`](personalos/models/gateway.py)
builds the four model-backed ports -- intent classifier, recruiter event
extractor, semantic assessor, draft writer -- from a `ModelConfig` JSON file:
a default `ModelSpec` (`provider`, `model`, `base_url`, `params`) and optional
per-role overrides. Providers are `rules` (the deterministic stand-ins; no
draft writer), `anthropic` (the `llm` extra), and `ollama` and
`openai_compatible` for local models (the `local` extra). A deployment points
`LLM_GATEWAY_CONFIG` at a file and takes its ports from
`bootstrap.build_model_gateway()`; the harness does the same per run. Swapping
models changes no graph, node or prompt.

**The golden dataset** ([`evals/golden/`](evals/golden/)):

| File | Cases | Expected |
| --- | --- | --- |
| `routing/<domain>.json` | user requests, per domain (`job`, `file`, `communications`, `calendar`, plus `general`): clear, ambiguous, cross-domain, adversarial, out-of-scope and negative | Supervisor route (a subgraph, a cross-domain `plan` with its step domains, or `clarify` with its reason) and domain; `never_routes` a request must not reach |
| `recruiter_emails.json` | 35 emails | event classification |
| `candidate_profiles.json`, `job_descriptions.json` | 2 candidates × 10 postings | acceptable recommendations, top-k relevance |
| `scenarios.json` | end-to-end runs | classification, tool intents, approval pause, final application status |
| `adversarial.json` | injected mail, postings, a draft request | expected-safe rules: no new intents, no redirected reply, no attacker address in an action, no forged transition, no fabricated draft |
| `human_review.json` | top-k and drafts | automatic checks, plus a rubric a person rates |
| `approval_boundaries.json` | labelled tool calls, every `ActionKind` | consequential / external write, scored against the policy engine and the graph's approval requests |
| `recovery.json` | failed-and-resumed runs, redeliveries | end state, whether resuming alone finished the work, no duplicated side effect |
| `domain_workflows.json` | Calendar, Communications and File subgraph runs, approved / rejected / pending / fault-then-resume | proposed actions, approval before any effect, effects that reached the fakes, recovery |

**What a run produces** (`evals/model_benchmarks/results/<config>/`, not
tracked): `results.json` with every case's expected and actual outcome and
each suite's metrics; `summary.md`; and `review_packet.md` with the
human-review outputs, their rubric and a fingerprint. A reviewer's ratings go
in `evals/model_benchmarks/reviews/<config>.json` and count only while the
output still has the fingerprint they rated.

**Before a model or prompt change.** Each configuration's baseline is
`evals/model_benchmarks/baselines/<config>.json`. A run diffs against it and
exits 1 on a regression (a case that passed now does not), on any unsafe
adversarial case, or on a case that raised. A failure the baseline already had
is reported, not gated, so a weak model can still be benchmarked.
`--update-baseline` accepts a run.

**Quality metrics** ([`evals/metrics.py`](evals/metrics.py)). Every run also
computes a `quality` section from its cases and from the spans the graphs
emitted (captured in memory, saved as `spans.otlp.json`): routing accuracy,
overall and per domain, with a confusion matrix and a count of requests that
reached a route their case forbids;
approval precision, recall and external-write recall, overall and per domain
(the worst of the policy boundary and the domain's subgraph); tool-call validity
(the `schema_valid` attribute on each model attempt's span) and tool
selection accuracy; workflow completion, recovery rate and mean recovery
time, each also per domain; p50/p95 latency for interactive steps (routing,
extraction, the calendar, mail and file reads) and long-running ones
(discovery, drafting, external writes), per-model latency and tokens/sec
where usage is on the span; `job_match_precision@k`,
`event_classification_f1`, `resume_evidence_grounding` and
`duplicate_side_effect_rate`. `python -m evals.harness quality` recomputes it
from a results file and/or OTLP/JSON exported by a Collector from a
deployment. Each run records its model labels and per-role prompt versions
(`gateway.prompt_versions`, a hash of the system and repair prompts) and is
appended to `results/history.jsonl`; `python -m evals.harness trend` lists
runs side by side. [`evals/thresholds.json`](evals/thresholds.json) bounds the
metrics: a critical breach -- approval recall or external-write approval
recall below 1.0 (overall or in any domain), per-domain routing accuracy below
its floor, a forbidden route reached, or any duplicated side effect -- fails
the run and the CI "Quality metrics gate" step; the rest are warnings in the
report. Every `ActionKind` must also have a row in
`personalos/policy/actions.py` that policy holds for approval
(`tests/contracts/test_action_policy_coverage.py`, the CI "Action policy
gate"). Adding a domain end to end is in
[`docs/ADDING_A_DOMAIN.md`](docs/ADDING_A_DOMAIN.md).

**Tests:** [`tests/unit/test_eval_harness.py`](tests/unit/test_eval_harness.py)
pins the `rules` run to its baseline, and runs the suites with a scripted
honest model and one that obeys injected text, which the adversarial suite
must catch. [`tests/unit/test_model_gateway.py`](tests/unit/test_model_gateway.py)
covers the gateway and checks every shipped config builds.
[`tests/unit/test_quality_metrics.py`](tests/unit/test_quality_metrics.py)
checks approval recall on a fixture of known consequential and harmless
actions, each metric against hand-computed inputs, and that a critical breach
fails the run.

---

## MCP Framework Guide

### Overview

The **Model Context Protocol (MCP)** enables AI agents to execute external capabilities through modular tool servers. PersonalOS provides a complete MCP framework.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            MCP Server Manager (Orchestrator)                 │
│  - Routes tool calls to appropriate servers                  │
│  - Manages server lifecycle                                  │
│  - Caching and result aggregation                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌────────┐    ┌────────┐    ┌────────┐
   │ Jobs   │    │ Files  │    │Google  │
   │  MCP   │    │  MCP   │    │  MCP   │
   └────────┘    └────────┘    └────────┘
```

### Core Components

#### 1. MCPServer Base Class

```python
class MCPServer(ABC):
    def register_tool(self, schema: ToolSchema, handler: callable)
    async def execute(self, tool_name: str, **kwargs) -> Dict
    def get_tools(self) -> List[Tool]
    @abstractmethod
    def initialize(self): pass
```

**Features:**
- Tool registration with automatic schema validation
- Async-first execution (supports both sync and async handlers)
- Error handling with try/except
- Logging of all tool executions
- Parameter validation before execution

#### 2. ToolSchema

Defines tool interface using JSON Schema:

```python
schema = ToolSchema(
    name="search_jobs",
    description="Search for jobs",
    parameters={
        "type": "object",
        "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}},
            "locations": {"type": "array", "items": {"type": "string"}},
        },
    },
    required=["keywords", "locations"],
)
```

#### 3. Caching System

Reduces API calls and improves performance:

```python
from personalos.mcp.cache import get_cache

cache = get_cache()  # Returns Redis or In-Memory based on env

# Get cached result
result = await cache.get("mcp:search_jobs:hash123")

# Set with TTL
await cache.set("mcp:search_jobs:hash123", result, ttl=3600)

# Delete
await cache.delete("mcp:search_jobs:hash123")
```

**Implementations:**
- `RedisCache` - Distributed, production-ready
- `InMemoryCache` - Development fallback

#### 4. MCPServerManager

Central orchestrator:

```python
from personalos.mcp.manager import get_mcp_manager

manager = get_mcp_manager()

# Register server
manager.register_server(build_files_mcp_server())

# Execute tool
result = await manager.execute_tool(
    ToolCallRequest(target=ActionTarget(server="files", tool="read_file"), params={...})
)

# Get all tools
tools = manager.get_all_tools()

# Get server info
server_info = manager.get_server_info("files")
```

### Creating a Custom MCP Server

**Template:**

```python
from personalos.mcp.base import MCPServer, ToolSchema
from personalos.mcp.manager import get_mcp_manager


class CustomMCPServer(MCPServer):
    def __init__(self):
        super().__init__("custom", "Description of this server")
        self.initialize()

    def initialize(self):
        """Register all tools with this server."""
        tool_schema = ToolSchema(
            name="my_tool",
            description="What this tool does",
            parameters={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "integer", "description": "Second parameter"},
                },
            },
            required=["param1"],
        )
        self.register_tool(tool_schema, self._my_tool)

    async def _my_tool(self, param1: str, param2: int = 10):
        """Implement tool logic."""
        # Your implementation here
        return {"result": f"Processed {param1} with {param2}"}


# Register globally
manager = get_mcp_manager()
manager.register_server(CustomMCPServer())
```

### Best Practices

**1. Caching**
Cache expensive API calls to reduce costs and latency:

```python
cache = get_cache()
cache_key = f"mcp:search_jobs:{hash(str(kwargs))}"

cached = await cache.get(cache_key)
if cached:
    return cached

result = await external_api.search(**kwargs)
await cache.set(cache_key, result, ttl=3600)
return result
```

**2. Error Handling**
Always handle errors gracefully with informative messages:

```python
try:
    result = await external_api.call()
except TimeoutError:
    logger.error("API timeout")
    return {"success": False, "error": "API timeout", "retryable": True}
except Exception as e:
    logger.error(f"API error: {e}")
    return {"success": False, "error": str(e)}
```

**3. Logging**
Log all tool executions for debugging:

```python
logger.info(f"Executing tool '{tool_name}' with params {kwargs}")
try:
    result = await handler(**kwargs)
    logger.info(f"Tool '{tool_name}' succeeded")
except Exception as e:
    logger.error(f"Tool '{tool_name}' failed: {e}", exc_info=True)
```

**4. Parameter Validation**
Always validate and document parameters:

```python
schema = ToolSchema(
    name="tool",
    parameters={
        "type": "object",
        "properties": {
            "param": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100,
                "description": "Parameter description",
            }
        },
    },
    required=["param"],
)
```

---

## Testing

The suite runs offline: a conftest guard fails any test that opens a
connection to anything other than localhost, and model-backed tests replay
recorded outputs. Tests that need PostgreSQL skip when `DATABASE_URL` is unset.

```bash
python scripts/check_boundaries.py        # layer imports, no install needed
python scripts/check_guide_tree.py        # this guide's tree against the repository
pytest tests                              # everything; Postgres-backed tests need DATABASE_URL
pytest tests/adversarial                  # the security boundary on its own
python -m evals.harness                   # golden-set quality gate on the rules config
python scripts/smoke_quickstart.py        # the quick start, end to end (needs a fresh database)
python scripts/export_openapi.py --check  # ui/openapi.json against the API
ruff check personalos apps mcp_servers tests scripts
(cd ui && npm ci && npm run typecheck && npm test && npm run build)   # the UI
```

The UI's typed client is generated, in two steps, from the API itself:
`scripts/export_openapi.py` writes `ui/openapi.json` from `create_app()`, and
`npm run api:generate` (in `ui/`) turns it into `ui/src/api/schema.d.ts`.
After changing a route or a response model, run both and commit the results;
CI fails if either file is stale.

| Suite | What it pins down |
| --- | --- |
| `tests/architecture` | Layer import rules, and that models emit intents only |
| `tests/contracts` | Every graph node has a typed contract and a contract test; transport interfaces; the offline guard |
| `tests/unit` | Pure logic, API routes, stores, worker wiring and reliability |
| `tests/graph_scenarios` | Whole-graph runs: approval interrupts, durable resume, recovery, idempotent mutations, tracing |
| `tests/adversarial` | Prompt injection in postings, approval tampering, path traversal, permission escalation, hostile tool payloads |
| `tests/migrations` | `alembic upgrade head` and downgrade against PostgreSQL |
| `ui/tests` | The approvals inbox: the intent diff, decisions bound to `action_hash`, 409 handling without resubmission, inert rendering of untrusted text, token handling. The application board: refused moves show the allowed states and leave the card in place, effectful moves go through approval. Workflow progress: live step events, `Last-Event-ID` resumption without duplicates, the polling fallback, the approval prompt. Audit and failures: the approved-send chain in order, no retry on a policy denial, a reconnect prompt for provider-permanent failures, one retry call updating the card, unredacted credentials withheld |

CI (`.github/workflows/ci.yml`) runs on pull requests and on `main`. It runs
the boundary and guide-tree checks first, then the quick-start smoke test on a
fresh pgvector database, then the full suite, the evaluation gate and lint on
another. `offline-tests.yml` runs the contract and pure-logic tests on every
push, with no database.

---

## Next Steps

This follows the phased backlog in *PersonalOS — Architecture Gap Analysis &
Phased Backlog* (Oct 29, 2026). Phases A–H are closed. Phase I makes the
system run end to end, and this guide is its last issue (I6). After Phase I,
Phases J and L can run in parallel. When an item lands, update this section,
and if the quick start changes, update `scripts/smoke_quickstart.py` too.

### Phase I: End-to-end composition (done with this guide)

| # | Issue | Where it shows up now |
| --- | --- | --- |
| I1 | Default worker graph factory | `apps.worker.graphs:build_worker_graphs`, the default `WORKER_GRAPHS` |
| I2 | Search profile store | `/v1/profile`, `personalos.cli profile` |
| I3 | Resume and evidence ingestion | `personalos.cli ingest-resume`, `POST /v1/evidence/documents` |
| I4 | Retire the legacy jobs path | `/api/v1/jobs` and the mock jobs server are gone; searches run as workflows |
| I5 | Remove stale scaffolding | No Celery; no empty `personalos/state` |
| I6 | Refresh the implementation guide | This guide, the quick start, `scripts/smoke_quickstart.py` and `scripts/check_guide_tree.py` |

### Phase J: Real integrations

These fill in the ports the worker reports as unbound at startup.

| # | Issue | Done when |
| --- | --- | --- |
| J1 | Google OAuth onboarding | `cli auth google` runs the consent flow and stores the refresh token in the keychain as `cred://` |
| J2 | Google MCP server | `mcp_servers/google` registers Gmail and Calendar tools behind the gateway, with permission classes |
| J3 | Gmail `RecruiterInbox` adapter | Polls or watches labelled mail, feeds classification, and is idempotent per message id |
| J4 | Google Calendar client and reader | `CalendarClient` and `CalendarReader` implemented; the reconciler finds events by idempotency stamp |
| J5 | Outbound reply drafting and send | `RecruiterReplyDrafter` implemented; `send_email` is `require_approval` and deduped by key |
| J6 | Embedder implementations | Local and hosted embedders beyond the `hashing` one I3 shipped, with a backfill command, since changing the embedder means re-embedding |
| J7 | More job-board providers | Lever and Ashby, with normalisation fixtures in the golden set. Only boards with public APIs, not LinkedIn or Indeed |
| J8 | Integration health checks | Each adapter reports to `integration_health`, and a degraded adapter shows up on `/failures` |

### Phase K: Reserved route domains

| # | Issue | Done when |
| --- | --- | --- |
| K1 | File subgraph and planner | Done (`personalos/graphs/file_task.py`; see [Orchestration Graphs](#11-orchestration-graphs-personalosgraphs)). `RouteDomain.FILE` routes to it, and every read, write, move and delete goes through the sandboxed files MCP |
| K2 | Communications subgraph | Done (`personalos/graphs/communications.py`; see [Orchestration Graphs](#11-orchestration-graphs-personalosgraphs)). `RouteDomain.COMMUNICATIONS` routes to it. Triage, thread summaries, action items and reply drafts run over the J2/J3/J5 Gmail pieces, and a send needs approval |
| K3 | Calendar subgraph | Done (`personalos/graphs/calendar.py`; see [Orchestration Graphs](#11-orchestration-graphs-personalosgraphs)). `RouteDomain.CALENDAR` routes to it. Availability and proposed times from free/busy in the configured zone; creates, moves and cancels need approval, are made exactly once, and never touch an event without the PersonalOS stamp |
| K4 | Cross-domain task DAGs | Done (`personalos/domain/task_plan.py`, `personalos/graphs/plan_steps.py`; see [Orchestration Graphs](#11-orchestration-graphs-personalosgraphs)). The Supervisor plans a validated DAG of steps across domains. Each step runs through its subgraph on a thread of the one workflow, with its own approval, durable resumption, dependent-only cancellation, and a proposed release for a dangling calendar hold |
| K5 | Classifier and evals for new domains | Each domain has routing golden cases, and the quality report covers them |

### Phase L: API security

| # | Issue | Done when |
| --- | --- | --- |
| L1 | Authentication | Done (local token; see [Security Posture](#security-posture)). OIDC replaces `LocalTokenVerifier` behind the same `TokenVerifier` seam |
| L2 | Actor-scoped authorization | Done ([docs/AUTHORIZATION.md](docs/AUTHORIZATION.md)). Scoped stores return 404 across actors, tokens carry `read`/`write`/`approve`/`admin`, system identities cannot approve, and denials are audited |
| L3 | Rate limiting and request limits | Done ([docs/LIMITS.md](docs/LIMITS.md)). Per-actor token buckets, active-workflow and SSE caps, auth-failure backoff, JSON depth/string caps, bounded page sizes, idempotent chat replays. The buckets are per process for now, behind `RateLimitStore` |
| L4 | CORS and security headers | Done (`apps/api/security.py`; see [Security Posture](#security-posture)) |

### Phase M: Operator UI

| # | Issue | Done when |
| --- | --- | --- |
| M1 | Approvals inbox | Done (`ui/`; see [Review it in the browser](#optional-review-it-in-the-browser)). Lists pending approvals with the intent diff; approve and reject call the API with the listed `action_hash` |
| M2 | Application board and workflow progress | Done (`ui/`, `#/board`). A kanban of `GET /v1/applications` with filters and pages, an application detail view, moves through `POST .../transition` (effectful ones only via approval), and live progress from the SSE stream with `Last-Event-ID` resumption and a polling fallback |
| M3 | Audit and failure views | Done (`ui/`, `#/workflows/{id}`, `#/audit`, `#/integrations`). The per-workflow audit chain and `GET /v1/audit` search with filters and pages, failures with Retry and Dismiss (denials shown as denied with no retry, a reconnect prompt for provider-permanent ones), integration health, and a client-side guard that withholds any unredacted credential |

### Phase N: Packaging and operations

| # | Issue | Done when |
| --- | --- | --- |
| N1 | Container images and compose (done) | `docker compose up` starts the API, the worker, Postgres with pgvector, and a collector; see [docs/CONTAINERS.md](docs/CONTAINERS.md) |
| N2 | Dashboards and alerts | Dashboards for latency, error rate and dead letters, and alerts on stalled workflows |
| N3 | Backup and data retention | A scheduled Postgres backup, and a retention policy for the event log, traces and drafts |
| N4 | Second event transport (optional) | A NATS JetStream or Redis Streams adapter passes the transport contract tests |

### Not yet in the backlog

The quick start turned up these gaps, and none of the issues above covers
them:

- **Application submitter and application store** (`application_submitter`,
  `application_store`). An approved submission has no adapter to run it, so
  the quick start ends there.
- **Submission idempotency per candidate.** The `submit_application` key
  (`submit-<dedupe_key>`) is derived from the posting alone. Two searches
  against one database, or two candidates applying to the same posting,
  collide with an `IdempotencyConflict`.

### Open questions

- Is a second event transport (N4) still wanted, or can it be closed?

---

## Configuration

Settings come from the environment, then `.env` in the working directory.
`.env.example` lists every one. These are the ones you are likely to set:

| Setting | Default | Notes |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://user:password@localhost:5432/personalos` | A bare `postgresql://` URL is rewritten to name the installed `psycopg2` driver. |
| `API_HOST` / `API_PORT` / `API_WORKERS` | `127.0.0.1` / `8000` / `4` | A non-loopback host is refused while `API_AUTH_ENABLED=false`. |
| `API_AUTH_ENABLED` | `true` | `false` makes every request `API_ACTOR`; loopback only. |
| `API_TOKEN_STORE` / `API_TOKEN_FILE` | `file` / `~/.config/personalos/api-token` | Or `keychain` for the OS keychain. `personalos.cli auth token [--rotate]`. |
| `API_ACTOR` | `api` | The actor the local token authenticates as. |
| `API_TOKEN_SCOPES` | `read,write,approve,admin` | Scopes the local token carries; see [docs/AUTHORIZATION.md](docs/AUTHORIZATION.md). |
| `API_STREAM_TICKET_SECONDS` | `60` | Lifetime of an event-stream ticket. |
| `API_MAX_BODY_BYTES` / `API_MAX_RESUME_BODY_BYTES` | 1 MiB / 10 MiB | Body caps; the second only for `POST /v1/evidence/documents`. |
| `API_LIMITS_ENABLED` | `true` | Per-actor rates, concurrency caps and the auth throttle. Every `API_RATE_*`, `API_MAX_*` and `API_AUTH_*` limit setting is in [docs/LIMITS.md](docs/LIMITS.md). |
| `FILES_ALLOWED_ROOTS` | `[]` | JSON list of absolute paths. If empty, every file read and write is refused. |
| `FILES_MAX_UNATTENDED` | `3` | The most files a File subgraph plan may change before it needs approval. Deletes and overwrites always do. |
| `SUPERVISOR_MAX_PLAN_STEPS` / `SUPERVISOR_PLAN_REPAIR_ATTEMPTS` | `6` / `2` | The most steps a cross-domain plan may have (1-12), and how many times an invalid plan is handed back to the planner before the request is asked back. |
| `WORKER_GRAPHS` | `apps.worker.graphs:build_worker_graphs` | `module:callable`. Empty runs no graphs. |
| `GREENHOUSE_BOARDS` | `{}` | JSON, board token to employer name. |
| `LEVER_SITES` | `{}` | JSON, Lever site (`jobs.lever.co/<site>`; prefix `eu/` for EU-hosted) to employer name. |
| `ASHBY_BOARDS` | `{}` | JSON, Ashby board (`jobs.ashbyhq.com/<board>`) to employer name. |
| `WORKDAY_SITES` | `{}` | JSON, `<tenant>.wd<N>.myworkdayjobs.com/<site>` to employer name. Opt-in per employer: reads the career page's undocumented endpoint, not a published API. See "Job board coverage" in `docs/ARCHITECTURE_BOUNDARIES.md`. |
| `LLM_GATEWAY_CONFIG` | empty (rules) | A `ModelConfig` JSON file, such as those in `evals/model_benchmarks/configs/`. |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `ollama` / empty (`nomic-embed-text`) | `ollama`, `hashing` or `openai`. After a change, run `reembed`. |
| `CLOUD_INFERENCE_ALLOWED` | `false` | Must be `true` before a hosted embedder (`openai`) can be selected. |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `json` | Use `text` in a terminal. |
| `TRACING_ENABLED` / `OTEL_EXPORTER_OTLP_ENDPOINT` | `false` / empty | An OTLP/HTTP collector, for example Jaeger at `http://localhost:4318`. |
| `SECRET_STORE_SERVICE` | `personalos` | The OS keychain service that holds secrets. Never put a secret in `.env`. |
| `SECRET_STORE_BACKEND` | `keychain` | `file` for containers: secrets are `<provider>.<name>` files in `SECRET_STORE_DIR` (writable) and `SECRET_STORE_READ_ONLY_DIRS` (Docker secrets). See [docs/CONTAINERS.md](docs/CONTAINERS.md#secrets-in-containers). |
| `DATABASE_PASSWORD_FILE` | empty | A file whose contents replace the password in `DATABASE_URL`. The compose stack uses it. |
| `WORKER_HEARTBEAT_FILE` / `WORKER_HEARTBEAT_SECONDS` | empty / `10` | A file the worker touches while alive, which `personalos.cli worker-health` checks. The worker image sets it. |
| `GOOGLE_OAUTH_CLIENT_ID` | empty | A Desktop-app OAuth client. Its secret goes in the keychain via `auth google-client`. |
| `GOOGLE_OAUTH_PUBLISHING_STATUS` | `testing` | `testing` records Google connections as expiring after 7 days, as Google does for unverified apps. |
| `MCP_GOOGLE_ENABLED` | `false` | Registers the Gmail and Calendar MCP server. Connect an account first. |
| `GOOGLE_ACCOUNT` | empty | Which connected Google account the tools act on. If empty, the most recently connected active account is used. |
| `CALENDAR_TIME_ZONE` | empty | The IANA zone calendar events are written in, and the zone the Calendar subgraph reads phrases like "next Tuesday afternoon" in. All-day events are also read in this zone. Interview scheduling on Google Calendar and the Calendar subgraph stay unbound until it is set. The server's local zone is never used. |
| `CALENDAR_ID` | `primary` | The calendar that interview events and prep blocks go on. |
| `GMAIL_RECRUITER_LABELS` / `GMAIL_RECRUITER_QUERIES` | `[]` / `[]` | JSON lists: the Gmail labels and searches recruiter mail is found by. The only mail PersonalOS looks at; both empty leaves the inbox off. |
| `GMAIL_POLL_INTERVAL_SECONDS` | `300` | How often the worker reads the Gmail history. |
| `GMAIL_BACKFILL_DAYS` | `14` | How far back a first poll, or a resync, searches. `0` means only new mail. |
| `GMAIL_INBOX_ACTOR` | `api` | Whose applications inbound mail is matched against, as `--actor` / `API_ACTOR`. |
| `INTEGRATION_HEALTH_CHECK_SECONDS` | `300` | How often the worker health-checks every adapter. Results are at `GET /v1/integrations` and `GET /v1/failures/integrations`. |
| `INTEGRATION_CIRCUIT_RETRY_SECONDS` | `60` | How long work that needs an integration that is down, or awaiting a reconnect, waits in its queue before it is looked at again. |
| `REDIS_URL` | `redis://localhost:6379/0` | Optional MCP response cache. |

---

## Further Reading

- [README.md](README.md): what PersonalOS is, in one page
- [docs/ARCHITECTURE_BOUNDARIES.md](docs/ARCHITECTURE_BOUNDARIES.md): layer ownership and the dependency direction
- [docs/EVENT_TRANSPORT.md](docs/EVENT_TRANSPORT.md): the outbox and how to replace it
- [docs/AUTHORIZATION.md](docs/AUTHORIZATION.md): scopes, system identities, actor-scoped stores and denial auditing
- [evals/README.md](evals/README.md): the evaluation harness and model benchmarks
- [GitHub Issues](https://github.com/madhurlak0810/PersonalOS/issues)

# PersonalOS

Local-first, multi-agent personal assistant that automates job search, orchestrated with LangGraph. An LLM proposes actions; a deterministic policy engine authorizes them and an isolated executor runs them, with durable state, human-approval gates, and a full audit trail.

## Architecture

The model emits intent only. Orchestration proposes a `ToolIntent`, the policy engine turns it into an `ApprovedIntent` (or refuses), executors run only approved intents, and adapters do the I/O.

| Layer | Package | Owns |
| --- | --- | --- |
| Orchestration | `personalos/graphs` | Supervisor and domain subgraphs, state machines, interrupts |
| Policy | `personalos/policy` | Allow / require-approval / deny decisions; never executes |
| Execution | `personalos/executor` | Deterministic action execution, retries, credential brokering |
| Persistence | `personalos/persistence` | Relational state, checkpoints, leases, outbox, audit |
| Capability adapters | `personalos/tools`, `personalos/mcp`, `mcp_servers/*` | Tool gateway and MCP servers |
| Secrets | `personalos/secrets` | OS-keychain secret store and token exchange |

The boundaries are enforced, not just documented: `scripts/check_boundaries.py` and `tests/architecture` fail CI on a forbidden import. See `docs/ARCHITECTURE_BOUNDARIES.md`.

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| A | Foundation hardening: module boundaries, typed contracts, error model | Done |
| B | Data model and persistence (PostgreSQL + pgvector) | Done |
| C | Orchestration runtime with LangGraph | Done |
| D | Security boundary: policy engine and executor isolation | Done |
| E | Domain capability build-out (flagship job-search workflow) | Next |
| F | API/UX status surfaces | Planned |
| G | Eventing and workers | Planned |
| H | Observability, evaluation, and testability | Planned |

## What's implemented

### Phase A: foundation and contracts

- Layered module boundaries with an AST-based import checker run first in CI
- Typed intent/action contracts (`ToolIntent`, `ApprovedIntent`, `PolicyDecision`) for every tool call and side effect
- A single error taxonomy (`PersonalOSError` with `error_code` and `retryable`)
- `ExecutionContext` carrying workflow, correlation, and actor identity across API, executor, and tool boundaries
- Idempotency keys on mutating operations

### Phase B: data model and persistence

- Alembic migrations in `migrations/versions`
- Workflow tables: `users`, `workflows`, `workflow_runs`, `workflow_steps`, `checkpoints`, `checkpoint_writes`, `workflow_leases`, `pending_checkpoints`, `approvals`
- Job-search tables: `job_postings`, `candidate_profiles`, `applications`, `artifact_versions`, `communication_events`
- Enforcement and audit tables: `tool_executions`, `policy_decisions`, `audit_events`, `credentials`
- Immutable `event_log` with a mutable `application_status_view` projection, plus `outbox_events` for async processing
- pgvector tables for semantic retrieval: `evidence_chunks`, `job_posting_embeddings`, `message_embeddings`

### Phase C: orchestration runtime

- Supervisor graph that classifies each request into a typed `RouteDecision`, plans a bounded task DAG, and hands off to a subgraph
- Intent classification behind a port: a structured-output LLM classifier (optional `llm` extra) or a deterministic keyword classifier
- Job Search subgraph: search, normalize, deduplicate, filter, score, evidence check, rank, shortlist, approval, execute, persist
- Database-backed LangGraph checkpointer, so a run survives a crash or deploy and resumes by workflow/thread ID
- Workflow leases, so two workers cannot resume the same workflow
- Approval interrupt before any external write
- Pending checkpoints with trigger and expiry for long waits (for example "follow up after 7 days if no reply"), swept by `apps/worker/checkpoint_monitor.py`

### Phase D: security boundary

- Default-deny policy engine with three outcomes (`allow`, `require_approval`, `deny`); every verdict is written to `policy_decisions` before it is returned
- Permission classes by recoverability: `read_local`, `read_external`, `write_reversible`, `write_external`, `destructive`, `sensitive`
- `ToolGateway` as the only path to a tool adapter; an `ApprovedIntent` cannot be constructed outside the policy layer
- Credential broker: long-lived secrets stay in the OS keychain, the rest of the system holds only a `CredentialRef` (`cred://...`), and short-lived tokens are issued at execution time
- Redaction of credentials from logs, trace spans, checkpoints, and audit payloads
- Files MCP server sandbox: allowed roots only, decisions made on the canonical path (after `..` and symlinks), credential directories denied, compare-and-swap writes
- Idempotent mutations: each mutating action claims its idempotency key, and its execution row, audit event, and outbox event are committed together with the policy decision and approval that authorized it

### Service and tooling

- FastAPI service (`apps/api`) with job-search endpoints under `/api/v1/jobs` (create, get, list)
- MCP framework (base server, manager, Redis/in-memory cache) with a jobs server (`search_jobs`, `scrape_job_details`, `filter_jobs`, `save_favorite_job`) and a files server (`read_file`, `write_file`)
- Test suite of 600+ tests across `tests/unit`, `tests/graph_scenarios`, `tests/architecture`, and `tests/migrations`

## Stack

FastAPI · SQLAlchemy 2.0 · Alembic · Pydantic 2 · LangChain/LangGraph · PostgreSQL + pgvector · Redis · Celery · OpenTelemetry · MCP

## Quick start

Requires Python 3.10+ and PostgreSQL with the `vector` extension (the `pgvector/pgvector:pg16` image works).

```
pip install -e ".[dev,test]"
cp .env.example .env   # set DATABASE_URL and FILES_ALLOWED_ROOTS
python -m personalos.cli db_init
python -m personalos.cli api --reload
```

`FILES_ALLOWED_ROOTS` is a JSON list of absolute paths; left empty, every file read and write is rejected. Secrets (refresh tokens, API keys, the OAuth client secret) go in the OS keychain, never in `.env`.

Run the checks CI runs:

```
python scripts/check_boundaries.py
pytest tests
ruff check personalos apps mcp_servers tests scripts
```

See `IMPLEMENTATION_GUIDE.md` for data flow and API details.

## Not yet implemented

- File, Communications, and Calendar subgraphs: the route domains are reserved, but only the Job Search subgraph exists
- The Google MCP server (`mcp_servers/google`)
- Application lifecycle state machine, artifact tailoring, and recruiter-event classification (Phase E)
- Workflow, approval, application-board, and audit endpoints, and streaming progress (Phase F)
- Outbox publisher and worker consumer; `personalos.cli worker` is a placeholder (Phase G)
- OpenTelemetry tracing across layers, the evaluation harness in `evals/`, and adversarial tests (Phase H)

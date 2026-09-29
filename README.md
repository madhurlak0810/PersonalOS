# PersonalOS

Local-first, multi-agent personal assistant that automates job search, orchestrated with LangGraph. An LLM proposes actions; a deterministic policy engine authorizes and executes them, with durable state, human-approval gates, and full audit trails.

## Architecture

Orchestration proposes intents, policy decides, executors run only approved intents, and adapters do the I/O. See `docs/ARCHITECTURE_BOUNDARIES.md` for the enforced layer boundaries (checked in CI via `scripts/check_boundaries.py`).

## What's implemented

- FastAPI service (`apps/api`) with job-search endpoints (create, get, list)
- Domain models and SQLAlchemy/PostgreSQL persistence with a repository layer
- Event-driven pub/sub event bus
- A Model Context Protocol (MCP) framework: base server, manager, Redis/in-memory caching
- Jobs MCP server with 4 tools (search_jobs, scrape_job_details, filter_jobs, save_favorite_job)
- LangGraph supervisor and job-search subgraph
- Policy engine with default-deny allowlists and an approval gate on mutating actions
- 15 passing unit tests (pytest + pytest-asyncio) covering the MCP server, executor, and domain models

## Stack

FastAPI · SQLAlchemy 2.0 · Pydantic 2 · LangChain/LangGraph · Celery · Redis · PostgreSQL · OpenTelemetry

## Quick start

```
pip install -e ".[dev]"
cp .env.example .env   # add your database URL and settings
python -m personalos.cli db_init
python -m personalos.cli api --reload
```

See `IMPLEMENTATION_GUIDE.md` for full architecture, data flow, and API details.

## Status

Actively in development — the `files/`/`google/` MCP servers, integration/adversarial test suites, and the retrieval/observability modules are still in progress.

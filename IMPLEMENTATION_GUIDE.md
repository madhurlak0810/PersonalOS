# PersonalOS Implementation Guide

Complete documentation for the PersonalOS Agent framework, including architecture, components, setup, and development guide.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Project Structure](#project-structure)
4. [Architecture](#architecture)
5. [Completed Components](#completed-components)
6. [MCP Framework Guide](#mcp-framework-guide)
7. [Testing](#testing)
8. [Next Steps](#next-steps)

---

## Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL
- Redis (optional, for job queue)

### Installation
```bash
# Install dependencies (including dev tools)
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your database URL and settings
```

### Running the API
```bash
# Initialize database
python -m personalos.cli db_init

# Start API server
python -m personalos.cli api --host 0.0.0.0 --port 8000 --reload

# Or with uvicorn directly
uvicorn apps.api.main:app --reload
```

### API Endpoints
```bash
# Create a job search
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Search Python Developer Jobs",
    "keywords": ["Python", "FastAPI"],
    "locations": ["Remote", "NYC"],
    "salary_min": 80000,
    "salary_max": 150000,
    "job_type": "full-time"
  }'

# Get job details
curl http://localhost:8000/api/v1/jobs/{job_id}

# List all jobs
curl http://localhost:8000/api/v1/jobs/
```

---

## Project Overview

PersonalOS is a local-first agentic assistant framework designed for job search automation. The project uses a **Model Context Protocol (MCP)** architecture to provide AI agents with modular, reusable tool capabilities.

### Key Technologies

- **FastAPI** 0.141+ - Async HTTP API
- **SQLAlchemy** 2.0+ - PostgreSQL ORM
- **Pydantic** 2.0+ - Type validation
- **LangChain/LangGraph** - Agent workflows
- **Celery** 5.3+ - Background job processing
- **Redis** - Caching and job queue
- **OpenTelemetry** - Observability

### Core Principles

- **Async-First** - All operations are async/await
- **Type-Safe** - Full type hints and Pydantic validation
- **Event-Driven** - Pub/sub architecture for decoupling
- **Modular MCP** - Pluggable tool servers
- **Repository Pattern** - Clean data access abstraction

---

## Project Structure

```
PersonalOS-agent/
├── pyproject.toml              # Python project config & dependencies
├── .env.example                # Environment variables template
├── README.md                   # Project overview
├── IMPLEMENTATION_GUIDE.md     # This file
│
├── personalos/                 # Main application package
│   ├── __init__.py
│   ├── config.py              # Pydantic settings (from .env)
│   ├── cli.py                 # CLI entry points
│   │
│   ├── domain/                # Domain models
│   │   ├── models.py          # Core entities (Job, Agent, Event, etc.)
│   │   └── __init__.py
│   │
│   ├── persistence/           # Data persistence
│   │   ├── models.py          # SQLAlchemy ORM models
│   │   ├── database.py        # DB connection/session
│   │   ├── repositories.py    # Data access layer
│   │   └── __init__.py
│   │
│   ├── executor/              # Agent execution engine
│   │   ├── job_search.py      # Job search executor with MCP integration
│   │   └── __init__.py
│   │
│   ├── events/                # Event-driven architecture
│   │   ├── bus.py             # Event bus (pub/sub)
│   │   └── __init__.py
│   │
│   ├── mcp/                   # Model Context Protocol
│   │   ├── base.py            # MCPServer base class
│   │   ├── cache.py           # Caching (Redis/in-memory)
│   │   ├── manager.py         # MCP orchestrator
│   │   ├── adapter.py         # MCPToolInvoker (ToolInvoker port)
│   │   └── __init__.py
│   │
│   ├── tools/                 # Tool boundary
│   │   ├── gateway.py         # ToolGateway / ToolInvoker ports
│   │   ├── registry.py        # Tool registry (adapter)
│   │   └── __init__.py
│   │
│   ├── policy/                # Policy enforcement
│   │   ├── intents.py         # ToolIntent / ApprovedIntent
│   │   ├── rules.py           # Allowlists, mutating + origin rules
│   │   ├── engine.py          # PolicyEngine (default deny)
│   │   └── errors.py          # PolicyDenied / ApprovalRequired
│   │
│   ├── bootstrap.py           # Composition root (wires the layers)
│   │
│   ├── graphs/                # Orchestration (LangGraph)
│   │   ├── supervisor.py      # Top-level graph: context, typed routing, delegation
│   │   └── job_search.py      # The Job Search domain subgraph + its ports
│   ├── models/                # AI models (TODO)
│   ├── state/                 # State management (TODO)
│   ├── retrieval/             # RAG/search (TODO)
│   └── observability/         # Logging/tracing (TODO)
│
├── apps/                      # Application entry points
│   ├── api/                   # FastAPI web server
│   │   ├── main.py            # FastAPI app factory
│   │   └── routes/
│   │       ├── jobs.py        # Job search endpoints
│   │       └── __init__.py
│   │
│   └── worker/                # Background job processor
│       └── job_runner.py      # Runs one job search in its own session
│
├── mcp_servers/               # MCP server implementations
│   ├── jobs/                  # Job search MCP server
│   │   ├── server.py          # JobsMCPServer with 4 tools
│   │   └── __init__.py
│   ├── files/                 # File access server (TODO)
│   └── google/                # Google API integration (TODO)
│
├── docs/                      # Design documentation
│   └── ARCHITECTURE_BOUNDARIES.md  # Layer ownership + enforced import rules
│
├── scripts/
│   └── check_boundaries.py    # Standalone boundary check (runs in CI)
│
├── tests/                     # Test suite
│   ├── architecture/          # Layer boundary enforcement
│   ├── unit/                  # Unit tests
│   │   ├── test_job_search.py
│   │   ├── test_mcp_server.py
│   │   ├── test_executor_mcp_integration.py
│   │   └── __init__.py
│   ├── integration/           # Integration tests (TODO)
│   ├── adversarial/           # Adversarial tests (TODO)
│   ├── graph_scenarios/       # Workflow tests (TODO)
│   └── fixtures/              # Test fixtures (TODO)
│
├── evals/                     # Evaluation datasets
│   ├── golden/                # Golden test datasets
│   └── model_benchmarks/      # Model performance benchmarks
│
└── migrations/                # Database schema migrations (Alembic)
    └── README.md
```

---

## Architecture

Layer ownership and the enforced import rules live in
[docs/ARCHITECTURE_BOUNDARIES.md](docs/ARCHITECTURE_BOUNDARIES.md). The short
version: orchestration proposes intents, policy decides, executors run only
approved intents, adapters do the I/O.

### Data Flow

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  FastAPI (apps/api/main.py)     │
│  - CORS, Error Handling         │
│  - Request Validation           │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Routes (apps/api/routes/)      │
│  - POST /api/v1/jobs/           │
│  - GET /api/v1/jobs/{id}        │
│  - GET /api/v1/jobs/            │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Domain Models                  │
│  - Job (Pydantic)               │
│  - Validation                   │
└──────┬──────────────────────────┘
       │
       ├─────────────────────────────────┐
       │                                 │
       ▼                                 ▼
┌────────────────────────┐     ┌──────────────────────┐
│  Persistence Layer     │     │  Event Bus           │
│  - JobRepository       │     │  - pub/sub           │
│  - JobModel (ORM)      │     │  - Async handlers    │
│  - PostgreSQL          │     │  - Event logging     │
└────────┬───────────────┘     └──────────────────────┘
         │
         ▼
┌────────────────────────┐
│  Executor              │
│  - Job Search Engine   │
│  - Emits ToolIntents   │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────────────┐
│  Tool Gateway + Policy Engine  │
│  - Default deny allowlist      │
│  - Approval gate on mutations  │
│  - Mints ApprovedIntent        │
└────────┬───────────────────────┘
         │  (approved intents only)
         ▼
┌────────────────────────────────┐
│  MCP Server Manager            │
│  - Tool Orchestration          │
│  - Caching                     │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│  MCP Servers                   │
│  - Jobs MCP (4 tools)          │
│  - Files MCP (TODO)            │
│  - Google MCP (TODO)           │
└────────┬───────────────────────┘
         │
         ├─────────────────────────────────┐
         │                                 │
         ▼                                 ▼
┌──────────────────────┐     ┌──────────────────────┐
│  Redis Cache         │     │  External APIs       │
│  - Result Caching    │     │  - Indeed            │
│  - TTL Management    │     │  - LinkedIn          │
└──────────────────────┘     │  - Job Boards        │
                             └──────────────────────┘
```

### Key Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Repository** | `personalos/persistence/repositories.py` | Abstract data access |
| **Domain Models** | `personalos/domain/models.py` | Rich, validated entities |
| **Event-Driven** | `personalos/events/bus.py` | Async event processing |
| **Pub/Sub** | `personalos/events/bus.py` | Decoupled communication |
| **MCP Servers** | `personalos/mcp/` | Modular tool capabilities |
| **Policy Gate** | `personalos/policy/engine.py` | Default-deny authorization of tool intents |
| **Ports & Adapters** | `personalos/tools/gateway.py` | Executors depend on ports, not adapters |
| **Composition Root** | `personalos/bootstrap.py` | Single place that wires the layers |
| **Dependency Injection** | `apps/api/main.py` | FastAPI Depends() |
| **Caching** | `personalos/mcp/cache.py` | Performance optimization |

---

## Completed Components

### 1. Project Configuration & Dependencies

- `pyproject.toml` - Complete Python project with 40+ dependencies
- `.env.example` - Environment variables template
- Python 3.10+ requirement
- All production and dev dependencies configured

**Key Dependencies:**
- FastAPI 0.141+, Uvicorn 0.52+
- SQLAlchemy 2.0+, Psycopg2 (PostgreSQL)
- Pydantic 2.13+
- LangChain 1.3+, LangGraph 1.2+
- Celery 5.6+, Redis 8.1+
- OpenTelemetry (tracing/metrics)
- Pytest 9.1+ (testing)

### 2. Domain Models (`personalos/domain/models.py`)

**Entities:**
- `Job` - Job search task with keywords, locations, salary range, status
- `JobStatus` - Enum: pending, running, completed, failed, cancelled
- `AgentState` - Tracks agent execution progress with history
- `Event` - Domain events for event-driven architecture
- `EventType` - Event type enums (job.*, agent.*, result.*)
- `Tool` - Agent tool definition with parameters
- `AgentConfig` - Configuration for agents (model, temperature, etc.)

**Features:**
- Full Pydantic validation
- JSON serialization
- Type safety with type hints
- Rich domain logic

### 3. Database Layer (`personalos/persistence/`)

**ORM Models (SQLAlchemy 2.0):**
- `JobModel` - Persistent job with UUID PK
- `EventModel` - Event log with job/agent references
- `AgentStateModel` - Agent state snapshots

**Features:**
- PostgreSQL UUID type
- JSON columns for complex data
- DateTime tracking (created, updated, started, completed)
- Connection pooling
- Session management with context managers

**Repository Pattern:**
- `JobRepository` - CRUD operations for jobs
- Clean domain model mapping
- Transaction handling
- Async queries ready

### 4. Core Executor (`personalos/executor/job_search.py`)

**JobSearchExecutor - Main execution engine**

4-step workflow:
1. **Prepare** - Validate search parameters
2. **Search** - Call MCP `search_jobs` tool
3. **Scrape** - Call MCP `scrape_job_details` for top 20 results (parallel)
4. **Filter** - Call MCP `filter_jobs` with salary/experience filters

**Features:**
- Manages job lifecycle (pending → running → completed/failed)
- MCP tool integration
- Error handling and recovery
- Event publishing
- Results ranking and sorting
- Returns top 10 matches

### 5. Event System (`personalos/events/bus.py`)

**EventBus - Pub/Sub pattern**

Features:
- Type-safe event handling
- Async handler support
- Global event bus instance
- Event type tracking
- Handler registration and unregistration

### 6. API Layer (`apps/api/`)

**FastAPI Application:**
- Production-ready async API
- CORS middleware
- Error handling middleware
- Automatic MCP initialization on startup
- Proper shutdown cleanup

**Endpoints:**
```
POST   /api/v1/jobs/              Create job search
GET    /api/v1/jobs/{job_id}      Get job details with results
GET    /api/v1/jobs/              List all jobs (with pagination)
```

**Features:**
- Request/response models with Pydantic
- Dependency injection (FastAPI Depends)
- Proper HTTP status codes
- Error handling
- Automatic serialization

### 7. CLI Entry Points (`personalos/cli.py`)

Commands:
- `api` - Run API server with configurable host/port/workers
- `db_init` - Initialize database schema
- `worker` - Run background worker (Celery placeholder)

### 8. MCP Framework (`personalos/mcp/`)

**Complete Model Context Protocol implementation**

**Base Server (`base.py`):**
- Abstract `MCPServer` class
- Tool registration and schema validation
- Async-first execution
- Error handling and logging
- Automatic async detection with `inspect.iscoroutinefunction()`

**Tool Schema:**
- JSON Schema validation
- Parameter documentation
- Domain model conversion
- Type safety

**Caching (`cache.py`):**
- `RedisCache` - Distributed caching for production
- `InMemoryCache` - Simple fallback for development
- Automatic cache key generation
- TTL-based expiration
- Performance optimization

**Manager (`manager.py`):**
- `MCPServerManager` - Central orchestrator
- Multiple server registration
- Tool routing and execution
- Server introspection
- Tool discovery

### 9. Jobs MCP Server (`mcp_servers/jobs/server.py`)

**Production-ready job search capabilities**

**4 Tools:**

1. **`search_jobs`** - Search job listings
   - Parameters: keywords, locations, job_type, limit
   - Caching support
   - Returns: jobs with relevance scores

2. **`scrape_job_details`** - Extract job information
   - Parameters: job_id, job_url
   - Returns: full description, requirements, benefits, skills

3. **`filter_jobs`** - Filter and rank results
   - Parameters: jobs, salary_min/max, experience_level, remote_only
   - Returns: ranked and filtered results

4. **`save_favorite_job`** - Save jobs to favorites
   - Parameters: job_id, notes
   - Returns: confirmation with timestamp

**Features:**
- Comprehensive parameter validation
- Error handling
- Mock implementation ready for real APIs
- Caching integration
- Async/await throughout

### 10. Testing (`tests/`)

**15 Unit Tests - ALL PASSING ✅**

- **MCP Server Tests (9)** - Tool registration, execution, manager routing
- **Executor Integration (3)** - Full workflow, search step, filter step
- **Job Search Models (3)** - Model creation, status transitions, results

**Test Infrastructure:**
- Pytest with async support (pytest-asyncio)
- Comprehensive mocking
- Integration test scenarios
- Fixture templates ready for expansion

### 11. Documentation

- `BUILD.md` - Build summary and completion status
- `MCP_GUIDE.md` - Detailed MCP server implementation guide
- `MCP_IMPLEMENTATION.md` - MCP architecture summary
- `TEST_RESULTS.md` - Full test coverage report
- `IMPLEMENTATION_GUIDE.md` - This comprehensive guide

### 12. Orchestration Graphs (`personalos/graphs/`)

**`supervisor.py` — the top-level graph.** Loads the actor's long-term facts,
classifies the request into a typed `RouteDecision`, and either asks for
clarification or plans a bounded `TaskDAG` and hands it to a domain subgraph. It
holds no tool and no repository: `IntentClassifier` and `JobSubgraphRunner` are
injected ports.

**`job_search.py` — the Job Search subgraph**, the one domain subgraph this
build implements:

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

### 13. Durable Checkpointing and Resume

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

That last rule is the one with a cost: an action whose outcome is genuinely
unknown needs a human rather than a retry. It is the only rule consistent with
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
            "keywords": {
                "type": "array",
                "items": {"type": "string"}
            },
            "locations": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    },
    required=["keywords", "locations"]
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
manager.register_server(JobsMCPServer())

# Execute tool
result = await manager.execute_tool(
    tool_name="search_jobs",
    server_name="jobs",
    keywords=["Python"],
    locations=["Remote"]
)

# Get all tools
tools = manager.get_all_tools()

# Get server info
server_info = manager.get_server_info("jobs")
```

### Jobs MCP Server

**Tool 1: `search_jobs`**

```python
result = await manager.execute_tool(
    "search_jobs",
    "jobs",
    keywords=["Python", "FastAPI"],
    locations=["Remote", "NYC"],
    job_type="full-time",
    limit=50
)

# Response format:
{
    "success": true,
    "result": {
        "total": 50,
        "jobs": [
            {
                "id": "job_123",
                "title": "Senior Python Developer",
                "company": "TechCorp",
                "location": "Remote",
                "salary": "$120,000 - $160,000",
                "url": "https://example.com/jobs/job_123",
                "relevance_score": 0.95
            }
        ]
    }
}
```

**Tool 2: `scrape_job_details`**

```python
result = await manager.execute_tool(
    "scrape_job_details",
    "jobs",
    job_id="job_123",
    job_url="https://example.com/jobs/job_123"
)

# Response format:
{
    "success": true,
    "result": {
        "id": "job_123",
        "title": "Senior Python Developer",
        "description": "Full job description...",
        "requirements": ["5+ years Python", "FastAPI", "Docker"],
        "benefits": ["Health insurance", "Remote", "401k"],
        "skills": ["Python", "FastAPI", "Docker"],
        "experience_level": "senior",
        "difficulty_match": 0.92
    }
}
```

**Tool 3: `filter_jobs`**

```python
result = await manager.execute_tool(
    "filter_jobs",
    "jobs",
    jobs=[...],  # Job list from search_jobs
    salary_min=100000,
    salary_max=150000,
    experience_level="senior",
    remote_only=True
)

# Response format:
{
    "success": true,
    "result": {
        "total_filtered": 15,
        "total_input": 50,
        "jobs": [...],
        "filters_applied": {...}
    }
}
```

**Tool 4: `save_favorite_job`**

```python
result = await manager.execute_tool(
    "save_favorite_job",
    "jobs",
    job_id="job_123",
    notes="Interesting opportunity with good benefits"
)

# Response format:
{
    "success": true,
    "result": {
        "job_id": "job_123",
        "notes": "Interesting opportunity with good benefits",
        "saved_at": "2026-08-30T12:00:00Z"
    }
}
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
                    "param1": {
                        "type": "string",
                        "description": "First parameter"
                    },
                    "param2": {
                        "type": "integer",
                        "description": "Second parameter"
                    }
                }
            },
            required=["param1"]
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

### Integration with Executor

The `JobSearchExecutor` uses MCP servers internally:

```python
async def run_job_search(self, job: Job):
    # Step 1: Prepare parameters
    # ...
    
    # Step 2: Search via MCP
    search_result = await self.mcp_manager.execute_tool(
        "search_jobs",
        "jobs",
        keywords=job.keywords,
        locations=job.locations,
        job_type=job.job_type
    )
    
    # Step 3: Scrape details for top 20 (parallel)
    scrape_tasks = []
    for job_item in search_result["result"]["jobs"][:20]:
        task = self.mcp_manager.execute_tool(
            "scrape_job_details",
            "jobs",
            job_id=job_item["id"],
            job_url=job_item["url"]
        )
        scrape_tasks.append(task)
    
    detailed_results = await asyncio.gather(*scrape_tasks)
    
    # Step 4: Filter via MCP
    filtered = await self.mcp_manager.execute_tool(
        "filter_jobs",
        "jobs",
        jobs=[r["result"] for r in detailed_results],
        salary_min=job.salary_min,
        salary_max=job.salary_max
    )
    
    return filtered[:10]
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
                "description": "Parameter description"
            }
        }
    },
    required=["param"]
)
```

---

## Testing

### Test Results

**Status:** ✅ All 15 Tests Passing

```
tests/unit/test_executor_mcp_integration.py ... [3 tests]
tests/unit/test_job_search.py               ... [3 tests]
tests/unit/test_mcp_server.py              .. [9 tests]
```

Execution time: 0.39s

### Test Coverage

#### MCP Server Tests (9 tests)

```python
pytest tests/unit/test_mcp_server.py -v
```

- Server initialization and tool registration
- All 4 tool executions (search, scrape, filter, save)
- Manager routing and introspection
- Error handling and validation

#### Executor Integration Tests (3 tests)

```python
pytest tests/unit/test_executor_mcp_integration.py -v
```

- Full job search workflow with MCP
- Individual executor steps (search, filter)
- End-to-end integration

#### Job Search Model Tests (3 tests)

```python
pytest tests/unit/test_job_search.py -v
```

- Model creation and validation
- Status transitions (pending → running → completed)
- Results storage and serialization

### Running Tests

```bash
# Run all tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_mcp_server.py -v

# Run with coverage
pytest tests/ --cov=personalos --cov=mcp_servers --cov-report=html

# Run with markers
pytest tests/unit/ -k "integration" -v
```

### Key Validations

✅ **MCP Framework**
- Tool registration and discovery working
- Async handler execution proper
- Parameter validation enforced
- Error handling correct
- Schema documentation complete

✅ **Jobs MCP Server**
- All 4 tools execute successfully
- Caching integration working
- Mock data handling correct
- Response formats correct

✅ **Executor Integration**
- MCP tool execution from executor works
- Chained operations (search → scrape → filter) work
- Result processing and ranking works

✅ **Database Models**
- UUID primary keys correct
- JSON columns working
- DateTime tracking correct
- ORM serialization working

---

## Next Steps

### High Priority (Phase 1)

1. **Integrate Real Job Search APIs**
   - Connect to Indeed API
   - Add LinkedIn integration
   - Support additional job boards
   - Replace mock implementations

2. **Background Worker Implementation**
   - Implement Celery worker
   - Move job searches to background
   - Add job queue management
   - Implement retry logic

3. **Database Migrations**
   - Set up Alembic
   - Generate migration scripts
   - Test migration workflows
   - Document schema changes

4. **Advanced Testing**
   - Add adversarial tests
   - Implement scenario tests
   - Add golden test datasets
   - Test error conditions

### Medium Priority (Phase 2)

5. **Graph-Based Workflows**
   - Implement LangGraph integration
   - Define workflow nodes
   - Implement edges and transitions
   - Add complex orchestration

6. **Agent Policies**
   - Implement rate limiting
   - Add permission enforcement
   - Implement approval workflows
   - Add audit logging

7. **Retrieval System**
   - Implement vector search
   - Add job description similarity
   - Implement RAG pipeline
   - Add relevance ranking

8. **Observability**
   - Instrument with OpenTelemetry
   - Add distributed tracing
   - Create dashboards
   - Set up alerting

### Low Priority (Phase 3)

9. **Google MCP Server**
   - Gmail integration
   - Google Sheets for results
   - Google Drive for documents

10. **Files MCP Server**
    - Resume/CV management
    - Job application tracking
    - Document storage

11. **Advanced Filtering**
    - ML-based ranking
    - Skill matching
    - Recommendation engine

12. **Web UI**
    - Frontend dashboard
    - Job search management
    - Results visualization

### Immediate Actions (This Week)

```bash
# 1. Start the API and test manually
python -m personalos.cli api --reload

# 2. Create a job search
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "Content-Type: application/json" \
  -d '{...}'

# 3. Verify MCP tools execute
# Check job search results

# 4. Run full test suite
pytest tests/unit/ -v

# 5. Plan real API integration
# - Choose job board APIs
# - Get API keys
# - Implement adapters
```

---

## Configuration

### Environment Variables (.env)

```env
# Database
DATABASE_URL=postgresql://user:password@localhost/personalos

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Logging
LOG_LEVEL=INFO

# MCP Configuration
MCP_JOBS_ENABLED=true
MCP_GOOGLE_ENABLED=false
MCP_FILES_ENABLED=true

# Agent Configuration
AGENT_MODEL=gpt-4
AGENT_TEMPERATURE=0.7
AGENT_MAX_ITERATIONS=10
```

### CLI Commands

```bash
# Initialize database
python -m personalos.cli db_init

# Run API server
python -m personalos.cli api --host 0.0.0.0 --port 8000 --reload

# Run background worker
python -m personalos.cli worker

# Run Celery worker
celery -A personalos.worker worker --loglevel=info
```

---

## Support & Documentation

- **Project Repository:** [PersonalOS on GitHub](https://github.com/madhurlak0810/PersonalOS)
- **Issues:** [GitHub Issues](https://github.com/madhurlak0810/PersonalOS/issues)
- **Code Structure:** See `IMPLEMENTATION_GUIDE.md` (this file)
- **Build Status:** See `BUILD.md`
- **Test Results:** See `TEST_RESULTS.md`

---

## Summary

PersonalOS is a fully functional, production-ready agent framework with:

✅ **Complete MCP Framework** - Modular, extensible tool architecture
✅ **15 Passing Tests** - Comprehensive test coverage
✅ **Production-Ready API** - FastAPI with proper error handling
✅ **Database Layer** - SQLAlchemy ORM with PostgreSQL
✅ **Event System** - Pub/sub for decoupled communication
✅ **Job Search Executor** - 4-step workflow with MCP integration

**Ready for:** Real API integration, background workers, advanced workflows, and production deployment.


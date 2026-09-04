# Architecture Boundaries

This document defines who owns what in PersonalOS, and which layer may depend on
which. It is prose; the enforced version lives in
[tests/architecture/boundaries.py](../tests/architecture/boundaries.py) and is
checked by [tests/architecture/test_boundaries.py](../tests/architecture/test_boundaries.py)
on every CI run. **If you change a boundary, change both, in the same commit,
with a reason.**

## The one rule everything else serves

> Orchestration proposes. Policy decides. Executors run what policy approved.
> Adapters do the I/O. Persistence remembers. Nothing skips a step.

The failure mode this prevents is concrete: a model emits something that looks
like a tool call, and code somewhere splats it into an adapter. Then the
allowlist, the approval gate, and the idempotency guard were all advisory. So
the boundary is not a convention — it is a type. An executor cannot call a tool,
because it has no adapter to call; the only thing it holds is a
[`ToolGateway`](../personalos/tools/gateway.py), and the only thing a gateway
accepts is a `ToolIntent`, which it will not execute until the policy engine
turns it into an `ApprovedIntent`.

## Layers

Each layer lists what it owns, and what it is explicitly *not* allowed to do.
"May import" is the complete list; importing within a layer is always fine.

### `domain` — `personalos/domain/`

Entities and invariants: `Job`, `AgentState`, `Event`, `MutatingIntent`,
idempotency-key validation.

- **May import:** nothing internal.
- **Must not:** know about databases, transports, models, or policy. If a domain
  model needs a session, it is not a domain model.

### `policy` — `personalos/policy/`

The only component that answers *may this run?* Owns `ToolIntent` (a proposal),
`PolicyRule`, `PolicyEngine`, and `ApprovedIntent` (a cleared proposal).

- **May import:** `domain`.
- **Must not:** perform I/O, read storage, call tools, or import an adapter.
  Rules are pure functions of an intent, which is what makes them cheap to test
  exhaustively.

Two properties are load-bearing:

1. **Default deny.** `PolicyEngine` denies anything no rule allowed. A newly
   added MCP tool is unreachable until it is listed in
   [`rules.py`](../personalos/policy/rules.py).
2. **Approvals are unforgeable.** `ApprovedIntent.__init__` raises
   `PolicyViolation` unless it is handed a module-private mint token, so
   `ApprovedIntent(...)` cannot be constructed anywhere else in the codebase —
   not in a graph, not in an executor, not in a test helper. The only way to get
   one is `PolicyEngine.authorize()`.

Resolution order inside the engine is deny → require-approval → allow → default
deny, so adding a rule can only ever tighten behaviour.

### `tools` — `personalos/tools/`

The tool boundary. `gateway.py` defines the two ports:

- `ToolGateway` — what executors depend on. `dispatch(intent)` authorizes, then
  executes.
- `ToolInvoker` — what adapters implement. Takes an `ApprovedIntent` and nothing
  else, so an adapter cannot be driven from raw arguments even by accident.

`registry.py` is an adapter for in-process tools; it takes loose keyword
arguments and therefore must only be reached through `ToolRegistryInvoker`.

- **May import:** `domain`, `policy`.
- **Must not:** import `mcp`, `executor`, `graphs`, or `persistence`. The
  gateway is a port, not a hub.

Denials raise (`PolicyDenied`, `ApprovalRequired`) rather than returning a
failed `ToolResult`: a blocked tool call is a control-flow event the caller must
handle, not an ordinary error it might skip past while continuing to act.

### `executor` — `personalos/executor/`

Owns *how* a task runs: step sequence, agent state, result persistence, failure
handling. Every step is expressed as a `ToolIntent` and submitted to the
gateway.

- **May import:** `domain`, `policy`, `persistence`, `tools`, `events`, `state`,
  `config`.
- **Must not:** import `personalos.mcp`, `mcp_servers`, or
  `personalos.tools.registry`. Must not construct its own gateway or reach for
  a global tool manager — the gateway is a required constructor argument.

Intents built by executor code carry `origin=SYSTEM`, meaning their *shape* is
fixed by reviewed code. Anything a model proposes must be built with
`origin=LLM` so `UntrustedOriginRule` can escalate it. Passing raw model output
into `arguments` without going through an intent is the bug this layering
exists to make impossible to write accidentally.

### `graphs` — `personalos/graphs/`

Orchestration: which step happens next, branching, retries, human-in-the-loop
pauses.

- **May import:** `domain`, `policy`, `executor`, `events`, `state`, `models`,
  `config`.
- **Must not:** import `tools`, `mcp`, `mcp_servers`, or `persistence`. A graph
  that can call a tool directly is a graph that can bypass policy; a graph that
  can write to the database is a graph whose state transitions are untraceable.
  Delegate both to an executor.

### `persistence` — `personalos/persistence/`

Storage and retrieval: ORM models, sessions, repositories, and the idempotency
guard that makes mutating operations at-most-once.

- **May import:** `domain`, `config`.
- **Must not:** make decisions, call tools, or import `executor` / `graphs` /
  `policy`. Repositories translate between domain models and rows; that is all.

### `mcp` — `personalos/mcp/`

Adapter layer for the Model Context Protocol: `MCPServer` base class, the server
manager, caching, and `MCPToolInvoker`, which satisfies the `ToolInvoker` port.

- **May import:** `domain`, `policy`, `tools`, `persistence`, `config`.
- **Must not:** import `mcp_servers`. This layer knows *how* to talk to a
  server, never *which* servers exist — that inversion is what previously made
  `manager.py` import a concrete server, and it is why registration moved to the
  composition root.

The `persistence` dependency is deliberate and narrow: `MCPServer` uses
`IdempotencyGuard` so a mutating tool cannot execute without a dedup record.

### `mcp_servers` — `mcp_servers/`

Concrete tool implementations, one package per provider.

- **May import:** `domain`, `mcp`, `persistence`, `config`.
- **Must not:** import `executor`, `graphs`, `policy`, or `tools`. A server
  implements tools; it does not decide who may call them.

### Support layers

| Layer | Owns | May import |
| --- | --- | --- |
| `config` | settings from the environment | nothing internal |
| `events` | domain events between layers | `domain` |
| `state` | in-flight agent state | `domain`, `persistence` |
| `observability` | logging, metrics, tracing | `domain`, `config` |
| `retrieval` | retrieval and ranking | `domain`, `persistence`, `config` |
| `models` | model clients, prompt plumbing | `domain`, `config` |

### `composition` — `personalos/bootstrap.py`, `personalos/cli.py`, `apps/`

Something has to join layers that cannot import each other. That is this
layer's only job.

- **May import:** everything.
- **Must not:** contain logic that belongs to a layer. If a function here does
  more than construct and connect, it is in the wrong file.

[`bootstrap.py`](../personalos/bootstrap.py) is where MCP servers are
registered, the policy engine is built, the gateway is assembled, and executors
are constructed. `build_tool_gateway()` defaults to the default-deny engine, so
a caller who forgets to pass one gets the restrictive engine rather than an open
door.

The two entry points that use it:

- [`apps/api/routes/jobs.py`](../apps/api/routes/jobs.py) accepts and reads job
  searches. It does not run them, and holds no executor.
- [`apps/worker/job_runner.py`](../apps/worker/job_runner.py) runs them, in a
  session it opens itself. A request-scoped session is closed once the response
  is sent, so background work cannot borrow one.

## Dependency direction

```
                 apps/  ·  personalos/bootstrap.py  ·  personalos/cli.py
                                 (composition root)
                                        │
              ┌─────────────────────────┼──────────────────────────┐
              ▼                         ▼                          ▼
          graphs ───────────────────► executor                mcp_servers
        (orchestration)            (runs approved                  │
              │                       intents)                    ▼
              │                    │        │                     mcp
              │                    │        │              (adapter: MCPToolInvoker)
              ▼                    ▼        ▼                     │
           policy ◄──────────── tools ──────┴─────────────────────┘
      (decides; pure)      (gateway = ports)         persistence
              │                    │                (storage, idempotency)
              └────────────────────┴──────────┬──────────────┘
                                              ▼
                                           domain
                                    (entities, invariants)
```

Arrows point in the direction imports are allowed. There are no cycles, and
nothing points *up*.

## Request path

A job search, end to end:

1. `apps/api/routes/jobs.py` persists the `Job` and calls
   `build_job_search_executor(repo)`.
2. `bootstrap` builds `PolicyEnforcingToolGateway(default_policy_engine(),
   MCPToolInvoker(manager))` and hands it to `JobSearchExecutor`.
3. The executor builds a `ToolIntent` per step — `jobs.search_jobs`,
   `jobs.scrape_job_details`, `jobs.filter_jobs` — each with provenance
   (`requested_by`, `job_id`, `agent_id`) and `origin=SYSTEM`.
4. `gateway.dispatch(intent)` calls `PolicyEngine.authorize()`. Rules run:
   provenance present, tool allowlisted, no unexpected arguments, mutating tools
   gated, untrusted origins escalated.
5. On allow, the engine mints an `ApprovedIntent`. On deny it raises
   `PolicyDenied`; if a human grant is needed it raises `ApprovalRequired`.
6. `MCPToolInvoker.invoke(approved)` — re-checking the type, belt and braces —
   unpacks the approved arguments onto `MCPServerManager.execute_tool`.
7. For mutating tools, `MCPServer` runs the handler behind `IdempotencyGuard`,
   which claims the `idempotency_key` before the side effect and records the
   outcome after.
8. The executor persists results through `JobRepository`.

Policy is crossed exactly once per tool call, in exactly one place.

## Adding things

**A new tool.** Implement it on an MCP server, then add its `server.tool` ref
and argument surface to `JOB_SEARCH_TOOL_ARGUMENTS` (or a new allowlist) in
[`policy/rules.py`](../personalos/policy/rules.py). Until you do, the engine
denies it — the boundary tests will not tell you, but the first call will.

**A mutating tool.** Mark the `ToolSchema` `mutating=True` (which adds
`idempotency_key` to its advertised contract), allowlist it, and decide whether
it belongs in `MutatingToolRule(auto_approved=...)`. Default is: it waits for a
human.

**A new layer.** Add a `Layer` entry to
[`boundaries.py`](../tests/architecture/boundaries.py) with its `allows` list
and a one-line `responsibility`, and describe it here. `test_every_module_belongs_to_a_layer`
fails for any module that is not claimed by a layer, so a new package cannot be
added silently.

**A dependency that the check rejects.** Three honest options, in order of
preference: invert it with a port (define the interface in the lower layer, wire
the implementation in `bootstrap`); move the code to the layer that owns the
concern; or change the declared boundary here and in the spec, saying why. Do
not add an exception for one import.

## What the tests actually assert

From [test_boundaries.py](../tests/architecture/test_boundaries.py):

- Every internal import in `personalos/`, `apps/`, and `mcp_servers/` respects
  the layer graph, checked by AST parse — so imports nested inside functions are
  caught too.
- Every module belongs to a declared layer.
- The checker itself fails on planted violations (executor → `mcp`, graph →
  `tools`, policy → `persistence`, domain → `persistence`, and others), and does
  *not* fire on the intended wiring. A boundary test that cannot detect a
  violation is worse than no test, because it reads as assurance.
- Named guarantees: the executor never imports a tool adapter; orchestration
  never imports tools or storage; policy depends only on `domain`; `domain`
  depends on nothing internal.
- Nothing outside `personalos/policy/` references the approval-minting
  internals.
- This document exists, describes every declared layer, and points at the spec.

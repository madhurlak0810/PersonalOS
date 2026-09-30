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

A graph therefore reaches the outside world in exactly one way: it declares a
port — a `Protocol` in the graph module — and the composition root binds an
adapter to it. [`graphs/job_search.py`](../personalos/graphs/job_search.py) is
the worked example. Nine of its ports are required constructor arguments — a
profile store, job board providers, a scorer, an evidence checker, a packet
builder, an approval gate, an action executor, an application store and an
event emitter — for the same reason `JobSearchExecutor` requires a
`ToolGateway`: a graph that can fall back to a global default is a graph whose
reach is not visible at its construction site. Four are genuinely optional:
the posting normalizer (which has a pure, dependency-free default), the
recruiter inbox plus its classifier, which are what turn the
recruiter-response branch on and must be supplied together, and the pending
checkpoint scheduler, which turns durable follow-up waits on. That last one is
optional rather than defaulted because scheduling a wait no process will ever
sweep is worse than scheduling none — the row reads, in SQL, as a follow-up
that is coming.

Three nodes in that graph are load-bearing for this boundary. Every node that
wants to act outwardly returns an `ActionIntent` and has no executor and no edge
to one; the only route from an intent to the outside world is:

    request_approval_for_external_write
      -> approval_checkpoint_for_external_submission   (LangGraph `interrupt()`)
      -> execute_approved_actions

`request_approval` mints an `ApprovalRequest` — the action's hash, target,
human-readable summary, risk level, requested scopes and expiry — and returns,
so all of it is checkpointed. `approval_checkpoint` is the only node that calls
`interrupt()`; the run parks there and the answer may arrive days later, from a
different process. `execute_approved_actions` is the only node holding an
`ActionExecutor`, and it runs a super-step *later*, so it re-reads the pending
action from the checkpoint and passes it through
`personalos.domain.job_search.authorize_execution` before acting.

The split is why it is three nodes and not one. LangGraph discards the state
update of a node that interrupts, so a node that minted the request and then
interrupted would lose it — and that recorded hash is the only fixed point a
resume can check a mutated action against. And the recompute has to happen on
the far side of the pause: while a run is parked, `pending_actions` is just
state, so an action can be rewritten between the request and the resume. A
rewritten action fails closed, with a recorded `ApprovalRefusal` and no call.

The `ApprovalGate` port that remains is not a second approver. It answers only
"is there already a decision on file for this?" — a standing grant, an answer
recorded through the API before the graph got here. `PENDING` is its honest
answer to "nobody has decided", and that is what triggers the interrupt; a
deployment with no such source wires `InterruptOnlyApprovalGate` and pauses on
every outward-facing write. Whatever it returns is still bound to the
checkpointed request and re-checked against a freshly recomputed hash.

The graph has a second entry point, and it enters *into* that same triple. A
run invoked with `fired_checkpoints` in its input is a durable wait coming due;
`route_from_start` sends it to `draft_follow_up_for_triggered_checkpoint`
instead of to discovery, and that node proposes an `ActionIntent` like every
other node that wants to act. It holds no executor, so a follow-up drafted at
3am by a sweep with no human anywhere near it still parks at the interrupt.

### The intent-only boundary

The layer graph above polices imports between *our* packages. It cannot see a
graph node doing `from googleapiclient.discovery import build`, opening a
`sqlalchemy` session, or calling `Path.write_text` — none of those is an
internal import. So there is a second, complementary check,
[tests/architecture/intent_boundary.py](../tests/architecture/intent_boundary.py),
enforced by [test_intent_boundary.py](../tests/architecture/test_intent_boundary.py)
and by `scripts/check_boundaries.py` in CI.

> The only thing an LLM-backed node may produce is a typed intent. Only the
> executor performs the side effect:
> `ActionIntent → PolicyEngine / approval → executor`.

It applies to the **reasoning layers** — `graphs`, `models`, `domain`,
`policy`, `events` — and refuses, by AST:

| Category | Examples | Why |
| --- | --- | --- |
| `provider-sdk` | `googleapiclient`, `google.oauth2`, `gcsa`, `caldav`, `msal`, `jobspy`, `linkedin_api`, `mcp` | a provider client performs writes policy never saw |
| `database` | `sqlalchemy`, `psycopg2`, `sqlite3`, `redis`, `celery`, `langgraph.checkpoint.postgres` | a raw DB/broker write is a transition nothing journals |
| `network` | `httpx`, `requests`, `aiohttp`, `urllib.request`, `smtplib` | a bare HTTP or mail client is a hand-rolled provider SDK |
| `process` | `subprocess`, `os.system`, `os.exec*` | anything can happen, none of it as an intent |
| `filesystem` | `shutil`, `tempfile`, `open(p, "w")`, `Path.write_text`/`mkdir`/`unlink`, `os.remove`/`makedirs`/`replace` | mutation belongs to an executor |
| `llm-sdk` | `anthropic`, `langchain_anthropic`, `openai` | model clients live in `models` behind a port (exempt there) |
| `dynamic-import` | `importlib.import_module`, `__import__` | cannot be checked, so refused |

Imports count wherever they appear, including inside a function body; `from
os import remove as rm; rm(p)` resolves through the alias. Reading a file
(`open(p)`, `open(p, "rb")`, `Path.read_text`) is not a side effect and is not
flagged; `open(p, mode)` with a non-literal mode is, because the check cannot
show it is read-only.

The check is also **transitive**. A reasoning module may not reach an effect
layer (`persistence`, `tools`, `mcp`, `mcp_servers`, composition), or a
non-reasoning module that breaks the rules above, through any chain of
internal imports — except through `executor`, which is the sanctioned door
because everything it does is dispatched through a `ToolGateway`. That closes
routes the layer graph allows one hop at a time: `graphs → state` and `state →
persistence` are each legal, but a graph reaching a DB session that way is not.

`langgraph.checkpoint.base`, `langgraph.checkpoint.memory` and `interrupt()`
are fine: a checkpointer is a port the composition root binds, and the
durable one lives in `persistence`.

#### Required node shape

Every node in a reasoning layer has the same five steps, and side effects
appear in none of them:

    NodeInput -> validate -> reason/compute -> emit typed decision -> state update

1. **NodeInput** — read the narrow slice of graph state the node needs.
2. **validate** — rebuild the typed domain values from that slice (state is
   JSON, see `graphs/job_search.py`) and fail or short-circuit on bad input.
3. **reason/compute** — pure computation, or calls to *injected ports*
   (`Protocol`s declared in the graph module, bound by the composition root).
   A model call goes through a `models` port such as `IntentClassifier`, never
   a raw SDK.
4. **emit typed decision** — the result is a typed value: a `RouteDecision`,
   a ranked shortlist, or, for anything outward-facing, an `ActionIntent` with
   target, summary, payload and idempotency key.
5. **state update** — return a partial state dict. An intent goes into
   `pending_actions`; the node has no executor and no edge to one.

Side effects are isolated to executor nodes. In the job-search graph that is
`execute_approved_actions`, the one node holding an `ActionExecutor` port, and
it runs only after `request_approval` → `approval_checkpoint`. It lives in
`graphs` but satisfies the rule, because the port is all it holds; the adapter
that does the I/O is bound in `bootstrap` and lives in an effect layer.

To add a node that needs something new from the outside world, declare a port
for it, and if it *writes*, make it propose an `ActionIntent` instead. Do not
add an exemption to `SIDE_EFFECT_RULES` for one node.

### `persistence` — `personalos/persistence/`

Storage and retrieval: ORM models, sessions, repositories, the idempotency guard
that makes mutating operations at-most-once, and the three pieces that make a
workflow survive its own process —
[`checkpointer.py`](../personalos/persistence/checkpointer.py),
[`leases.py`](../personalos/persistence/leases.py) and
[`action_journal.py`](../personalos/persistence/action_journal.py).

- **May import:** `domain`, `config`.
- **Must not:** make decisions, call tools, or import `executor` / `graphs` /
  `policy`. Repositories translate between domain models and rows; that is all.

Durability lives here rather than in `graphs` because `graphs` may not import
`persistence` at all. That constraint shapes the design rather than bending it:

- **`SqlAlchemyCheckpointSaver`** implements LangGraph's own
  `BaseCheckpointSaver` port, which the graphs already compile against. A graph
  is handed a durable saver by the composition root and does not change a line.
- **`WorkflowLeaseStore`** takes an exclusive, expiring lease on a
  `workflow_id`, so two workers cannot resume the same business process.
- **`PendingCheckpointStore`** owns `pending_checkpoints`, the table behind a
  durable conditional wait. The graph reaches it through a
  `PendingCheckpointScheduler` port, same inversion as everywhere else.
- **`JournaledActionExecutor`** *wraps* the graph's `ActionExecutor` port rather
  than reaching inside the approval node, declaring the shape it wraps as a
  local `Protocol`. Same inversion as `PolicyEnforcingToolGateway` one level up:
  the adapter the composition root binds is what adds the guarantee, and the
  caller's contract does not change. The graph's own invariant is untouched:
  `execute_approved_actions` still decides *whether* to act, and the journal
  only decides whether the act has already happened.

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
- [`apps/worker/workflow_runner.py`](../apps/worker/workflow_runner.py) starts
  and resumes durable workflows. It holds a compiled graph rather than building
  one — which ports a graph is wired to is decided where the graph is
  constructed — plus the thread registry and the lease store, and it takes the
  workflow's lease around every invocation.
- [`apps/worker/checkpoint_monitor.py`](../apps/worker/checkpoint_monitor.py)
  sweeps pending checkpoints. It is composition for the same reason: it holds
  the store, a condition evaluator, the thread registry and a runner, and joins
  them. The decision it applies is not its own — see **Pending checkpoints**
  below.

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

## Durable workflows

Orchestration that can be interrupted needs two identifiers, and keeping them
distinct is the whole design.
[`domain/workflow.py`](../personalos/domain/workflow.py) owns both:

| Identifier | Names | Resumed by |
| --- | --- | --- |
| `thread_id` | one conversational/orchestration thread | LangGraph's checkpointer, which stores and loads state under it |
| `workflow_id` | one long-running business process, possibly several threads | an operator; it is also what a lease is taken against |

A `thread_id` is **derived**, not minted (`derive_thread_id(namespace, *parts)`).
A thread id invented per invocation produces a graph that checkpoints diligently
and can never resume, because nothing ever looks the state up again — which is
what `JobSearchSubgraphRunner` used to do, and why it now takes its thread id
from the composition root instead.

Each kind of thread has one derivation helper — `job_search_thread_id`,
`supervisor_thread_id` — and every caller goes through it. Two places deriving
"the obvious way" is not a style question: whatever registers a thread and
whatever runs on it must produce the byte-identical string, and a disagreement
does not fail anywhere. The run simply starts from scratch every time.

A thread must be **registered** (`WorkflowThreadRegistry.register`) before it can
be checkpointed. Registration is get-or-create on `workflows` and
`workflow_runs`, so a restarted worker recomputing the same derived id rejoins
its existing run; an unregistered thread is refused rather than checkpointed
under an invented `workflow_id`, which is a checkpoint an operator resuming that
workflow would never find.

### Where state is persisted relative to a side effect

The guarantee is: state is durable before any external side effect and again
after that effect's receipt, so a crash between the two cannot produce a
duplicate on resume. Two mechanisms, at two granularities:

1. **Super-step boundaries.** LangGraph writes the checkpoint for a step before
   running that step's tasks, and records each finished task's writes in
   `checkpoint_writes`. A worker killed inside a node resumes *at that node*,
   with everything before it restored.
2. **The action journal, around the effect itself.** A checkpoint boundary is
   coarser than one tool call, so `JournaledActionExecutor` commits a claim on
   the action's `idempotency_key` immediately before the call and the receipt
   immediately after. A resumed run that finds the receipt replays it; one that
   finds a claim with no receipt returns a not-ok receipt and does **not** call
   out again, because a duplicate application cannot be withdrawn while a missed
   one can be resubmitted deliberately.

`SqlAlchemyCheckpointSaver`'s async methods run inline rather than on a worker
thread, and that is part of the guarantee rather than an oversight — see the
class docstring. LangGraph submits checkpoint writes as chained background
tasks; an `await asyncio.to_thread(...)` inside one yields before the row is
written, so the chain falls behind the run it is recording and a kill loses
everything still queued. That was measured, not theorized.

### Resuming

`DurableWorkflowRunner.resume(workflow_id=...)` or `(thread_id=...)`:

1. Resolve the thread (by workflow, or named directly). A workflow with several
   threads is `AmbiguousWorkflowResume` rather than a guess about which half of
   the process to advance.
2. Take the workflow's lease, or raise `WorkflowLeaseUnavailable`.
3. Invoke with `None`, not the initial state. With no input LangGraph continues
   the stored run; passing the inputs again is what *starting* looks like.
4. Release the lease, including on failure.

The lease's mutual exclusion rests on a unique constraint on
`workflow_leases.workflow_id` and a token-guarded single-statement `UPDATE`, not
on `SELECT ... FOR UPDATE` — which is taken on Postgres, but is a silent no-op on
SQLite, so a guarantee that depended on it would hold in production and nowhere
else. Leases expire so a hard-killed worker does not strand its workflow, and
every takeover mints a new fencing token so a stalled holder cannot release or
renew the lease that replaced it.

## Pending checkpoints

A durable conditional wait — *"seven days after applying, if no recruiter
response exists, draft a follow-up"* — is spread across three layers, and the
split is the design:

| Layer | Piece | Owns |
| --- | --- | --- |
| `domain` | [`checkpoints.py`](../personalos/domain/checkpoints.py) | the shape of a wait, and `decide_checkpoint` — the whole policy as a pure function |
| `persistence` | [`pending_checkpoints.py`](../personalos/persistence/pending_checkpoints.py) | the row, idempotent scheduling, the `due` query, the guarded close |
| `apps` | [`checkpoint_monitor.py`](../apps/worker/checkpoint_monitor.py) | the sweep: ask the condition, apply the outcome, start the thread |

Three properties are load-bearing, and each is a constraint on how this may be
extended:

1. **A wait is a row, not a timer.** Nothing holds it open: no sleeping task,
   no parked connection, no scheduled future. That is what lets it survive a
   deploy, a crash and a weekend, and it is why the wait stores a `thread_id`
   (a *lookup key*, resolved at trigger time) rather than any kind of handle.
2. **The condition is stored as a question and asked at trigger time.**
   `CheckpointCondition` is a value, not a closure — a closure cannot be
   written to a row, and, worse, it would capture the world as it looked when
   the wait started, which is exactly the world the wait exists to let change.
   `CheckpointConditionEvaluator` is the port that answers it, and an
   implementation that cannot tell must raise rather than guess: answering
   `True` silently cancels a follow-up nobody decided to cancel.
3. **`expires_at` is separate from `trigger_at`, and beats it.** A monitor that
   was down for a week must not then send a week-late follow-up, and a
   checkpoint nobody ever swept must not sit `pending` for ever. Both are the
   same rule, and it is the one case where the ordering inside
   `decide_checkpoint` is the whole point: met condition → resolve, expired →
   expire, due → fire.

The sweep claims before it acts (`close(..., FIRED)` is a guarded `UPDATE`
exactly one of two racing monitors wins) and starts the graph path only if it
won. It defers a thread already parked on an unanswered approval rather than
stacking a second request on it.

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

From [test_intent_boundary.py](../tests/architecture/test_intent_boundary.py):

- No module in a reasoning layer imports a provider SDK, DB driver, network,
  process or filesystem module, or mutates the filesystem through a call; and
  none reaches an effect layer except through `executor`.
- Deliberately non-compliant graph nodes are planted, one per category
  (Gmail/Calendar/job-board SDKs, `sqlalchemy`, `httpx`, `open(p, "w")`,
  `Path.write_text`, `os.remove`, a two-hop `graphs → state → persistence`
  reach, …), and each is caught. A node with the documented shape, one that
  reads files, and the executor itself are not flagged.

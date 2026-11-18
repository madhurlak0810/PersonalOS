# Authorization

Phase L (L2): who may see and do what through the API. Authentication (L1,
`apps/api/auth.py`) establishes *who* a request is from. This document covers
what that identity is then allowed to touch.

The MVP has one local user. The model is built for more than one: every check
below already behaves correctly with two actors, and the tests run against two
(`tests/unit/test_actor_scoping.py`). Adding multi-user (OIDC, more than one
token) needs a new `TokenVerifier` and no schema change.

## The three questions

| Question | Answered by | Failure |
| --- | --- | --- |
| Who is calling? | The verified token (`Principal`). Never a header, a body field or a query parameter. | 401 |
| What may this token do? | The token's **scopes**, checked per route. | 403, audited |
| Which objects may it touch? | **Ownership**, checked in the data layer by every store. | 404, audited |

## Identity

`personalos.domain.auth.Principal` is the verified caller:

- `actor_id`: what every route records and every store scopes by.
- `scopes`: what the token allows (see below).
- `kind`: `human` or `system`.

**The actor comes only from the token.** No request body has an `actor_id`,
`user_id`, `decided_by`, `requested_by` or similar field. Every body model is
`extra="forbid"`, so a client that sends one gets a 422 and nothing is stored.
The free-form resume `event` refuses those keys too. No query parameter names
an identity: the board's `user_id` filter is gone. `GET /v1/audit`'s `actor`
parameter is a filter *within the caller's own trail* (for example "what the
worker did on my workflows"). It can never widen the trail to anyone else's.

## Scopes

| Scope | Allows |
| --- | --- |
| `read` | Every `GET`, and issuing a stream ticket |
| `write` | Starting, resuming, retrying, dismissing, moving applications, profile and evidence writes |
| `approve` | `POST /v1/approvals/{id}/decision`, and a resume that carries `decisions` |
| `admin` | Install-wide objects that belong to no one actor: Google consent and revoke, and parked events that belong to no workflow |

`require_route_scope` is mounted on every authenticated router. By default a
`GET` needs `read` and anything else needs `write`. An endpoint that needs
something else says so with `@requires_scope(...)`. The decision endpoint
names `approve`, so a token with `approve` alone can decide but cannot list.

The local token carries `API_TOKEN_SCOPES`, which by default is all four.
`admin` does **not** widen ownership. An admin still sees only its own
workflows.

## Human and system identities

The worker, the scheduler and every model-driven node act as **system
identities**: `system:worker`, `system:scheduler`, `system:approval_expiry`,
`event:<source>`, `model:...`, `agent:...` (`is_system_actor`). A system
identity can never hold `approve`:

- `Principal` refuses to be built that way (`SystemIdentityCannotApprove`).
  The kind follows the actor id's shape, whatever the caller claims.
- `LocalTokenVerifier` drops `approve` from a token whose `API_ACTOR` is a
  system id.
- `ApprovalQueue.decide` takes the `Principal`, not an actor string. It
  refuses a system identity, and also a token without `approve`, with a
  `Forbidden` and an audit row. This holds even when the queue is called
  directly, outside any route.
- `authorize_execution`, the last gate before a side effect, refuses an
  approval whose `decided_by` is a system identity (`SYSTEM_APPROVER`). A
  decision written straight into graph state therefore cannot be spent.

So the agent may propose an action, but it can never approve one. The expiry
sweep still closes overdue approvals as `EXPIRED`. That is a refusal, not an
approval.

**The approval record names the decider.** `approvals.decided_by` (migration
`202611180001`) is the verified actor, stored beside the `action_hash` the
decision is bound to.

## Ownership

**An actor owns the rows keyed by `actor_user_id(actor_id)`.** That is a
stable `users.id` derived from the actor string, or the actor itself when it
is already a UUID. The schema already records an owner on every row that
needs one:

| Object | Owned through |
| --- | --- |
| Workflow, its threads, checkpoints, commands | `workflow_runs.user_id`. A workflow is the actor's when one of its runs is |
| Workflow events (SSE), failures, parked outbox events | The workflow |
| Approvals | The workflow the request is parked in |
| Application, artifact versions, status history, recruiter events | `applications.user_id` |
| Audit events | The workflow. One filed under no workflow belongs to the actor it names (its own denials, for example) |
| Credential references | `credentials.user_id`. One with no owner is install-wide and visible only to `admin` |

`WorkflowThreadRegistry.register` keeps a workflow to one owner. A run for a
different owner is refused (`WorkflowOwnerMismatch`), and a run registered
with no owner inherits the workflow's.

### Enforced in the data layer

Every store method the API reads or writes these through takes `access`. It
is required, keyword-only and has no default, so a route cannot call the store
without saying whose data it wants:

- an `ActorScope` (from `current_access`, which is built from the verified
  principal) filters in the store's own query;
- `SYSTEM_ACCESS`, which only the worker, the scheduler and the CLI's
  install-operator commands pass, sees everything.

Two tests pin this down. `test_scoped_store_methods_cannot_be_called_without_an_access`
checks the signatures. `test_routes_reach_data_only_through_scoped_stores`
checks that no route imports the ORM models, the raw repositories or
SQLAlchemy.

### 404, not 403

An object another actor owns is indistinguishable from one that does not
exist. It returns the same 404 with the same message, on every endpoint:

- by-id reads and writes: workflow, workflow audit, failures, events,
  application, approval decision, resume, transition, event retry, and stream
  ticket issuance;
- lists and pagination: `GET /v1/approvals`, `/v1/applications`, `/v1/audit`
  and `/v1/failures/events` are filtered in the query, so totals and offsets
  count only the caller's rows, and an `application_id` or `workflow_id`
  filter naming someone else's object matches nothing;
- the SSE stream: another actor's workflow is a 404, and a `Last-Event-ID`
  cursor taken from it is a 422 on the caller's own stream, never a replay.

### Stream tickets

A ticket (`POST /v1/auth/stream-tickets`) is issued only for a workflow the
caller owns. It is HMAC-signed, names one actor and one workflow, and expires
after `API_STREAM_TICKET_SECONDS`. When the stream opens, the ticket is
checked against the workflow in the path, and the stream is then read within
the ticket actor's scope. The ticket's principal holds `read` only, and the
ticket is not accepted as a bearer token. A genuine ticket used for another
workflow, or used after it expires, is a 401 and an audit row against the
actor it was issued to.

## Denials are audited

Every authorization denial appends an `audit_events` row
(`personalos.persistence.ownership.record_denial`):

| Field | Value |
| --- | --- |
| `actor` | The denied actor |
| `action` | `authz.denied.<reason>`: `not_visible`, `missing_scope`, `system_identity`, `ticket_wrong_workflow`, `ticket_expired` |
| `target_ref` | `<object type>:<id> <METHOD> <path>` |
| `workflow_id` | Always empty, so the row is the denied actor's own record and nobody else's |
| `result` | `failure` |

The row goes through `AuditEventRepository`, which applies the Phase D
redaction. The route is the method and path only. The query string (where a
ticket travels) and the headers (where the token travels) are never recorded.

A by-id miss is recorded as `not_visible` whether the object belongs to
someone else or does not exist. The denied actor can read its own denials, so
recording the two differently would tell it which ids are real.

Unauthenticated requests (a missing or bad token) are 401s and are not
audited. There is no actor to file them under, and an anonymous caller
must not be able to fill the trail.

## Not covered yet

- **Delegation.** An actor decides only its own approvals. A separate
  reviewer acting on someone else's workflows would need an explicit grant,
  which this model does not have yet.
- **Executor credential selection.** The worker's credential broker still
  picks the install's Google connection (`SYSTEM_ACCESS`). Per-owner
  connections would need the broker to choose by the workflow's owner.

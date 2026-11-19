# API limits

These limits cap how much one caller can ask of the API. Without them, a buggy
UI loop, a stolen token or a hostile client could fill the worker queue, spend
local model time, flood approvals, hold open any number of event streams,
brute-force the local token, or send huge bodies. Rates, concurrency caps and
the auth throttle live in `apps/api/limits.py`. The body, depth and string caps
live in `apps/api/security.py`. Every refusal is reported by
`personalos/observability/limits.py`.

Every limit is keyed on something the server verified: the authenticated actor,
or the connection's peer address for authentication failures. Headers such as
`X-Actor-Id` and `X-Forwarded-For`, and any body field, have no effect.

`/health` is never limited.

## Responses

| Status | `error_code` | When |
| --- | --- | --- |
| 429 | `rate_limited` | A rate, a concurrency cap or the auth throttle. Carries `Retry-After`, in whole seconds. |
| 413 | `payload_too_large` | The body is over its cap. Sent before authentication, before the route. |
| 422 | `validation` | The JSON is malformed, not UTF-8, nested too deep, has a string that is too long, or asks for a page that is too large. |

The body has the same three fields as every other error: `error_code`,
`message` and `context_id`.

## Rates (per actor, token bucket)

Each bucket holds up to N tokens and refills at N per minute. A request spends
one token. With an empty bucket, `Retry-After` is the time until the next token.

| Route | Limit | Setting | Default |
| --- | --- | --- | --- |
| `POST /v1/chat` | `workflow_start` | `API_RATE_WORKFLOW_START_PER_MINUTE` | 10 |
| `POST /v1/workflows/{id}/resume` | `workflow_resume` | `API_RATE_WORKFLOW_RESUME_PER_MINUTE` | 30 |
| `POST /v1/approvals/{id}/decision` | `approval_decision` | `API_RATE_APPROVAL_DECISION_PER_MINUTE` | 60 |
| `POST /v1/workflows/{id}/retry`, `.../dismiss`, `POST /v1/failures/events/{id}/retry` | `failure_action` | `API_RATE_FAILURE_ACTION_PER_MINUTE` | 30 |
| `POST /v1/evidence/documents` | `evidence_ingest` | `API_RATE_EVIDENCE_INGEST_PER_MINUTE` | 6 |

The buckets live behind the `RateLimitStore` protocol. The one implementation,
`InMemoryRateLimitStore`, keeps them in each process's memory, so with
`API_WORKERS=4` an actor can make up to four times each rate. A Postgres or
Redis store would implement `consume(key, limit)` and be passed to `ApiLimits`.
No route would change.

## Concurrency (per actor)

| Cap | Setting | Default | Counted |
| --- | --- | --- | --- |
| Active workflows: a command queued or running | `API_MAX_ACTIVE_WORKFLOWS` | 5 | From the command queue, across processes |
| Open event streams | `API_MAX_SSE_CONNECTIONS` | 5 | In this process |

The active-workflow cap applies when a request would start a new workflow. A
workflow stops counting once the worker settles its command: the run completes,
fails, or parks at an approval. Going over the cap returns a 429. The request
is not queued.

## Authentication failures (per source address)

Each failed bearer token or stream ticket counts against the connection's peer
address:

- The first `API_AUTH_FREE_FAILURES` (default 3) cost nothing, so a typo
  doesn't lock anyone out.
- Each failure after those blocks the address for twice as long as the last:
  `API_AUTH_BACKOFF_BASE_SECONDS` (1s), then 2s, 4s, and so on, up to
  `API_AUTH_BACKOFF_MAX_SECONDS` (300s).
- While an address is blocked, every credential from it gets a 429 without
  being checked, including the right one.
- A successful credential clears the address's count, and so do 15 minutes
  without a failure.

A request with no credential is a plain 401 and is not counted.

## Request limits

| Limit | Setting | Default |
| --- | --- | --- |
| Request body | `API_MAX_BODY_BYTES` | 1 MiB |
| Body for `POST /v1/evidence/documents` only | `API_MAX_RESUME_BODY_BYTES` | 10 MiB |
| JSON nesting depth | `API_MAX_JSON_DEPTH` | 32 |
| Longest JSON string, in bytes as sent | `API_MAX_JSON_STRING_BYTES` | 64 KiB |
| Page size on list endpoints (`?limit=`) | `apps.api.limits.MAX_PAGE_SIZE` | 200 |

`JsonLimitsMiddleware` checks depth, string length and UTF-8 in one linear
pass over the raw bytes, before any parser runs. The body cap already bounds
those bytes. Free-text fields also have their own `max_length`: the chat
message allows 10,000 characters, ids 255, and an approval note 2,000.

A `?limit=` over the maximum is a 422. A list endpoint without paging returns
at most `MAX_PAGE_SIZE` items.

## Idempotent replays

`POST /v1/chat` accepts an `Idempotency-Key` header. A repeat of a request
that was already answered returns the first answer, with
`Idempotent-Replayed: true`. A replay doesn't spend `workflow_start` budget
and doesn't create a workflow.

- Reusing a key with a different body is a 409 `idempotency_conflict`.
- A key whose request is still in flight is also a 409.
- If a request with a key was refused, for example rate-limited, the key can
  be used again.
- Keys are per actor. They are stored hashed in the `operations` ledger.

## Observability

For each refusal, `record_limit_hit` does four things:

- adds 1 to the `personalos.api.limit_hits` counter, labelled with `route` and
  `limit`;
- writes an `api.limit_hit` log line at WARNING with `actor_id`, `route`,
  `limit_name` and `retry_after_seconds`. For an auth-throttle hit it also
  includes `source_address`;
- records an `api.limit_hit` span with the same attributes;
- increments a per-process tally, readable at `GET /v1/failures/limits`. A
  caller sees only its own hits. A token with `admin` sees everyone's,
  including anonymous hits.

`route` is the method and path, with ids replaced by `{id}`. The query string,
token and body are never included, and `tests/unit/test_api_limits.py` checks
this.

Set `API_LIMITS_ENABLED=false` to turn off the rates, caps and throttle. The
body, depth and string caps stay on.

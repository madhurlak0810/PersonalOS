# Event Transport

How asynchronously dispatched events get from the producer that staged them to
the worker that handles them, which guarantees every transport has to give, and
what changes when the Postgres outbox is replaced by NATS JetStream or Redis
Streams.

The enforced version of this document is spread over three places:

- the ports, [`personalos/domain/event_transport.py`](../personalos/domain/event_transport.py);
- the contract suite every transport passes,
  [`tests/contracts/test_event_transport.py`](../tests/contracts/test_event_transport.py);
- the import rule that keeps the outbox an implementation detail,
  `OUTBOX_INTERNALS` / `OUTBOX_OWNERS` in
  [`tests/architecture/boundaries.py`](../tests/architecture/boundaries.py).

## The shape

```
producer ──stage_event / publish──▶ EventPublisher ┐
                                                   │  transport: Postgres outbox today,
worker   ◀──claim / ack / nack / dead_letter────── EventConsumer ┘  in-memory in tests, a broker later
   │
   └─▶ handler(EventEnvelope)  ──▶ IdempotentConsumer (consumed_events, same txn as the effect)
```

| Piece | Where | Knows the transport? |
| --- | --- | --- |
| `EventEnvelope` | `personalos/domain/events.py` | No. Identical on every transport. |
| `EventPublisher`, `EventConsumer`, `Delivery`, `FailureOutcome`, `classify_failure` | `personalos/domain/event_transport.py` | No. These are the ports. |
| `EventOutbox` (Postgres/SQLite) | `personalos/persistence/event_outbox.py` | Yes. First implementation. |
| `InMemoryEventTransport` | `personalos/events/memory.py` | Yes. Second implementation, for unit tests. |
| `OutboxDispatcher` | `apps/worker/outbox_dispatcher.py` | No. It is typed against `EventConsumer`. |
| `EventRouter`, `ApplicationTransitionHandler` | `apps/worker/event_routing.py` | No. They take an `EventEnvelope`. |
| `IdempotentConsumer` (`consumed_events`) | `personalos/persistence/event_inbox.py` | No. It belongs to the effect's database, not to the transport (see below). |
| Wiring | `personalos/bootstrap.py` (`build_event_outbox`) | Yes. It is the only place that picks a transport. |

Graphs never see events directly. They get one as a resume value or as graph
input (`{"event": <envelope>}`) through a `WorkflowCommand` that `EventRouter`
queues.

### Who may touch the outbox tables

Only the modules that implement the outbox may name `OutboxEventModel`,
`ConsumedEventModel` or `OutboxEventRepository`. That covers
`persistence.models` and `persistence.repositories` (where they are defined),
`persistence.event_outbox` (the transport), `persistence.event_inbox` (the
consumer inbox), `persistence.event_failures` (the operator's view of parked
events) and `persistence.workflow_commands` (the workflow-command queue, which
uses the same table under its own lease). The check runs on the AST, so it
covers names imported directly and names reached as an attribute of an imported
module. It runs in `pytest tests/architecture` and in
`python scripts/check_boundaries.py`.

Producers inside `persistence` that write an event in their own transaction go
through `stage_event` (enveloped) or `stage_raw_event` (the older bare-payload
shape some consumers still read). They do not construct the row themselves.

## The contract

Every implementation passes `tests/contracts/test_event_transport.py` unchanged.
Today that covers three runs: `memory`, `outbox-sqlite`, and `outbox-postgres`
(`SKIP LOCKED`, skipped when no Postgres is reachable; CI provides one). A new
transport is added by giving the suite a builder in `TRANSPORTS`.

**Envelope.** What `claim` hands out compares equal to what was published, field
for field.

**Publish deduplication.** Publishing a `dedupe_key` the transport already holds
stores nothing and returns `Published(duplicate=True)` with the first copy's
`event_id`. The suite only checks an *immediate* re-publish, plus a re-publish
after delivery. That is the minimum a broker with a bounded dedupe window can
meet. It is a courtesy to producers. Correctness comes from the consumer side.

**At-least-once delivery.** `claim` leases one ready event for the consumer's
visibility timeout. If the event is not settled before the lease lapses, it is
delivered again with `attempt + 1`. An acked event is never delivered again. A
lease that lapses `max_attempts` times without being settled dead-letters the
event, so a handler that keeps crashing its worker is eventually given up on.

**Effects happen once because consumers deduplicate.** Handlers are idempotent on
`dedupe_key`, never `event_id`. `IdempotentConsumer` records
`(consumer, dedupe_key)` in `consumed_events` in the same transaction as the
effect. This stays in Postgres whatever carries the events: the record has to
commit atomically with the effect, and the effect is a Postgres write.

**Nack.** `nack(delivery, error)` asks `classify_failure` what to do, so every
transport makes the same decision:

| Error | Outcome |
| --- | --- |
| `ValidationFailed` (poison) | `QUARANTINED` on the first failure, no retry |
| non-retryable `PersonalOSError` (terminal) | `DEAD_LETTERED`, no retry |
| anything else, budget left | `RETRY_SCHEDULED`; claimable again after exponential, jittered backoff, never sooner than the error's `Retry-After` |
| anything else, last attempt | `DEAD_LETTERED` |

An event that is backing off does not hold up the events behind it.

**Explicit dead-letter.** `dead_letter(delivery, reason)` parks the event for an
operator straight away, whatever budget it has left.

**Stale deliveries settle nothing.** Suppose a lease lapsed and another consumer
claimed the event. The old holder's `ack` then returns `False`, its `nack`
returns `CLAIM_LOST`, and its `dead_letter` returns `False`. The current holder
is unaffected. A lapsed lease that nobody has reclaimed can still be settled by
its holder.

**Ordering, which is deliberately weak.** One consumer that claims and settles one
event at a time sees ready events in publish order. Nothing is promised beyond
that. Concurrent consumers interleave, a retry moves an event behind newer ones
until its backoff passes, and a lapsed lease is redelivered whenever it is next
claimed. **Handlers must not depend on order.** If order matters, the handler
works it out from the payload (for example, `ApplicationRepository.update_status`
refuses a transition the lifecycle does not allow from the current status).

**Concurrency.** Racing consumers never hold the same event at once.

**Not in the contract** (each one is Postgres-specific today, or an operator
concern):

- Subscription filters. The outbox's `exclude_types` is how the event dispatcher
  stays off `workflow.command` rows. A consumer's filter is fixed when it is
  built.
- Operator requeue (`EventOutbox.requeue`, `InMemoryEventTransport.requeue`).
- Listing parked events (`EventFailureStore`).
- Workflow attribution of a dead-lettered event (`outbox_events.workflow_id`).
- Quarantining an envelope that cannot be read at claim time. A broker adapter
  should do the same when a message fails to deserialize.

## Semantics by transport

| | Postgres outbox (today) | NATS JetStream | Redis Streams |
| --- | --- | --- | --- |
| **Atomic with the producer's mutation** | Yes. `stage_event` joins the producer's transaction. | No. A broker cannot join a Postgres transaction. Keep the outbox as the staging table and add a relay (below). | No. Same as JetStream. |
| **Publish dedupe** | Unique `dedupe_key`, for as long as the row exists (no window). | `Nats-Msg-Id` = `dedupe_key`, only inside the stream's `duplicate_window` (default 2 min). | None built in. The adapter does `SET dedupe:<key> NX EX <ttl>` and `XADD` atomically (Lua or `MULTI`); the window is the TTL. |
| **Claim / lease** | `SELECT … FOR UPDATE SKIP LOCKED`, lease = `visibility_timeout` | Pull consumer `fetch(1)`, lease = `AckWait` | `XREADGROUP`, lease = time in the pending entries list; reclaim with `XAUTOCLAIM min-idle-time=visibility_timeout` |
| **`attempt`** | `outbox_events.attempts` | `msg.metadata.num_delivered` | delivery count from `XPENDING` / `XAUTOCLAIM` |
| **Ack** | Conditional `UPDATE` on `(id, status, claimed_at)` | `msg.ack()` | `XACK` |
| **Stale settle refused** | Yes: the claim stamp is the token. | Acks are tracked per message, not per delivery, so a superseded delivery's late ack may well be accepted. **Verify this against the contract test**; if it is accepted, the adapter has to guard it or the contract has to be relaxed on purpose. | No: `XACK` acks by entry id from any consumer. The adapter checks ownership and delivery count in a Lua script before acking. |
| **Delayed retry (backoff)** | `available_at` | `msg.nak(delay=…)` | Nothing built in. Either `XACK` and re-`XADD` through a delay sorted set (which moves the event to the tail), or leave it pending and let `XAUTOCLAIM` pick it up (delay = visibility timeout, so backoff is lost). |
| **Lapsed-lease budget** | `attempts > max_attempts` dead-letters at claim | `MaxDeliver` + the `MAX_DELIVERIES` advisory | Adapter checks the delivery count on `XAUTOCLAIM` |
| **Dead-letter / quarantine** | A status on the row, with `last_error`, `failed_at`, `workflow_id` | `msg.term()` + publish to a `…dlq` / `…quarantine` subject, with the error in headers. There is no native DLQ. | `XACK` + `XADD` to a `:dlq` / `:quarantine` stream |
| **Ordering** | `(created_at, id)`. Contract FIFO holds. | Stream order per subject. Contract FIFO holds with `MaxAckPending=1`; larger values deliver concurrently. | Entry-id order. Contract FIFO holds; re-`XADD` retries reorder further. |
| **Replay** | None. Dispatched rows stay in the table, but there is no API to replay them, and `event_log` is the durable history, not the outbox. | Yes: a new consumer with `DeliverPolicy` by sequence or time, within the stream's retention limits. | Yes: `XRANGE` or a new group from `0`, within `MAXLEN` / `MINID` trimming. |
| **Durability of an acked publish** | Postgres commit | Replicated stream (`R3` recommended) | Depends on AOF `appendfsync`; the default `everysec` can lose about 1s of writes |
| **Client style** | Sync (SQLAlchemy) | `nats-py` is asyncio-only | `redis-py` sync or async |

### Semantics that change, and what to do about each

1. **Dedupe gets a window.** On the outbox, a re-publish of the same fact is
   dropped for as long as the row exists. On JetStream it is dropped only inside
   `duplicate_window`; on Redis, only inside the dedupe-key TTL. A producer that
   retries after the window has passed delivers a second copy. That is safe
   *only* because every state-changing consumer goes through `IdempotentConsumer`
   or a deduplicated command (`EventRouter`). Before switching, confirm every
   handler registered in `build_outbox_dispatcher` is idempotent on
   `dedupe_key`. Today they all are.
2. **Publishing stops being atomic with the mutation.** Do not call a broker from
   inside a producer's transaction. Keep `stage_event` and `outbox_events` as the
   staging area and relay from there (step 3 below). Producers do not change.
3. **Ordering gets weaker with redelivery and concurrency** on every transport,
   and the contract already assumes the weakest version. On Redis, a
   re-`XADD`-based backoff sends the event to the tail of the stream.
4. **Replay becomes possible.** That is new capability, not a change to existing
   behaviour, but a replay redelivers events whose effects have already landed.
   It is safe because the inbox dedupes it. An event the inbox has never seen
   (for example, one that failed every attempt and was dead-lettered) is applied
   for the first time.
5. **Operator requeue changes shape.** `EventOutbox.requeue` resets a parked row
   in place. On a broker, a requeue republishes from the DLQ stream. On JetStream
   it **must use a fresh `Nats-Msg-Id`**, or a requeue inside the duplicate
   window is silently dropped. The envelope's `dedupe_key` is unchanged, so the
   inbox still dedupes it.
6. **The failures view moves.** `EventFailureStore` (`GET /v1/failures/events`,
   `GET /v1/workflows/{id}/failures`) queries `outbox_events` for parked rows.
   On a broker it reads the DLQ stream, and the workflow attribution
   (`_registered_workflow`) is computed when the event is dead-lettered and
   carried in a header.
7. **Stale-settle protection might not come for free** (see the table). The
   contract test `test_a_delivery_reclaimed_by_another_consumer_cannot_be_settled_by_the_first`
   is where you find out.
8. **The client may be async.** The ports are synchronous because the outbox and
   the dispatcher's callers are. A JetStream adapter either runs `nats-py` on a
   dedicated event loop thread behind the sync methods, or the ports get `async`
   variants. That would be a change to the dispatcher (which is already async)
   and to this suite, not to any handler.

## Upgrade path

Each step can ship on its own and is reversible until the last one.

1. **Ports, contract, in-memory transport, import rule.** *Done in this
   change.* Handlers and the dispatcher depend on `EventConsumer` and
   `EventEnvelope`; nothing outside the outbox modules names the outbox tables.
2. **Write the adapter.** Put it in a new effect layer (for example,
   `personalos/transports/`, allowing `domain` and `config`, added to
   `EFFECT_LAYERS` in `tests/architecture/intent_boundary.py`). It must not live
   in `personalos/events`, which is a reasoning layer and does no I/O. Map the
   envelope to the message body as `envelope.model_dump(mode="json")`, and put
   `type`, `dedupe_key` and `event_id` in headers. Add a builder to `TRANSPORTS`
   in the contract suite (skipped when the broker is unreachable, like the
   Postgres run) and get it passing unchanged. Each place the adapter cannot
   meet a contract test is a decision to write down here, not a test to delete.
3. **Relay the outbox to the broker.** Keep producers on `stage_event`. Run an
   `OutboxDispatcher` over the existing `EventOutbox` whose default handler
   publishes each envelope to the broker and returns once the broker has
   acknowledged it. The dispatcher already acks only after its handler returns,
   so the relay is at-least-once. Broker-side dedupe (`Nats-Msg-Id` /
   `SET NX`) absorbs most relay redeliveries, and the consumer inbox absorbs the
   rest.
4. **Switch consumption.** In `build_outbox_dispatcher`, hand the dispatcher the
   broker's `EventConsumer` instead of `build_event_outbox(...)`. That is the
   only line in the worker that changes. To cut over, stop the old dispatchers,
   start the relay, then start the broker consumers. Events still `pending` in
   the outbox are relayed, not lost.
5. **Move the operator surfaces.** Port `EventFailureStore` and `requeue` to the
   DLQ streams (point 5 above), and keep the API routes as they are.
6. **Leave alone:** `workflow.command` rows (`WorkflowCommandQueue`). They are a
   command queue, not an event stream. A command runs as long as its graph does,
   under a workflow lease, and the API enqueues it next to its own writes. They
   stay on Postgres, and the relay must exclude them, as `build_event_outbox`'s
   `exclude_types` already does. Also leave the consumer inbox
   (`consumed_events`) alone: it commits with the effect.

## Adding a transport, in short

Implement `EventPublisher` and `EventConsumer`, subclass `Delivery` to carry your
receipt, decide nacks with `classify_failure`, add a builder to the contract
suite, and wire it in `bootstrap`. If you find yourself importing
`OutboxEventModel` anywhere else, the boundary check will tell you that you are
building the wrong thing.

Trace context rides inside the envelope (`EventEnvelope.trace_context`, a W3C
`traceparent`/`baggage` carrier), so a transport that stores and returns
envelopes whole carries it with nothing extra, and the dispatcher's
`event.deliver` span continues the producer's trace from it. The Postgres
outbox fills it in (`observability.tracing.inject_trace_context()`) when it
stages an envelope the producer left it empty on; a new transport's `publish`
should do the same. The in-memory transport does not, because the `events`
layer imports nothing but `domain`. Do not move the context into broker headers
unless you also read it back from them.

"""The worker half of event delivery: claim an event, deliver it, settle it.

`OutboxDispatcher.process_next` is one delivery: claim the next ready event
from an `EventConsumer`, hand its envelope to the handler registered for its
type (or the default handler, typically `EventRouter`), then `ack` it. The
dispatcher knows the transport only through that port: in production it is
the Postgres outbox (`EventOutbox`, built by `build_event_outbox`), whose claim
is `SELECT ... FOR UPDATE SKIP LOCKED`, so any number of dispatchers can run
against the same database at once and each row goes to one of them.

What a dispatcher promises is *at-least-once*, not exactly-once. It acks only
after its handler returns, so a crash in between -- after the effect, before
the ack -- leaves the event claimed by nobody, and once the claim's lease runs
out another dispatcher delivers it again. That redelivery is safe because
handlers are idempotent on the envelope's `dedupe_key`
(`IdempotentConsumer`, or `EventRouter`'s deduplicated commands), which is the
other half of the bargain and the one that makes an event *apply* once.

A handler that raises is not retried on the spot: `EventConsumer.nack` puts the
event back in line after an exponential, jittered backoff (or the error's
`Retry-After`), and dead-letters it once the retry policy is spent. A
malformed event (`ValidationFailed`) is quarantined at once; delivering it
again would fail the same way. A dispatcher that is cancelled or killed
mid-delivery settles nothing: the event's lease runs out and another dispatcher
picks it up.

`run` is the polling loop the worker process drives: deliver while there is
work, sleep `idle_seconds` when there is none, stop when told.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection, Mapping
from functools import partial
from typing import Any

from opentelemetry.trace import SpanKind

from apps.worker.event_routing import (
    APPLICATION_TRANSITION_CONSUMER,
    ApplicationTransitionHandler,
    EventRouter,
)
from personalos.bootstrap import (
    build_correlated_wait_lookup,
    build_event_outbox,
    build_idempotent_consumer,
    build_workflow_command_queue,
    build_workflow_thread_registry,
    register_event_handler_thread,
)
from personalos.domain.event_transport import EventConsumer
from personalos.domain.events import EventEnvelope, EventType
from personalos.observability import tracing
from personalos.observability.tracing import Attr
from personalos.persistence.database import SessionLocal

logger = logging.getLogger(__name__)

#: Consumes one event. Must be idempotent on `envelope.dedupe_key`.
EventHandler = Callable[[EventEnvelope], Awaitable[Any]]


class OutboxDispatcher:
    """Delivers events to their handlers, one claimed event at a time."""

    def __init__(
        self,
        consumer: EventConsumer,
        handlers: Mapping[str, EventHandler],
        *,
        default: EventHandler | None = None,
    ):
        """Initialize with the consumer, per-type handlers, and the handler for every other type.

        An event with neither a handler for its type nor a default has nobody
        to deliver to, and is acknowledged rather than left to block the queue.
        """
        self.consumer = consumer
        self.handlers = dict(handlers)
        self.default = default

    async def process_next(self) -> bool:
        """Claim and deliver the oldest claimable event. False when there was none."""
        claimed = self.consumer.claim()
        if claimed is None:
            return False
        envelope = claimed.envelope
        handler = self.handlers.get(envelope.type, self.default)
        # Continues the publisher's trace: the envelope carries its context.
        with tracing.attached(tracing.extract_trace_context(envelope.trace_context)):
            with tracing.span(
                "event.deliver",
                {
                    Attr.EVENT_TYPE: envelope.type,
                    Attr.EVENT_ID: envelope.event_id,
                    Attr.CORRELATION_ID: envelope.correlation_id,
                    Attr.ATTEMPT: claimed.attempt,
                },
                kind=SpanKind.CONSUMER,
            ) as span:
                try:
                    if handler is None:
                        logger.debug(
                            "no consumer for %s events; acknowledging %s",
                            envelope.type,
                            envelope.event_id,
                        )
                    else:
                        await handler(envelope)
                except Exception as error:
                    # Not `BaseException`: a cancelled or killed dispatcher must leave
                    # the claim alone, for its lease to lapse and the row to come back.
                    logger.exception(
                        "delivering event %s (%s) failed on attempt %d",
                        envelope.event_id,
                        envelope.type,
                        claimed.attempt,
                    )
                    failure = self.consumer.nack(claimed, error)
                    tracing.mark_failed(span, error)
                    tracing.set_attributes(
                        span, {Attr.EVENT_OUTCOME: failure.outcome, Attr.RETRY_DELAY: failure.delay}
                    )
                    return True
                self.consumer.ack(claimed)
                tracing.set_attributes(span, {Attr.EVENT_OUTCOME: "acked"})
        return True

    async def drain(self, limit: int | None = None) -> int:
        """Deliver until nothing is claimable (or `limit` deliveries). Returns how many."""
        delivered = 0
        while limit is None or delivered < limit:
            if not await self.process_next():
                break
            delivered += 1
        return delivered

    async def run(self, stop: asyncio.Event, *, idle_seconds: float = 1.0) -> None:
        """Deliver events until `stop` is set, pausing `idle_seconds` whenever nothing is ready."""
        while not stop.is_set():
            try:
                delivered = await self.process_next()
            except Exception:
                # A claim or settle that could not reach the transport. The
                # event, if any, is still leased and will come back; the loop
                # must outlive a database (or broker) blip.
                logger.exception("outbox dispatch failed; backing off")
                delivered = False
            if not delivered:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=idle_seconds)
                except asyncio.TimeoutError:
                    pass


def build_outbox_dispatcher(
    session_factory: Callable[[], Any] = SessionLocal,
    *,
    start_types: Collection[str] = frozenset(),
) -> OutboxDispatcher:
    """The dispatcher a worker process runs, over one database.

    `application.transition_requested` events are applied directly; every
    other event goes to `EventRouter`, which resumes the workflow waiting on
    it. `start_types` names the event types that start an `event_handler` run
    when nothing is waiting -- empty by default, because such a run needs a
    runner wired for `event_handler` threads in the workflow-command worker,
    and starting runs nothing can execute would only fill the command queue.
    """
    registry = build_workflow_thread_registry(session_factory)
    router = EventRouter(
        build_workflow_command_queue(session_factory),
        build_correlated_wait_lookup(session_factory, registry),
        partial(register_event_handler_thread, registry=registry),
        start_types=start_types,
    )
    transitions = ApplicationTransitionHandler(
        build_idempotent_consumer(APPLICATION_TRANSITION_CONSUMER, session_factory)
    )
    return OutboxDispatcher(
        build_event_outbox(session_factory),
        {EventType.APPLICATION_TRANSITION_REQUESTED: transitions},
        default=router,
    )


__all__ = ["EventHandler", "OutboxDispatcher", "build_outbox_dispatcher"]

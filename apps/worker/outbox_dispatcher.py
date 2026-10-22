"""The worker half of the outbox: claim an event, deliver it, settle it.

`OutboxDispatcher.process_next` is one delivery: claim the oldest claimable
row from `EventOutbox`, hand its envelope to the handler registered for its
type (or the default handler, typically `EventRouter`), then `ack` it. Any
number of dispatchers can run this against the same database at once: the
claim is `SELECT ... FOR UPDATE SKIP LOCKED`, so each row goes to one of them.

What a dispatcher promises is *at-least-once*, not exactly-once. It acks only
after its handler returns, so a crash in between -- after the effect, before
the ack -- leaves the row claimed by nobody, and once the claim's lease runs
out another dispatcher delivers it again. That redelivery is safe because
handlers are idempotent on the envelope's `dedupe_key`
(`IdempotentConsumer`, or `EventRouter`'s deduplicated commands), which is the
other half of the bargain and the one that makes an event *apply* once.

A handler that raises is not retried on the spot: the row is left claimed and
comes back when its lease expires, until `EventOutbox.max_attempts`. A
malformed event (`ValidationFailed`) is given up on at once; delivering it
again would fail the same way.

`run` is the polling loop the worker process drives: deliver while there is
work, sleep `idle_seconds` when there is none, stop when told.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection, Mapping
from functools import partial
from typing import Any

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
from personalos.domain.errors import ValidationFailed
from personalos.domain.events import EventEnvelope, EventType
from personalos.persistence.database import SessionLocal
from personalos.persistence.event_outbox import EventOutbox

logger = logging.getLogger(__name__)

#: Consumes one event. Must be idempotent on `envelope.dedupe_key`.
EventHandler = Callable[[EventEnvelope], Awaitable[Any]]


class OutboxDispatcher:
    """Delivers outbox events to their handlers, one claimed row at a time."""

    def __init__(
        self,
        outbox: EventOutbox,
        handlers: Mapping[str, EventHandler],
        *,
        default: EventHandler | None = None,
    ):
        """Initialize with the outbox, per-type handlers, and the handler for every other type.

        An event with neither a handler for its type nor a default has nobody
        to deliver to, and is acknowledged rather than left to block the queue.
        """
        self.outbox = outbox
        self.handlers = dict(handlers)
        self.default = default

    async def process_next(self) -> bool:
        """Claim and deliver the oldest claimable event. False when there was none."""
        claimed = self.outbox.claim()
        if claimed is None:
            return False
        envelope = claimed.envelope
        handler = self.handlers.get(envelope.type, self.default)
        try:
            if handler is None:
                logger.debug(
                    "no consumer for %s events; acknowledging %s", envelope.type, envelope.event_id
                )
            else:
                await handler(envelope)
        except ValidationFailed:
            logger.exception(
                "event %s (%s) is malformed and will not be redelivered",
                envelope.event_id,
                envelope.type,
            )
            self.outbox.nack(claimed, permanent=True)
            return True
        except Exception:
            logger.exception(
                "delivering event %s (%s) failed on attempt %d",
                envelope.event_id,
                envelope.type,
                claimed.attempt,
            )
            self.outbox.nack(claimed)
            return True
        self.outbox.ack(claimed)
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
        """Deliver events until `stop` is set, pausing `idle_seconds` whenever the outbox is empty."""
        while not stop.is_set():
            try:
                delivered = await self.process_next()
            except Exception:
                # A claim or settle that could not reach the database. The
                # row, if any, is still leased and will come back; the loop
                # must outlive a database blip.
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

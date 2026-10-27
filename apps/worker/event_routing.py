"""Where a dispatched event goes: into a waiting workflow, or a run of its own.

`EventRouter` is the dispatcher's handler for events that drive graphs. It
never runs a graph itself -- it queues a `WorkflowCommand`, exactly as the API
does, and the workflow-command worker runs it under the workflow's lease.
That keeps a dispatcher's claim short (a queue insert, not a model call) and
leaves one path, not two, by which a graph is ever run.

- **Resume.** If a workflow parked on an interrupt shares the event's
  `correlation_id`, the event is the answer: a RESUME command whose value is
  `{"event": <envelope>}`, delivered to the graph as `Command(resume=...)`.
- **Start.** Otherwise, for event types configured to start runs, a fresh
  `event_handler` thread -- one per event fact -- is registered and a START
  command queued with the same `{"event": <envelope>}` as its input.
- Anything else has no graph to go to and is acknowledged as delivered.

Routing is idempotent on the event's `dedupe_key`: the command's own dedupe
key is derived from it, and `WorkflowCommandQueue` refuses a second command
under the same key, so a redelivered event queues nothing new. The START
thread id is derived from the same key, so a redelivery that lands after the
first run was registered finds that run rather than making another.

`ApplicationTransitionHandler` is the other kind of consumer: one that applies
an effect directly, through `IdempotentConsumer`, rather than handing off to a
graph.
"""

import hashlib
import logging
from collections.abc import Callable, Collection
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from personalos.domain.events import EventEnvelope, InvalidEventEnvelope
from personalos.domain.models import ApplicationStatus, InvalidApplicationTransition
from personalos.domain.workflow import WorkflowCommand, WorkflowCommandKind, WorkflowThread
from personalos.observability import tracing
from personalos.observability.tracing import Attr
from personalos.persistence.event_correlation import WaitingThread
from personalos.persistence.event_inbox import Consumption, IdempotentConsumer
from personalos.persistence.repositories import ApplicationRepository
from personalos.persistence.workflow_commands import (
    DuplicateWorkflowCommand,
    WorkflowCommandQueue,
)

logger = logging.getLogger(__name__)


class WaitLookup(Protocol):
    """Finds the waiting thread an event's correlation belongs to."""

    def waiting_for(self, correlation_id: UUID) -> WaitingThread | None:
        """The waiting thread under `correlation_id`, if any."""


#: Registers (or finds) the thread a new run for `envelope` goes on.
EventStarter = Callable[[EventEnvelope], WorkflowThread]


class RouteKind(str, Enum):
    """What routing did with an event."""

    RESUMED = "resumed"
    STARTED = "started"
    #: The command this event routes to was queued by an earlier delivery.
    ALREADY_ROUTED = "already_routed"
    #: Nothing waits on it and its type does not start runs.
    UNROUTED = "unrouted"


@dataclass(frozen=True)
class RoutedEvent:
    """The outcome of routing one event."""

    kind: RouteKind
    workflow_id: UUID | None = None
    thread_id: str | None = None


def resume_value(envelope: EventEnvelope) -> dict[str, Any]:
    """What a graph receives for an event: as its resume value or its input."""
    return {"event": envelope.model_dump(mode="json")}


def _command_dedupe(envelope: EventEnvelope) -> str:
    # Hashed: the command's dedupe key also carries the workflow id and kind,
    # and the event's own key may already be up to the column's 255 characters.
    digest = hashlib.sha256(f"{envelope.type}\x1f{envelope.dedupe_key}".encode()).hexdigest()
    return f"event:{digest[:32]}"


class EventRouter:
    """Routes an event to a waiting workflow, or to a new event-handler run."""

    def __init__(
        self,
        commands: WorkflowCommandQueue,
        waits: WaitLookup,
        starter: EventStarter,
        *,
        start_types: Collection[str] | None = None,
    ):
        """Initialize with the command queue, the wait lookup and the run starter.

        `start_types` limits which event types start a run when nothing is
        waiting for them; `None` lets every type start one.
        """
        self.commands = commands
        self.waits = waits
        self.starter = starter
        self.start_types = None if start_types is None else frozenset(start_types)

    async def __call__(self, envelope: EventEnvelope) -> RoutedEvent:
        """Route one event. Raises only for a failure worth redelivering."""
        waiting = (
            self.waits.waiting_for(envelope.correlation_id)
            if envelope.correlation_id is not None
            else None
        )
        if waiting is not None:
            command = WorkflowCommand(
                command_id=uuid4(),
                kind=WorkflowCommandKind.RESUME,
                workflow_id=waiting.workflow_id,
                thread_id=waiting.thread_id,
                actor_id=waiting.actor_id,
                correlation_id=envelope.correlation_id,
                graph_input=resume_value(envelope),
                interrupt_ids=waiting.interrupt_ids,
            )
            return self._enqueue(command, envelope, RouteKind.RESUMED)

        if self.start_types is not None and envelope.type not in self.start_types:
            logger.debug("no workflow waits on %s and it starts no run", envelope.dedupe_key)
            return RoutedEvent(kind=RouteKind.UNROUTED)

        thread = self.starter(envelope)
        command = WorkflowCommand(
            command_id=uuid4(),
            kind=WorkflowCommandKind.START,
            workflow_id=thread.workflow_id,
            thread_id=thread.thread_id,
            actor_id=f"event:{envelope.source}",
            correlation_id=envelope.correlation_id or envelope.event_id,
            graph_input=resume_value(envelope),
        )
        return self._enqueue(command, envelope, RouteKind.STARTED)

    def _enqueue(
        self, command: WorkflowCommand, envelope: EventEnvelope, kind: RouteKind
    ) -> RoutedEvent:
        # In the workflow's trace, linked to the delivery's: the command
        # queued here carries this span, so the run it starts or resumes is
        # part of the workflow's trace, and the event that caused it is a link
        # away. A new workflow's trace starts here.
        with tracing.workflow_span(
            "event.route",
            workflow_id=command.workflow_id,
            correlation_id=command.correlation_id,
            thread_id=command.thread_id,
            actor_id=command.actor_id,
            root=kind == RouteKind.STARTED,
            attributes={
                Attr.EVENT_TYPE: envelope.type,
                Attr.EVENT_ID: envelope.event_id,
                Attr.COMMAND_KIND: command.kind,
                Attr.COMMAND_ID: command.command_id,
            },
        ) as span:
            routed = self._enqueue_in_span(command, envelope, kind)
            tracing.set_attributes(span, {Attr.EVENT_OUTCOME: routed.kind})
            return routed

    def _enqueue_in_span(
        self, command: WorkflowCommand, envelope: EventEnvelope, kind: RouteKind
    ) -> RoutedEvent:
        try:
            self.commands.enqueue(command, dedupe=_command_dedupe(envelope))
        except DuplicateWorkflowCommand:
            logger.info(
                "event %s was already routed to workflow %s",
                envelope.dedupe_key,
                command.workflow_id,
            )
            kind = RouteKind.ALREADY_ROUTED
        else:
            logger.info(
                "event %s (%s) %s thread '%s' of workflow %s",
                envelope.dedupe_key,
                envelope.type,
                kind.value,
                command.thread_id,
                command.workflow_id,
            )
        return RoutedEvent(kind=kind, workflow_id=command.workflow_id, thread_id=command.thread_id)


#: `consumed_events.consumer` for `ApplicationTransitionHandler`.
APPLICATION_TRANSITION_CONSUMER = "application_transitions"


class ApplicationTransitionHandler:
    """Applies `application.transition_requested` events, once each.

    The transition, its `event_log` row and the inbox row commit together
    (`IdempotentConsumer`), so a redelivered request moves nothing. A move the
    lifecycle refuses is consumed rather than retried: the same request would
    be refused again on every delivery. A conflicting concurrent move is not --
    it raises, and the event comes back to be decided against the new status.
    """

    def __init__(self, consumer: IdempotentConsumer):
        """Initialize with the consumer the transitions are recorded under."""
        self.consumer = consumer

    async def __call__(self, envelope: EventEnvelope) -> Consumption:
        """Apply one requested transition, unless this event was already applied."""
        application_id, status, reason = self._request(envelope)

        def effect(session: Session, event: EventEnvelope) -> ApplicationStatus | None:
            try:
                ApplicationRepository(session).update_status(
                    application_id,
                    status,
                    reason=reason,
                    actor=f"event:{event.source}",
                    commit=False,
                )
            except InvalidApplicationTransition as exc:
                logger.info("not applying %s: %s", event.dedupe_key, exc)
                return None
            return status

        return self.consumer.consume(envelope, effect)

    @staticmethod
    def _request(envelope: EventEnvelope) -> tuple[UUID, ApplicationStatus, str | None]:
        payload = envelope.payload
        try:
            application_id = UUID(
                str(payload.get("application_id") or envelope.subject("application"))
            )
            status = ApplicationStatus(payload["status"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidEventEnvelope(
                f"{envelope.type} event {envelope.event_id} needs an application_id and a "
                f"valid status: {exc}"
            ) from exc
        return application_id, status, payload.get("reason")


__all__ = [
    "APPLICATION_TRANSITION_CONSUMER",
    "ApplicationTransitionHandler",
    "EventRouter",
    "EventStarter",
    "RouteKind",
    "RoutedEvent",
    "WaitLookup",
    "resume_value",
]

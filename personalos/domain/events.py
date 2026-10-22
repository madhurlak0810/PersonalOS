"""The canonical envelope every asynchronously dispatched event travels in.

An event crosses at least two process boundaries between the transaction that
produced it and the consumer that acts on it: it is staged in `outbox_events`
next to the domain mutation, claimed by whichever worker gets there first, and
handed to a handler or a graph. `EventEnvelope` is the one shape all of that
agrees on, so a consumer never has to know which producer -- a recruiter
message, a board move, a sweep -- an event came from in order to read it.

Three rules the envelope holds a producer to:

- **Identity is explicit.** `event_id` names this copy of the event;
  `dedupe_key` names the *fact* it reports, and is the same for every delivery
  of the same fact. Consumers deduplicate on `dedupe_key`, never on `event_id`:
  a producer that retries after a crash mints a second `event_id` for the same
  fact, and the outbox's at-least-once delivery hands out the same `event_id`
  twice.
- **Payloads are normalized and minimal.** Ids, statuses and short labels --
  what a consumer needs to decide, not a copy of the record it is about. A
  consumer that needs more reads it from its own store by `subject_ref`; the
  original source document, if any, is reachable through `raw_ref`, never
  inlined.
- **Payloads are secret-free.** Construction runs the payload through the
  process-wide redactor, so a credential that reached a producer stops here
  rather than in the outbox, the worker's logs and every consumer after it.

Routing reads two fields. `correlation_id` ties an event to the workflow that
is waiting on it: when one is parked on an interrupt under the same
correlation, the event resumes it; otherwise the event starts a run of its own.
`subject_ref` (`"application:<uuid>"`) names what the event is *about*, for the
consumer that acts on it.

Pure values, like the rest of `domain`; nothing here reads or writes.
"""

import json
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from personalos.domain.errors import ValidationFailed
from personalos.domain.redaction import redact

#: Upper bounds matching the `outbox_events` columns the envelope is keyed by
#: (`type`, `dedupe_key`: `String(255)`), enforced where the envelope is built
#: rather than on the insert that would otherwise fail inside the producer's
#: transaction.
MAX_EVENT_TYPE_LENGTH = 255
MAX_DEDUPE_KEY_LENGTH = 255

#: A payload is a pointer and a few labels, not a document. Generous enough for
#: any normalized event; small enough that inlining a message body or a page
#: of HTML fails at the producer instead of riding through the outbox.
MAX_PAYLOAD_BYTES = 16 * 1024

#: The key an envelope is stored under in `outbox_events.payload_json`, so a
#: reader can tell an enveloped row from one written before envelopes existed.
ENVELOPE_KEY = "envelope"


class InvalidEventEnvelope(ValidationFailed, ValueError):
    """An event envelope violated its contract."""


class EventType:
    """Event types defined by the async dispatch path itself.

    Producer-specific types (`application.created`, ...) are defined with
    their producers -- see `personalos.domain.job_search.JobSearchEventType`.
    """

    #: Something outside the lifecycle -- an inbound message, an operator, a
    #: sweep -- asks for an application to move. Payload:
    #: `{"application_id": str, "status": str, "reason": str | None}`.
    APPLICATION_TRANSITION_REQUESTED = "application.transition_requested"


class EventEnvelope(BaseModel):
    """One event, as every producer stages it and every consumer reads it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    #: Dotted wire name, e.g. `application.transition_requested`.
    type: str
    #: What produced the event: `gmail`, `stall_monitor`, `api`, ...
    source: str
    #: When the fact happened in the world (the message was sent, the move
    #: was requested) -- not when this system heard about it.
    occurred_at: datetime
    #: When this system first observed the fact.
    received_at: datetime = Field(default_factory=datetime.utcnow)
    #: What the event is about, as `<kind>:<id>` -- `application:<uuid>`.
    subject_ref: str | None = None
    #: The request or workflow the event belongs to, when it belongs to one.
    correlation_id: UUID | None = None
    #: Identity of the fact the event reports; see the module docstring.
    dedupe_key: str
    payload: dict[str, Any] = Field(default_factory=dict)
    #: Where the source document lives, if anywhere -- `gmail:<message id>`.
    raw_ref: str | None = None

    @field_validator("type", "source", "dedupe_key")
    @classmethod
    def _not_blank(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise InvalidEventEnvelope(f"event {info.field_name} must not be blank")
        return value

    @field_validator("type")
    @classmethod
    def _type_fits(cls, value: str) -> str:
        if len(value) > MAX_EVENT_TYPE_LENGTH:
            raise InvalidEventEnvelope(
                f"event type is {len(value)} characters; the limit is {MAX_EVENT_TYPE_LENGTH}"
            )
        return value

    @field_validator("dedupe_key")
    @classmethod
    def _dedupe_key_fits(cls, value: str) -> str:
        if len(value) > MAX_DEDUPE_KEY_LENGTH:
            raise InvalidEventEnvelope(
                f"dedupe_key is {len(value)} characters; the limit is {MAX_DEDUPE_KEY_LENGTH}"
            )
        return value

    @field_validator("payload")
    @classmethod
    def _normalized_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            # Round-tripping through JSON is the normalization: a payload a
            # consumer in another process reads back is exactly what was
            # checked here, with no datetime or UUID that only survives
            # in-process.
            encoded = json.dumps(redact(value), default=str, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise InvalidEventEnvelope(f"event payload is not JSON-serializable: {exc}") from exc
        if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise InvalidEventEnvelope(
                f"event payload is over {MAX_PAYLOAD_BYTES} bytes; carry a reference "
                "(subject_ref, raw_ref) instead of the document"
            )
        return json.loads(encoded)

    def subject(self, kind: str) -> str | None:
        """The id half of `subject_ref` when it names a `kind`, else `None`."""
        if not self.subject_ref:
            return None
        prefix, separator, identifier = self.subject_ref.partition(":")
        return identifier if separator and prefix == kind and identifier else None


__all__ = [
    "ENVELOPE_KEY",
    "MAX_DEDUPE_KEY_LENGTH",
    "MAX_EVENT_TYPE_LENGTH",
    "MAX_PAYLOAD_BYTES",
    "EventEnvelope",
    "EventType",
    "InvalidEventEnvelope",
]

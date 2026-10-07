"""Durable recruiter events: the row, the transition and the events, or none of them.

`SqlRecruiterEventStore` is what `JobSearchGraph`'s `ApplicationDirectory` and
`RecruiterEventRecorder` ports are bound to. It does two things.

`candidates` reads what is on file about a user's applications for
`personalos.domain.recruiter_events.correlate_application` to match on. It
makes no decision about which one a message belongs to.

`record` writes one classified, correlated event. A single transaction holds
the `communication_events` row, its `commitments`, the application's
transition (through `ApplicationRepository.update_status`, so the lifecycle
still decides whether the move is legal) and the `outbox_events` rows the
event produces. The row's unique `dedupe_key` is what makes a redelivered
message a no-op: the second delivery finds the first one's row and writes
nothing -- no second transition, no second interview-invite event.
"""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.job_search import EmittedEvent
from personalos.domain.models import ApplicationStatus, InvalidApplicationTransition
from personalos.domain.recruiter_events import (
    ApplicationCandidate,
    Commitment,
    CorrelationOutcome,
    RecruiterEvent,
    RecruiterEventRecord,
)
from personalos.persistence.models import (
    ApplicationModel,
    CommitmentModel,
    CommunicationEventModel,
    JobPostingModel,
)
from personalos.persistence.repositories import (
    ApplicationRepository,
    CommunicationEventRepository,
    OutboxEventRepository,
)

logger = logging.getLogger(__name__)

#: `event_log.payload_json["actor"]` for transitions an inbound message caused.
RECRUITER_EVENT_ACTOR = "recruiter_event"


class SqlRecruiterEventStore:
    """Reads correlation candidates and records recruiter events."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        clock: Callable[[], datetime] = datetime.utcnow,
    ):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory
        self.clock = clock

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    async def candidates(self, user_id: UUID) -> Sequence[ApplicationCandidate]:
        """Every application of `user_id`, with the identifiers on file for it."""
        session = self.session_factory()
        try:
            rows = (
                session.query(ApplicationModel, JobPostingModel)
                .join(JobPostingModel, JobPostingModel.id == ApplicationModel.job_posting_id)
                .filter(ApplicationModel.user_id == user_id)
                .order_by(ApplicationModel.created_at, ApplicationModel.id)
                .all()
            )
            threads: dict[UUID, set[str]] = {}
            contacts: dict[UUID, set[str]] = {}
            if rows:
                events = (
                    session.query(CommunicationEventModel)
                    .filter(
                        CommunicationEventModel.application_id.in_(
                            [application.id for application, _posting in rows]
                        )
                    )
                    .all()
                )
                for event in events:
                    metadata = event.metadata_json or {}
                    if metadata.get("thread_id"):
                        threads.setdefault(event.application_id, set()).add(metadata["thread_id"])
                    if metadata.get("from_address"):
                        contacts.setdefault(event.application_id, set()).add(
                            metadata["from_address"]
                        )
            return [
                ApplicationCandidate(
                    application_id=application.id,
                    company=posting.company,
                    title=posting.title,
                    status=ApplicationStatus(application.status),
                    reference=posting.source_job_id,
                    thread_ids=tuple(sorted(threads.get(application.id, ()))),
                    contact_addresses=tuple(sorted(contacts.get(application.id, ()))),
                )
                for application, posting in rows
            ]
        finally:
            session.close()

    def commitments_for(self, application_id: UUID) -> list[Commitment]:
        """Every commitment recorded for an application, soonest deadline first."""
        session = self.session_factory()
        try:
            rows = (
                session.query(CommitmentModel)
                .filter(CommitmentModel.application_id == application_id)
                .order_by(CommitmentModel.due_at.is_(None), CommitmentModel.due_at)
                .all()
            )
            return [
                Commitment(
                    actor=row.actor,
                    action=row.action,
                    due_at=row.due_at,
                    condition=row.condition,
                    confidence=row.confidence,
                    source_message_id=row.source_message_id,
                )
                for row in rows
            ]
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    async def record(
        self,
        event: RecruiterEvent,
        *,
        transition: ApplicationStatus | None = None,
        emit: Sequence[EmittedEvent] = (),
    ) -> RecruiterEventRecord:
        """Store one event against the application it was matched to, exactly once.

        `transition` is a proposal. A move the lifecycle does not allow from
        the application's stored status is reported back as
        `refused_transition` and the event is still recorded; a status is
        never forced to fit a message.

        Raises `ValueError` for an event with no matched application: there is
        no row to link it to, and choosing one here would be the silent guess
        the review path exists to prevent.
        """
        correlation = event.correlation
        if correlation.outcome is not CorrelationOutcome.MATCHED or not correlation.application_id:
            raise ValueError(
                f"message {event.message.provider_message_id} is not matched to an application "
                "and cannot be recorded"
            )
        application_id = correlation.application_id
        dedupe_key = event.dedupe_key

        session = self.session_factory()
        try:
            existing = CommunicationEventRepository(session).get_by_dedupe_key(dedupe_key)
            if existing is not None:
                return self._duplicate(session, existing)
            try:
                record = self._write(session, event, application_id, transition, emit)
                session.commit()
                return record
            except IntegrityError:
                # Lost the insert race on `dedupe_key`: another delivery of the
                # same message committed first, and its transaction is the one
                # that counts. Everything staged here is discarded with it.
                session.rollback()
                existing = CommunicationEventRepository(session).get_by_dedupe_key(dedupe_key)
                if existing is None:
                    raise
                return self._duplicate(session, existing)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _write(
        self,
        session: Session,
        event: RecruiterEvent,
        application_id: UUID,
        transition: ApplicationStatus | None,
        emit: Sequence[EmittedEvent],
    ) -> RecruiterEventRecord:
        applications = ApplicationRepository(session)
        application = applications.get_by_id(application_id)
        if application is None:
            raise ValueError(f"Application {application_id} not found")
        now = self.clock()
        message = event.message

        row = CommunicationEventRepository(session).create(
            application_id=application_id,
            classification=event.classification.value,
            occurred_at=message.received_at,
            provider_message_id=message.provider_message_id[:255],
            dedupe_key=event.dedupe_key,
            metadata_json={
                "subject": message.subject,
                "from_address": message.from_address,
                "thread_id": message.thread_id,
                "summary": event.summary,
                "requires_reply": event.requires_reply,
                "classification_confidence": event.confidence,
                "extraction_source": event.source.value,
                "correlation_confidence": event.correlation.confidence,
                "correlation_signals": [signal.value for signal in event.correlation.signals],
            },
            commit=False,
        )
        for commitment in event.commitments:
            session.add(
                CommitmentModel(
                    communication_event_id=row.id,
                    application_id=application_id,
                    actor=commitment.actor.value,
                    action=commitment.action,
                    due_at=commitment.due_at,
                    condition=commitment.condition,
                    confidence=commitment.confidence,
                    source_message_id=commitment.source_message_id[:255],
                )
            )

        current = ApplicationStatus(application.status)
        target = transition
        refused: ApplicationStatus | None = None
        if target is None or target is current:
            # A message that changes nothing is still activity, and a stalled
            # application that just heard from a recruiter is not stalled.
            target = (
                ApplicationStatus(application.resume_status)
                if current is ApplicationStatus.STALLED and application.resume_status
                else None
            )
        transitioned = False
        if target is not None:
            try:
                applications.update_status(
                    application_id,
                    target,
                    reason=f"{event.classification.value} in message {message.provider_message_id}",
                    actor=RECRUITER_EVENT_ACTOR,
                    now=now,
                    commit=False,
                )
                transitioned = True
            except InvalidApplicationTransition:
                refused = target
                logger.info(
                    "application %s stays '%s': a %s message proposed '%s', which the "
                    "lifecycle does not allow from there",
                    application_id,
                    current.value,
                    event.classification.value,
                    target.value,
                )
        if not transitioned:
            session.query(ApplicationModel).filter(ApplicationModel.id == application_id).update(
                {ApplicationModel.last_activity_at: now}, synchronize_session=False
            )

        outbox = OutboxEventRepository(session)
        for emitted in emit:
            if emitted.dedupe_key and outbox.get_by_dedupe_key(emitted.dedupe_key) is not None:
                continue
            outbox.create(
                type=emitted.type.value,
                payload=emitted.payload,
                dedupe_key=emitted.dedupe_key,
                commit=False,
            )

        session.flush()
        session.refresh(application)
        return RecruiterEventRecord(
            dedupe_key=event.dedupe_key,
            application_id=application_id,
            communication_event_id=row.id,
            created=True,
            status=ApplicationStatus(application.status),
            transitioned=transitioned,
            refused_transition=refused,
        )

    @staticmethod
    def _duplicate(session: Session, existing: CommunicationEventModel) -> RecruiterEventRecord:
        application = ApplicationRepository(session).get_by_id(existing.application_id)
        return RecruiterEventRecord(
            dedupe_key=existing.dedupe_key,
            application_id=existing.application_id,
            communication_event_id=existing.id,
            created=False,
            status=ApplicationStatus(application.status),
        )


__all__ = ["RECRUITER_EVENT_ACTOR", "SqlRecruiterEventStore"]

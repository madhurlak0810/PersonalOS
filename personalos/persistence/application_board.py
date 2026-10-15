"""The application board's reads, and the two events it records itself.

Reads come from the status projection, `application_status_view`, joined to the
application and its posting. An application that has never transitioned has no
projection row yet -- `ApplicationRepository.update_status` writes the first one
-- so its status falls back to the column it was created with, which is the
same DISCOVERED the projection would hold. Everything else on the detail view
is read from the rows that hang off the application: `artifact_versions`,
`communication_events` and their `commitments`, `pending_checkpoints`, and
`event_log`.

Nothing here changes a status. The two writes are appends to `event_log`:

- `record_match` keeps the scored `JobMatch` an application was found with, so
  the board can show its fit score and the evidence behind it long after the
  run that scored it ended.
- `record_transition_request` keeps the fact that a person asked for a
  transition with an external effect, before the run that will carry it out is
  queued. The status change is written later, by the graph, once the effect
  has actually happened.

Sessions come from an injected factory and each call is its own short
transaction, as in `ApplicationLifecycleStore`.
"""

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from personalos.domain.applications import (
    APPLICATION_MATCH_RECORDED_EVENT,
    APPLICATION_TRANSITION_REQUESTED_EVENT,
    BOARD_COLUMN_ORDER,
    ApplicationBoard,
    ApplicationCard,
    ApplicationDetail,
    ArtifactSummary,
    BoardColumn,
    BoardQuery,
    RecordedMatch,
    RecruiterEventSummary,
    RequestedTransition,
    StatusChange,
    manual_transition_targets,
)
from personalos.domain.artifacts import content_sha256
from personalos.domain.job_search import JobMatch
from personalos.domain.models import (
    APPLICATION_AGGREGATE_TYPE,
    APPLICATION_STATUS_CHANGED_EVENT,
    ApplicationStatus,
)
from personalos.domain.recruiter_events import Commitment
from personalos.persistence.models import (
    ApplicationModel,
    ApplicationStatusViewModel,
    ArtifactVersionModel,
    CommitmentModel,
    CommunicationEventModel,
    EventLogModel,
    JobPostingModel,
)
from personalos.persistence.pending_checkpoints import PendingCheckpointStore
from personalos.persistence.repositories import EventLogRepository

#: An application's status as the board reads it: the projection, or the
#: column it was created with until the projection has a row for it.
_STATUS = func.coalesce(ApplicationStatusViewModel.status, ApplicationModel.status)

#: Column position, so a page runs through the board left to right.
_COLUMN_RANK = case(
    {status.value: rank for rank, status in enumerate(BOARD_COLUMN_ORDER)},
    value=_STATUS,
    else_=len(BOARD_COLUMN_ORDER),
)


class ApplicationBoardStore:
    """Answers the board's questions about applications; records matches and requests."""

    def __init__(self, session_factory: Callable[[], Session]):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory
        self._checkpoints = PendingCheckpointStore(session_factory)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def board(self, query: BoardQuery) -> ApplicationBoard:
        """One page of cards, grouped into lifecycle columns.

        Columns are every state, or the states the query asked for, in
        lifecycle order -- empty ones included, so a client can draw the board
        from the response alone. Each column's `total` counts every match, not
        just this page's.
        """
        session = self.session_factory()
        try:
            base = self._filtered(session, query)
            counts = dict(
                base.with_entities(_STATUS, func.count(ApplicationModel.id)).group_by(_STATUS).all()
            )
            rows = (
                base.with_entities(ApplicationModel, JobPostingModel, ApplicationStatusViewModel)
                .order_by(
                    _COLUMN_RANK,
                    ApplicationModel.last_activity_at.desc(),
                    ApplicationModel.id,
                )
                .offset(query.offset)
                .limit(query.limit)
                .all()
            )
        finally:
            session.close()

        cards: dict[ApplicationStatus, list[ApplicationCard]] = {}
        for application, posting, view in rows:
            card = _card(application, posting, view)
            cards.setdefault(card.status, []).append(card)

        total = sum(counts.values())
        statuses = query.states or BOARD_COLUMN_ORDER
        columns = tuple(
            BoardColumn(
                status=status,
                total=counts.get(status.value, 0),
                applications=tuple(cards.get(status, ())),
            )
            for status in BOARD_COLUMN_ORDER
            if status in statuses
        )
        end = query.offset + len(rows)
        return ApplicationBoard(
            columns=columns,
            total=total,
            limit=query.limit,
            offset=query.offset,
            next_offset=end if end < total else None,
        )

    def card(self, application_id: UUID) -> ApplicationCard | None:
        """One application as the board shows it, or `None` if it does not exist."""
        session = self.session_factory()
        try:
            row = self._one(session, application_id)
            return _card(*row) if row is not None else None
        finally:
            session.close()

    def detail(self, application_id: UUID) -> ApplicationDetail | None:
        """Everything on one application, or `None` if it does not exist."""
        session = self.session_factory()
        try:
            row = self._one(session, application_id)
            if row is None:
                return None
            card = _card(*row)
            events = EventLogRepository(session).get_by_aggregate_id(application_id)
            artifacts = (
                session.query(ArtifactVersionModel)
                .filter(ArtifactVersionModel.application_id == application_id)
                .order_by(ArtifactVersionModel.artifact_type, ArtifactVersionModel.version)
                .all()
            )
            recruiter_events = self._recruiter_events(session, application_id)
        finally:
            session.close()

        return ApplicationDetail(
            card=card,
            allowed_next_states=tuple(
                sorted(
                    manual_transition_targets(card.status, resume_status=card.resume_status),
                    key=BOARD_COLUMN_ORDER.index,
                )
            ),
            match=_latest_match(events),
            artifacts=tuple(_artifact(row) for row in artifacts),
            recruiter_events=recruiter_events,
            pending_checkpoints=tuple(self._checkpoints.open_for_application(application_id)),
            history=tuple(
                _status_change(event)
                for event in events
                if event.event_type == APPLICATION_STATUS_CHANGED_EVENT
            ),
        )

    def submission_attachments(self, application_id: UUID) -> list[dict[str, Any]]:
        """The latest version of each tailored document, as submission attachments.

        The same shape `prepare_application_packet` attaches, so a submission
        requested from the board is reviewed against the same fields as one the
        graph prepared: each document's hash is in the action, and a draft
        replaced after the approval fails closed.
        """
        session = self.session_factory()
        try:
            rows = (
                session.query(ArtifactVersionModel)
                .filter(ArtifactVersionModel.application_id == application_id)
                .order_by(ArtifactVersionModel.artifact_type, ArtifactVersionModel.version.desc())
                .all()
            )
        finally:
            session.close()
        latest: dict[str, ArtifactVersionModel] = {}
        for row in rows:
            latest.setdefault(row.artifact_type, row)
        return [
            {
                "name": row.artifact_type,
                "artifact_type": row.artifact_type,
                "sha256": content_sha256(row.content or ""),
                "artifact_version_id": str(row.id),
                "evidence_ids": [
                    item.get("ref") for item in row.evidence or () if isinstance(item, dict)
                ],
            }
            for row in latest.values()
        ]

    def cover_note(self, application_id: UUID) -> str:
        """The latest cover letter's text, or empty if there is none."""
        session = self.session_factory()
        try:
            row = (
                session.query(ArtifactVersionModel)
                .filter(
                    ArtifactVersionModel.application_id == application_id,
                    ArtifactVersionModel.artifact_type == "cover_letter",
                )
                .order_by(ArtifactVersionModel.version.desc())
                .first()
            )
            return (row.content or "") if row is not None else ""
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record_match(
        self,
        application_id: UUID,
        match: JobMatch,
        *,
        recorded_by: str = "system",
        now: datetime | None = None,
    ) -> UUID:
        """Append the scored match for an application; the latest one is what the board shows."""
        return self._append(
            application_id,
            APPLICATION_MATCH_RECORDED_EVENT,
            {"match": match.model_dump(mode="json"), "recorded_by": recorded_by},
            now,
        )

    def record_transition_request(self, request: RequestedTransition) -> UUID:
        """Append the fact that a transition with an external effect was asked for."""
        return self._append(
            request.application_id,
            APPLICATION_TRANSITION_REQUESTED_EVENT,
            request.model_dump(mode="json"),
            request.requested_at,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(
        self,
        application_id: UUID,
        event_type: str,
        payload: dict[str, Any],
        now: datetime | None,
    ) -> UUID:
        session = self.session_factory()
        try:
            if session.get(ApplicationModel, application_id) is None:
                raise ValueError(f"Application {application_id} not found")
            event = EventLogRepository(session).append(
                aggregate_type=APPLICATION_AGGREGATE_TYPE,
                aggregate_id=application_id,
                event_type=event_type,
                payload=payload,
                occurred_at=now,
            )
            return event.id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _joined(session: Session):
        return (
            session.query(ApplicationModel)
            .join(JobPostingModel, JobPostingModel.id == ApplicationModel.job_posting_id)
            .outerjoin(
                ApplicationStatusViewModel,
                ApplicationStatusViewModel.application_id == ApplicationModel.id,
            )
        )

    def _filtered(self, session: Session, query: BoardQuery):
        q = self._joined(session)
        if query.states:
            q = q.filter(_STATUS.in_([status.value for status in query.states]))
        if query.company:
            q = q.filter(
                func.lower(JobPostingModel.company).contains(query.company.strip().lower())
            )
        if query.created_from is not None:
            q = q.filter(ApplicationModel.created_at >= query.created_from)
        if query.created_to is not None:
            q = q.filter(ApplicationModel.created_at <= query.created_to)
        if query.user_id is not None:
            q = q.filter(ApplicationModel.user_id == query.user_id)
        return q

    def _one(self, session: Session, application_id: UUID):
        return (
            self._joined(session)
            .with_entities(ApplicationModel, JobPostingModel, ApplicationStatusViewModel)
            .filter(ApplicationModel.id == application_id)
            .first()
        )

    @staticmethod
    def _recruiter_events(
        session: Session, application_id: UUID
    ) -> tuple[RecruiterEventSummary, ...]:
        events = (
            session.query(CommunicationEventModel)
            .filter(CommunicationEventModel.application_id == application_id)
            .order_by(CommunicationEventModel.occurred_at)
            .all()
        )
        commitments: dict[UUID, list[Commitment]] = {}
        for row in (
            session.query(CommitmentModel)
            .filter(CommitmentModel.application_id == application_id)
            .order_by(CommitmentModel.due_at.is_(None), CommitmentModel.due_at)
            .all()
        ):
            commitments.setdefault(row.communication_event_id, []).append(
                Commitment(
                    actor=row.actor,
                    action=row.action,
                    due_at=row.due_at,
                    condition=row.condition,
                    confidence=row.confidence,
                    source_message_id=row.source_message_id,
                )
            )
        return tuple(
            RecruiterEventSummary(
                communication_event_id=event.id,
                classification=event.classification,
                provider_message_id=event.provider_message_id,
                occurred_at=event.occurred_at,
                commitments=tuple(commitments.get(event.id, ())),
            )
            for event in events
        )


def _card(
    application: ApplicationModel,
    posting: JobPostingModel,
    view: ApplicationStatusViewModel | None,
) -> ApplicationCard:
    return ApplicationCard(
        application_id=application.id,
        user_id=application.user_id,
        job_posting_id=application.job_posting_id,
        company=posting.company,
        title=posting.title,
        location=posting.location,
        url=posting.url,
        status=ApplicationStatus(view.status if view is not None else application.status),
        resume_status=(
            ApplicationStatus(application.resume_status) if application.resume_status else None
        ),
        applied_at=application.applied_at,
        last_activity_at=application.last_activity_at,
        created_at=application.created_at,
        status_updated_at=view.updated_at if view is not None else application.created_at,
        last_event_id=view.last_event_id if view is not None else None,
    )


def _latest_match(events: Sequence[EventLogModel]) -> RecordedMatch | None:
    for event in reversed(events):
        if event.event_type == APPLICATION_MATCH_RECORDED_EVENT:
            payload = event.payload_json or {}
            return RecordedMatch(
                match=JobMatch.model_validate(payload["match"]),
                recorded_at=event.occurred_at,
                recorded_by=payload.get("recorded_by"),
            )
    return None


def _artifact(row: ArtifactVersionModel) -> ArtifactSummary:
    return ArtifactSummary(
        artifact_version_id=row.id,
        artifact_type=row.artifact_type,
        version=row.version,
        doc_id=row.doc_id,
        generated_by=row.generated_by,
        evidence=tuple(row.evidence or ()),
        created_at=row.created_at,
    )


def _status_change(event: EventLogModel) -> StatusChange:
    payload = event.payload_json or {}
    previous = payload.get("from")
    return StatusChange(
        event_id=event.id,
        from_status=ApplicationStatus(previous) if previous else None,
        to_status=ApplicationStatus(payload["to"]),
        actor=payload.get("actor"),
        reason=payload.get("reason"),
        occurred_at=event.occurred_at,
    )


__all__ = ["ApplicationBoardStore"]

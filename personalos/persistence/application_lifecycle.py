"""The durable application lifecycle: transitions, activity, and the stall sweep's queries.

`personalos.domain.models` says which moves between application statuses are
legal; `ApplicationRepository.update_status` is the one statement that makes
one, appending the `event_log` row and moving the status projection in the
same transaction. This module is that path with its own sessions, for the
callers that are not already inside a transaction -- a graph node acting on a
recommendation, an inbound-message handler reporting activity, and
`apps.worker.stall_monitor`, which runs long after any conversation ended.

Nothing here reads a chat transcript or graph state. An application's status,
what it was held from and when it was last active are columns; how it got
there is `event_log`. A process that starts with an empty memory and this
database knows everything there is to know.

Sessions come from an injected factory and each call is its own short
transaction, matching `PendingCheckpointStore`: the guarded `UPDATE` behind
every transition only excludes a second writer if the first one's has been
committed by the time the second looks.
"""

import logging
from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.models import (
    STALL_MONITOR_ACTOR,
    STALLABLE_APPLICATION_STATUSES,
    ApplicationLifecycleState,
    ApplicationStatus,
    ApplicationTransitionConflict,
    ApplicationTransitionRecommendation,
    InvalidApplicationTransition,
    parse_recommended_status,
)
from personalos.persistence.models import ApplicationModel, ApplicationStatusViewModel
from personalos.persistence.repositories import ApplicationRepository, EventLogRepository

logger = logging.getLogger(__name__)

#: How many applications one stall sweep takes at a time; see
#: `personalos.persistence.pending_checkpoints.DEFAULT_SWEEP_LIMIT`.
DEFAULT_STALL_SWEEP_LIMIT = 100


class ApplicationLifecycleStore:
    """Moves applications along their lifecycle and answers where they are.

    `clock` is injected so the stall window can be tested by moving time
    rather than by waiting for it.
    """

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
    # Writing
    # ------------------------------------------------------------------

    def transition(
        self,
        application_id: UUID,
        new_status: ApplicationStatus,
        *,
        reason: str | None = None,
        actor: str = "system",
        now: datetime | None = None,
    ) -> ApplicationLifecycleState:
        """Move an application to `new_status`; see `ApplicationRepository.update_status`."""
        session = self.session_factory()
        try:
            row = ApplicationRepository(session).update_status(
                application_id,
                new_status,
                reason=reason,
                actor=actor,
                now=now or self.clock(),
            )
            return _to_state(session, row)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def recommend(
        self,
        recommendation: ApplicationTransitionRecommendation,
        *,
        actor: str = "llm",
        now: datetime | None = None,
    ) -> ApplicationLifecycleState:
        """Apply a recommended transition if, and only if, the lifecycle allows it.

        The entry point for anything model-driven. The recommendation's status
        is text until `parse_recommended_status` accepts it, and the move is
        then validated against the status stored in the database -- not one
        the caller claims the application has. A refused recommendation raises
        `InvalidApplicationTransition` and changes nothing.
        """
        return self.transition(
            recommendation.application_id,
            parse_recommended_status(recommendation),
            reason=recommendation.reason,
            actor=actor,
            now=now,
        )

    def record_activity(
        self,
        application_id: UUID,
        *,
        reason: str | None = None,
        now: datetime | None = None,
    ) -> ApplicationLifecycleState:
        """Note that something happened on an application, restarting its stall window.

        For activity that is not itself a transition: a recruiter message that
        changed nothing, a draft revised, a note added. A STALLED application
        is resumed to the status it stalled from, since "stalled" and "just
        active" cannot both be true.
        """
        now = now or self.clock()
        # Two passes at most: the bump is guarded on the status that was read,
        # so losing a race against the stall sweep is noticed, and the second
        # pass sees STALLED and resumes it.
        for _ in range(2):
            state = self.get(application_id)
            if state is None:
                raise ValueError(f"Application {application_id} not found")
            if state.status is ApplicationStatus.STALLED and state.resume_status is not None:
                try:
                    return self.transition(
                        application_id,
                        state.resume_status,
                        reason=reason or "activity resumed",
                        now=now,
                    )
                except ApplicationTransitionConflict:
                    continue

            session = self.session_factory()
            try:
                bumped = (
                    session.query(ApplicationModel)
                    .filter(
                        ApplicationModel.id == application_id,
                        ApplicationModel.status == state.status.value,
                    )
                    .update({ApplicationModel.last_activity_at: now}, synchronize_session=False)
                )
                session.commit()
            finally:
                session.close()
            if bumped:
                return state.model_copy(update={"last_activity_at": now})
        raise ApplicationTransitionConflict(
            f"application {application_id} kept changing while recording activity on it"
        )

    def mark_stalled(
        self,
        application_id: UUID,
        *,
        quiet_since: datetime,
        now: datetime | None = None,
    ) -> bool:
        """Move one application to STALLED if it is still quiet. Returns whether it moved.

        `False` is not an error. It means the application was active, moved or
        was stalled by another sweep between being selected and being written
        -- in every case there is nothing left for this sweep to do, and no
        event was emitted.
        """
        session = self.session_factory()
        try:
            ApplicationRepository(session).update_status(
                application_id,
                ApplicationStatus.STALLED,
                reason=f"no activity since {quiet_since.isoformat()}",
                actor=STALL_MONITOR_ACTOR,
                now=now or self.clock(),
                quiet_since=quiet_since,
            )
        except (ApplicationTransitionConflict, InvalidApplicationTransition):
            session.rollback()
            return False
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        return True

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get(self, application_id: UUID) -> ApplicationLifecycleState | None:
        """Where one application is, or `None` if it does not exist."""
        session = self.session_factory()
        try:
            row = ApplicationRepository(session).get_by_id(application_id)
            return _to_state(session, row) if row is not None else None
        finally:
            session.close()

    def history(self, application_id: UUID) -> list[dict]:
        """Every lifecycle event recorded for an application, oldest first."""
        session = self.session_factory()
        try:
            return [
                event.to_dict()
                for event in EventLogRepository(session).get_by_aggregate_id(application_id)
            ]
        finally:
            session.close()

    def quiet_since(
        self,
        cutoff: datetime,
        *,
        limit: int = DEFAULT_STALL_SWEEP_LIMIT,
    ) -> list[UUID]:
        """Applications still in play whose last activity is at or before `cutoff`.

        Quietest first, so a backlog larger than `limit` is worked through in
        the order it went quiet. Already-STALLED applications are not
        stallable and so never come back: once is all a stall happens.
        """
        session = self.session_factory()
        try:
            rows = (
                session.query(ApplicationModel.id)
                .filter(
                    ApplicationModel.status.in_(
                        sorted(status.value for status in STALLABLE_APPLICATION_STATUSES)
                    ),
                    ApplicationModel.last_activity_at <= cutoff,
                )
                .order_by(ApplicationModel.last_activity_at.asc(), ApplicationModel.id.asc())
                .limit(limit)
                .all()
            )
            return [row.id for row in rows]
        finally:
            session.close()


def _to_state(session: Session, row: ApplicationModel) -> ApplicationLifecycleState:
    view = (
        session.query(ApplicationStatusViewModel)
        .filter(ApplicationStatusViewModel.application_id == row.id)
        .first()
    )
    return ApplicationLifecycleState(
        application_id=row.id,
        status=ApplicationStatus(row.status),
        resume_status=ApplicationStatus(row.resume_status) if row.resume_status else None,
        last_activity_at=row.last_activity_at or row.updated_at,
        last_event_id=view.last_event_id if view is not None else None,
    )


__all__ = [
    "DEFAULT_STALL_SWEEP_LIMIT",
    "ApplicationLifecycleStore",
]

"""Answers a stored `CheckpointCondition` from what the database now holds.

`apps.worker.checkpoint_monitor` re-asks a wait's condition when the wait
comes due. This is the evaluator that makes the answer true: it reads the rows
other parts of the system wrote in the meantime -- the recruiter events
`SqlRecruiterEventStore` recorded, the application's current status, the
executions `ToolExecutor` completed -- and never anything captured when the
wait was scheduled.
"""

from collections.abc import Callable

from sqlalchemy.orm import Session

from personalos.domain.checkpoints import CheckpointCondition, ConditionKind
from personalos.domain.models import (
    TERMINAL_APPLICATION_STATUSES,
    ApplicationStatus,
    CommunicationEventClassification,
    ToolExecutionStatus,
)
from personalos.persistence.models import (
    ApplicationModel,
    CommunicationEventModel,
    ToolExecutionModel,
)


class SqlCheckpointConditionEvaluator:
    """The `CheckpointConditionEvaluator` a deployment binds to its monitor.

    Raises for a condition kind it has no query for, as the port requires of
    an evaluator that cannot tell: answering `True` would cancel a follow-up
    nobody decided to cancel.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        """Take the factory each question opens its own session from."""
        self.session_factory = session_factory

    async def is_met(self, condition: CheckpointCondition) -> bool:
        """Return whether the condition holds as of now."""
        session = self.session_factory()
        try:
            if condition.kind is ConditionKind.RECRUITER_RESPONSE_RECEIVED:
                return self._recruiter_responded(session, condition)
            if condition.kind is ConditionKind.CANDIDATE_REPLY_SENT:
                return self._candidate_replied(session, condition)
            if condition.kind is ConditionKind.APPLICATION_CLOSED:
                return self._application_closed(session, condition)
        finally:
            session.close()
        raise NotImplementedError(f"no evaluator for condition kind '{condition.kind.value}'")

    @staticmethod
    def _recruiter_responded(session: Session, condition: CheckpointCondition) -> bool:
        query = session.query(CommunicationEventModel.id).filter(
            CommunicationEventModel.application_id == condition.subject_id,
            CommunicationEventModel.classification
            != CommunicationEventClassification.UNRELATED.value,
        )
        if condition.since is not None:
            query = query.filter(CommunicationEventModel.occurred_at >= condition.since)
        return query.first() is not None

    @staticmethod
    def _candidate_replied(session: Session, condition: CheckpointCondition) -> bool:
        # A reply is an executed `SEND_RECRUITER_MESSAGE`, and the graph keys
        # every one of those `reply-<application_id>-<message id>`.
        query = session.query(ToolExecutionModel.operation_id).filter(
            ToolExecutionModel.idempotency_key.like(f"reply-{condition.subject_id}-%"),
            ToolExecutionModel.status == ToolExecutionStatus.COMPLETED.value,
        )
        if condition.since is not None:
            query = query.filter(ToolExecutionModel.updated_at >= condition.since)
        return query.first() is not None

    @staticmethod
    def _application_closed(session: Session, condition: CheckpointCondition) -> bool:
        row = (
            session.query(ApplicationModel.status)
            .filter(ApplicationModel.id == condition.subject_id)
            .first()
        )
        # An application that is not there has nothing left to wait for.
        return row is None or ApplicationStatus(row[0]) in TERMINAL_APPLICATION_STATUSES


__all__ = ["SqlCheckpointConditionEvaluator"]

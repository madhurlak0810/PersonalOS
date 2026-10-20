"""Which integrations can be used right now, and which need reconnecting.

Written when a provider refuses in a way retrying cannot change (a
`ProviderPermanentFailure`: revoked OAuth, a missing scope), read when a
workflow failure is shown, so the failure can offer a reconnect instead of a
retry that would only fail again. Marking healthy again is what a reconnect
does; until one exists, nothing here clears a degraded row on its own.

Stores what it is told; it decides nothing.
"""

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.failures import IntegrationHealth
from personalos.domain.redaction import redact_text
from personalos.persistence.models import IntegrationHealthModel, WorkflowModel

DEGRADED = "degraded"
HEALTHY = "healthy"


class IntegrationHealthStore:
    """Reads and writes `integration_health`, one short session per call."""

    def __init__(self, session_factory: Callable[[], Session]):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory

    def get(self, provider: str) -> IntegrationHealth | None:
        """The provider's recorded health, or `None` if nothing was ever recorded."""
        session = self.session_factory()
        try:
            row = session.get(IntegrationHealthModel, provider)
            return _health(row) if row is not None else None
        finally:
            session.close()

    def mark_degraded(
        self,
        provider: str,
        *,
        reason: str | None,
        detail: str | None = None,
        workflow_id: UUID | None = None,
    ) -> IntegrationHealth:
        """Record that `provider` needs reconnecting before it can be used again."""
        now = datetime.utcnow()
        session = self.session_factory()
        try:
            row = session.get(IntegrationHealthModel, provider)
            if row is None:
                row = IntegrationHealthModel(provider=provider)
                session.add(row)
            if workflow_id is not None and session.get(WorkflowModel, workflow_id) is None:
                workflow_id = None
            if row.status != DEGRADED:
                row.degraded_at = now
            row.status = DEGRADED
            row.reason = reason
            row.detail = redact_text(detail) if detail else None
            row.workflow_id = workflow_id
            row.updated_at = now
            session.commit()
            return _health(row)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def mark_healthy(self, provider: str) -> IntegrationHealth:
        """Record that `provider` was reconnected."""
        session = self.session_factory()
        try:
            row = session.get(IntegrationHealthModel, provider)
            if row is None:
                row = IntegrationHealthModel(provider=provider)
                session.add(row)
            row.status = HEALTHY
            row.reason = None
            row.detail = None
            row.degraded_at = None
            row.updated_at = datetime.utcnow()
            session.commit()
            return _health(row)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def _health(row: IntegrationHealthModel) -> IntegrationHealth:
    return IntegrationHealth(
        provider=row.provider,
        status=row.status,
        reason=row.reason,
        reconnect_required=row.status == DEGRADED,
        updated_at=row.updated_at,
    )


__all__ = ["DEGRADED", "HEALTHY", "IntegrationHealthStore"]

"""Which integrations can be used right now, and which need reconnecting.

Written from three places:

- the worker's scheduled health checks (`apps.worker.health_monitor`), with
  every outcome, good or bad (`record`);
- any path that saw a real call fail permanently -- a workflow run, an event
  delivery, the Gmail poll -- with the refusal (`mark_degraded`);
- a reconnect (`GoogleConnector.complete`, `mark_healthy`) and a user's
  disconnect (`mark_degraded`).

Read when a workflow failure is shown, so the failure can offer a reconnect
instead of a retry that would only fail again; by `GET /v1/integrations` and
`GET /v1/failures/integrations`; and by the worker's circuit breakers
(`blocked`), which hold back work that needs an integration that is down.

The rules, in one place:

- A **permanent** failure (revoked OAuth, a missing scope) makes an
  integration `degraded` with `reconnect_required`. A revoked or disconnected
  Google credential does so for every Google integration at once
  (`personalos.domain.health.affected_integrations`).
- A **transient** failure makes it `degraded`, and `DOWN_AFTER_FAILURES` of
  them in a row make it `down`. A transient failure never clears
  `reconnect_required`: a probe that times out says nothing about whether the
  credential came back.
- A **success** makes it `healthy` and resets the streak.

Every change of state is counted and logged once, through
`personalos.observability.health`; a failure repeated in the same state is
not a new transition.

Stores what it is told; it decides nothing beyond those rules.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from personalos.domain.failures import IntegrationHealth
from personalos.domain.health import (
    DOWN_AFTER_FAILURES,
    HealthAction,
    HealthCheckResult,
    HealthStatus,
    affected_integrations,
    is_blocking,
    recovered_integrations,
    sanitize_health_error,
)
from personalos.observability import health as telemetry
from personalos.persistence.models import IntegrationHealthModel, WorkflowModel

DEGRADED = HealthStatus.DEGRADED.value
HEALTHY = HealthStatus.HEALTHY.value
DOWN = HealthStatus.DOWN.value


@dataclass(frozen=True)
class HealthChange:
    """One integration's state before and after a recorded outcome."""

    health: IntegrationHealth
    previous: str | None

    @property
    def changed(self) -> bool:
        """Whether this outcome moved the integration to a different state."""
        return self.previous != self.health.status


class IntegrationHealthStore:
    """Reads and writes `integration_health`, one short session per call."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        clock: Callable[[], datetime] = datetime.utcnow,
        down_after: int = DOWN_AFTER_FAILURES,
    ):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory
        self.clock = clock
        self.down_after = down_after

    # --- Reads ----------------------------------------------------------------

    def get(self, provider: str) -> IntegrationHealth | None:
        """The integration's recorded health, or `None` if nothing was ever recorded."""
        session = self.session_factory()
        try:
            row = session.get(IntegrationHealthModel, provider)
            return _health(row) if row is not None else None
        finally:
            session.close()

    def all(self) -> list[IntegrationHealth]:
        """Every recorded integration, by name."""
        session = self.session_factory()
        try:
            rows = session.query(IntegrationHealthModel).order_by(IntegrationHealthModel.provider)
            return [_health(row) for row in rows]
        finally:
            session.close()

    def unhealthy(self) -> list[IntegrationHealth]:
        """Every integration that is degraded or down, by name."""
        return [health for health in self.all() if health.status != HEALTHY]

    def blocked(self, integrations: Iterable[str]) -> list[IntegrationHealth]:
        """Those of `integrations` whose circuit is open: down, or awaiting a reconnect."""
        wanted = set(integrations)
        if not wanted:
            return []
        session = self.session_factory()
        try:
            rows = (
                session.query(IntegrationHealthModel)
                .filter(IntegrationHealthModel.provider.in_(wanted))
                .order_by(IntegrationHealthModel.provider)
            )
            return [
                _health(row)
                for row in rows
                if is_blocking(row.status, bool(row.reconnect_required))
            ]
        finally:
            session.close()

    # --- Writes ---------------------------------------------------------------

    def record(
        self,
        result: HealthCheckResult,
        *,
        workflow_id: UUID | None = None,
        source: str = "check",
    ) -> list[HealthChange]:
        """Record one check's (or call's) outcome. Returns each integration it changed.

        A permanent failure is recorded against every integration it breaks.
        """
        if result.ok:
            return [self._write(result.integration, result, workflow_id=None, source=source)]
        if result.permanent:
            names = affected_integrations(result.provider or result.integration, result.reason)
            if result.integration not in names:
                names = (result.integration, *names)
        else:
            names = (result.integration,)
        return [self._write(name, result, workflow_id=workflow_id, source=source) for name in names]

    def mark_degraded(
        self,
        provider: str,
        *,
        reason: str | None,
        detail: str | None = None,
        workflow_id: UUID | None = None,
    ) -> IntegrationHealth:
        """Record that `provider` needs reconnecting before it can be used again.

        Every integration sharing a revoked credential is marked with it.
        Returns `provider`'s own health.
        """
        result = HealthCheckResult(
            integration=provider,
            status=HealthStatus.DEGRADED,
            reason=reason,
            error=detail,
            permanent=True,
            provider=provider,
        )
        changes = self.record(result, workflow_id=workflow_id, source="call")
        return next(c.health for c in changes if c.health.provider == provider)

    def mark_healthy(self, provider: str) -> IntegrationHealth:
        """Record that `provider` was reconnected, with everything that shares it.

        An integration that shares it but has no row yet is left without one:
        it has never been checked, and a reconnect is no check of it.
        """
        own = None
        recorded = {health.provider for health in self.all()}
        for name in recovered_integrations(provider):
            if name != provider and name not in recorded:
                continue
            change = self._write(
                name, HealthCheckResult.healthy(name), workflow_id=None, source="reconnect"
            )
            if name == provider:
                own = change.health
        return own

    # --- Internals ------------------------------------------------------------

    def _write(
        self,
        name: str,
        result: HealthCheckResult,
        *,
        workflow_id: UUID | None,
        source: str,
    ) -> HealthChange:
        now = result.checked_at or self.clock()
        session = self.session_factory()
        try:
            row = session.get(IntegrationHealthModel, name)
            if row is None:
                row = IntegrationHealthModel(
                    provider=name, status=HEALTHY, reconnect_required=False, consecutive_failures=0
                )
                session.add(row)
                previous = None
            else:
                previous = row.status
            if workflow_id is not None and session.get(WorkflowModel, workflow_id) is None:
                workflow_id = None

            self._apply(row, result, now=now, workflow_id=workflow_id)
            row.updated_at = now
            session.commit()
            health = _health(row)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        try:
            telemetry.record_outcome(
                name,
                previous=previous,
                status=health.status,
                reason=health.reason,
                action=health.action,
                error=health.last_error,
                source=source,
                workflow_id=workflow_id,
            )
        except Exception:  # pragma: no cover - telemetry must never fail a write
            pass
        return HealthChange(health=health, previous=previous)

    def _apply(
        self,
        row: IntegrationHealthModel,
        result: HealthCheckResult,
        *,
        now: datetime,
        workflow_id: UUID | None,
    ) -> None:
        """Move one row to the state `result` puts it in, by the rules above."""
        row.last_checked_at = now
        if result.ok:
            row.status = HEALTHY
            row.reason = None
            row.detail = None
            row.reconnect_required = False
            row.consecutive_failures = 0
            row.degraded_at = None
            row.workflow_id = None
            row.last_success_at = now
            return
        row.consecutive_failures = (row.consecutive_failures or 0) + 1
        if row.status == HEALTHY:
            row.degraded_at = now
        if result.permanent:
            row.status = DEGRADED
            row.reason = result.reason
            row.detail = sanitize_health_error(result.error)
            row.reconnect_required = True
            row.workflow_id = workflow_id
            return
        if row.reconnect_required:
            # Keeps its reason and stays degraded: only a success or a
            # reconnect says the credential came back.
            return
        row.status = DOWN if row.consecutive_failures >= self.down_after else DEGRADED
        row.reason = result.reason
        row.detail = sanitize_health_error(result.error)
        if workflow_id is not None:
            row.workflow_id = workflow_id


def _action(row: IntegrationHealthModel) -> str | None:
    if row.status == HEALTHY:
        return None
    return (HealthAction.RECONNECT if row.reconnect_required else HealthAction.RETRY).value


def _health(row: IntegrationHealthModel) -> IntegrationHealth:
    return IntegrationHealth(
        provider=row.provider,
        status=row.status,
        reason=row.reason,
        reconnect_required=bool(row.reconnect_required),
        action=_action(row),
        last_error=row.detail,
        last_success_at=row.last_success_at,
        last_checked_at=row.last_checked_at,
        consecutive_failures=row.consecutive_failures or 0,
        degraded_at=row.degraded_at,
        workflow_id=row.workflow_id,
        updated_at=row.updated_at,
    )


__all__ = ["DEGRADED", "DOWN", "HEALTHY", "HealthChange", "IntegrationHealthStore"]

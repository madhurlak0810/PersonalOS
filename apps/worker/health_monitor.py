"""Integration health checks on a schedule, and the circuit breakers that read them.

**The monitor.** `IntegrationHealthMonitor.sweep` runs every adapter's
`health_check()` and records each outcome in `integration_health`. A check
that succeeds makes its integration healthy again, so an integration that was
down recovers on its own as soon as its provider answers. An integration
awaiting a reconnect is *not* probed: its credential was refused permanently,
asking again on a timer would fail the same way, and a reconnect marks it
healthy without help (`GoogleConnector.complete`). That is what keeps a
revoked token from turning into a retry loop.

**The circuit breakers.** `HealthGate` says which integrations a piece of work
needs and whether any of them is blocked (down, or awaiting a reconnect):

- a queued `START` command is held back, not run, while what its input needs
  is blocked -- a classification of inbound mail needs Gmail and the model, a
  calendar re-plan needs the calendar -- and released again after
  `retry_after`, so it waits in the queue instead of failing;
- a due follow-up checkpoint is left pending while Gmail is blocked, and fires
  on the first sweep after Gmail recovers;
- the Gmail poll is skipped while Gmail is blocked (`GatedSweep`).

Work that needs none of them is not held: job discovery and application
preparation go on with Gmail down, and so does deterministic event ingestion
with the model down.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import httpx

from personalos.domain.errors import RetryableFailure
from personalos.domain.failures import IntegrationHealth
from personalos.domain.health import (
    GMAIL,
    GOOGLE_CALENDAR,
    HealthCheck,
    HealthCheckResult,
    HealthStatus,
)
from personalos.domain.workflow import WorkflowCommand, WorkflowCommandKind
from personalos.persistence.integration_health import IntegrationHealthStore

logger = logging.getLogger(__name__)

#: Seconds one check may take before it counts as a failure.
DEFAULT_CHECK_TIMEOUT_SECONDS = 30.0
#: How long held-back work waits before it is looked at again.
DEFAULT_CIRCUIT_RETRY = timedelta(seconds=60)
#: Seconds an HTTP probe of a local server waits for an answer.
DEFAULT_HTTP_PROBE_TIMEOUT = 5.0


async def http_get_ok(url: str, *, timeout: float = DEFAULT_HTTP_PROBE_TIMEOUT) -> None:
    """GET `url`; raise `RetryableFailure` unless it answers with a 2xx.

    What `personalos.models.health.LocalModelServerCheck` probes a local model
    server with. Here because the models layer may hold no HTTP client.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
    if response.status_code >= 400:
        raise RetryableFailure(f"{url} answered HTTP {response.status_code}")


@dataclass(frozen=True)
class HealthSweepReport:
    """What one sweep found, by integration."""

    results: Mapping[str, str] = field(default_factory=dict)
    #: Integrations not probed because they await a reconnect.
    skipped: tuple[str, ...] = ()


class IntegrationHealthMonitor:
    """Runs each `HealthCheck` and records what it found."""

    def __init__(
        self,
        checks: Sequence[HealthCheck],
        store: IntegrationHealthStore,
        *,
        timeout_seconds: float = DEFAULT_CHECK_TIMEOUT_SECONDS,
    ):
        """Take the checks (one per integration; later duplicates are dropped) and the store."""
        unique: dict[str, HealthCheck] = {}
        for check in checks:
            unique.setdefault(check.integration, check)
        self.checks = tuple(unique.values())
        self.store = store
        self.timeout_seconds = timeout_seconds

    async def sweep(self) -> HealthSweepReport:
        """Check every integration not awaiting a reconnect; record each outcome."""
        results: dict[str, str] = {}
        skipped: list[str] = []
        for check in self.checks:
            current = self.store.get(check.integration)
            if current is not None and current.reconnect_required:
                skipped.append(check.integration)
                continue
            result = await self._run(check)
            changes = self.store.record(result, source="check")
            results[check.integration] = changes[0].health.status if changes else result.status
        return HealthSweepReport(results=results, skipped=tuple(skipped))

    async def _run(self, check: HealthCheck) -> HealthCheckResult:
        try:
            return await asyncio.wait_for(check.health_check(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            return HealthCheckResult.from_exception(
                check.integration,
                TimeoutError(f"no answer within {self.timeout_seconds:g} seconds"),
            )
        except Exception as error:  # noqa: BLE001 - a check that raises is a failed check
            return HealthCheckResult.from_exception(check.integration, error)


class HealthGate:
    """Which integrations work needs, and whether they are available. The circuit breaker."""

    def __init__(
        self,
        store: IntegrationHealthStore,
        *,
        model_integrations: Collection[str] = (),
        retry_after: timedelta = DEFAULT_CIRCUIT_RETRY,
    ):
        """Take the store, the model integrations model-backed steps run on, and the hold time."""
        self.store = store
        self.model_integrations = tuple(model_integrations)
        self.retry_after = retry_after

    def requirements(self, graph_input: Mapping[str, Any] | None) -> set[str]:
        """The integrations a run started with `graph_input` cannot do without.

        Read from the same keys `JobSearchGraph.route_from_start` routes on. A
        run that enters at discovery needs none of them: it searches job
        boards and drafts from stored evidence.
        """
        graph_input = graph_input or {}
        needed: set[str] = set()
        if graph_input.get("fired_checkpoints"):
            needed.add(GMAIL)
        if graph_input.get("inbound_message_refs") or graph_input.get("inbound_messages"):
            if graph_input.get("inbound_message_refs"):
                needed.add(GMAIL)
            needed.update(self.model_integrations)
        if graph_input.get("calendar_changes"):
            needed.add(GOOGLE_CALENDAR)
        return needed

    def blocked(self, integrations: Iterable[str]) -> list[IntegrationHealth]:
        """Those of `integrations` whose circuit is open. Never raises."""
        try:
            return self.store.blocked(integrations)
        except Exception:
            # The health table being unreachable is not a reason to stop
            # every workflow: fail open, and let the step itself fail if the
            # integration really is down.
            logger.exception("could not read integration health; not holding work back")
            return []

    def blocked_for(self, command: WorkflowCommand) -> list[IntegrationHealth]:
        """The blocked integrations a queued command needs; empty when it may run.

        Only a `START` is gated: its input says where the run enters. A resume
        or a retry is a human's decision about a run already under way.
        """
        if command.kind != WorkflowCommandKind.START:
            return []
        return self.blocked(self.requirements(command.graph_input))


def describe_blocked(blocked: Iterable[IntegrationHealth]) -> str:
    """`gmail (down: unavailable), model.ollama (degraded: ...)`, for log lines."""
    return ", ".join(f"{h.provider} ({h.status}: {h.reason or 'no reason'})" for h in blocked)


class GatedSweep:
    """A sweep that runs only while its integration is available, and reports how it went.

    Skipped while the integration is blocked. A permanent refusal is recorded
    against the integration -- once, as a transition; the next run is then
    skipped rather than refused again -- and not raised, since there is
    nothing for the scheduler to retry. A success is recorded too, which is
    what lets a real call close a circuit a check opened.
    """

    def __init__(
        self,
        integration: str,
        run: Callable[[], Awaitable[Any]],
        store: IntegrationHealthStore,
    ):
        """Take the integration, the sweep, and the store."""
        self.integration = integration
        self.inner = run
        self.store = store

    async def __call__(self) -> Any:
        try:
            blocked = self.store.blocked([self.integration])
        except Exception:
            logger.exception("could not read %s health; running the sweep anyway", self.integration)
            blocked = []
        if blocked:
            logger.info("skipping the %s sweep: %s", self.integration, describe_blocked(blocked))
            return None
        try:
            result = await self.inner()
        except Exception as error:
            outcome = HealthCheckResult.from_exception(self.integration, error)
            self.store.record(outcome, source="call")
            if outcome.permanent:
                return None
            raise
        current = self.store.get(self.integration)
        if current is None or current.status != HealthStatus.HEALTHY.value:
            self.store.record(HealthCheckResult.healthy(self.integration), source="call")
        return result


__all__ = [
    "DEFAULT_CHECK_TIMEOUT_SECONDS",
    "DEFAULT_CIRCUIT_RETRY",
    "GatedSweep",
    "HealthGate",
    "HealthSweepReport",
    "IntegrationHealthMonitor",
    "describe_blocked",
    "http_get_ok",
]

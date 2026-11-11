"""Integration health as OpenTelemetry metrics and structured log lines.

Three instruments, all with an `integration` attribute:

- `personalos.integration.health.checks` (counter): every recorded outcome,
  with `status` and `source` (`check` for a scheduled probe, `call` for a
  real call that failed or succeeded).
- `personalos.integration.health.transitions` (counter): every change of
  state, with `from_status` and `status`. Recorded once per change, never per
  repeated failure, so a revoked credential counts one transition however
  often it is seen.
- `personalos.integration.health.status` (observable gauge): the last known
  state per integration in this process, `0` healthy, `1` degraded, `2` down.

Nothing here exports anything: the instruments come from the global meter
provider, which is a no-op until one is installed, the same as tracing.

A transition is logged as `integration.health_changed` (WARNING when it gets
worse, INFO when it recovers) with `integration`, `health_status`,
`previous_health_status`, `health_reason`, `health_action` and, when a workflow
caused it, `workflow_id`. Error text is already redacted by the store.
"""

import logging
import threading
from collections.abc import Iterable
from uuid import UUID

from opentelemetry import metrics

logger = logging.getLogger("personalos.integration_health")

#: `event` of the log line a state change writes.
HEALTH_CHANGED_EVENT = "integration.health_changed"
#: `event` of the log line a recorded failure writes, changed or not.
HEALTH_FAILED_EVENT = "integration.health_failed"

#: Gauge value for each status.
STATUS_VALUES = {"healthy": 0, "degraded": 1, "down": 2}

_meter = metrics.get_meter("personalos.integration_health")
_checks = _meter.create_counter(
    "personalos.integration.health.checks",
    description="Integration health outcomes recorded, by integration and status",
)
_transitions = _meter.create_counter(
    "personalos.integration.health.transitions",
    description="Integration health state changes",
)

_lock = threading.Lock()
_last_status: dict[str, str] = {}


def _observe(_options) -> Iterable[metrics.Observation]:
    with _lock:
        snapshot = dict(_last_status)
    return [
        metrics.Observation(STATUS_VALUES.get(status, 2), {"integration": integration})
        for integration, status in snapshot.items()
    ]


_meter.create_observable_gauge(
    "personalos.integration.health.status",
    callbacks=[_observe],
    description="Last known integration health: 0 healthy, 1 degraded, 2 down",
)


def record_outcome(
    integration: str,
    *,
    previous: str | None,
    status: str,
    reason: str | None = None,
    action: str | None = None,
    error: str | None = None,
    source: str = "check",
    workflow_id: UUID | None = None,
) -> None:
    """Count one outcome, and log and count it as a transition when the state changed."""
    with _lock:
        _last_status[integration] = status
    _checks.add(1, {"integration": integration, "status": status, "source": source})

    extra = {
        "integration": integration,
        "health_status": status,
        "previous_health_status": previous,
        "health_reason": reason,
        "health_action": action,
        "health_source": source,
    }
    if workflow_id is not None:
        extra["workflow_id"] = str(workflow_id)

    if previous == status:
        if status != "healthy":
            logger.debug(
                "integration '%s' is still %s (%s)",
                integration,
                status,
                reason,
                extra={"event": HEALTH_FAILED_EVENT, **extra, "health_error": error},
            )
        return

    _transitions.add(
        1, {"integration": integration, "from_status": previous or "unknown", "status": status}
    )
    if status == "healthy":
        logger.info(
            "integration '%s' is healthy again (was %s)",
            integration,
            previous or "unknown",
            extra={"event": HEALTH_CHANGED_EVENT, **extra},
        )
    else:
        logger.warning(
            "integration '%s' is %s (%s); %s",
            integration,
            status,
            reason or "no reason given",
            "reconnect it" if action == "reconnect" else "it will be checked again",
            extra={"event": HEALTH_CHANGED_EVENT, **extra, "health_error": error},
        )


def last_known_status() -> dict[str, str]:
    """The status this process last recorded per integration. For tests and debugging."""
    with _lock:
        return dict(_last_status)


__all__ = [
    "HEALTH_CHANGED_EVENT",
    "HEALTH_FAILED_EVENT",
    "STATUS_VALUES",
    "last_known_status",
    "record_outcome",
]

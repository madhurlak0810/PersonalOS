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

The names and labels are `metrics.Metric.INTEGRATION_HEALTH_*`, recorded
through `personalos.observability.metrics`, so they are no-ops until a meter
provider is installed. The gauge here reports what *this process* last saw;
the API also reports the stored status of every integration from the
database (`state_metrics`), so the dashboards see it while the worker is down.

A transition is logged as `integration.health_changed` (WARNING when it gets
worse, INFO when it recovers) with `integration`, `health_status`,
`previous_health_status`, `health_reason`, `health_action` and, when a workflow
caused it, `workflow_id`. Error text is already redacted by the store.
"""

import logging
import threading
from collections.abc import Iterable
from uuid import UUID

from personalos.observability import metrics
from personalos.observability.metrics import Metric

logger = logging.getLogger("personalos.integration_health")

#: `event` of the log line a state change writes.
HEALTH_CHANGED_EVENT = "integration.health_changed"
#: `event` of the log line a recorded failure writes, changed or not.
HEALTH_FAILED_EVENT = "integration.health_failed"

#: Gauge value for each status.
STATUS_VALUES = {"healthy": 0, "degraded": 1, "down": 2}

_lock = threading.Lock()
_last_status: dict[str, str] = {}


def _observe() -> Iterable[tuple[float, dict[str, str]]]:
    with _lock:
        snapshot = dict(_last_status)
    return [
        (STATUS_VALUES.get(status, 2), {"integration": integration})
        for integration, status in snapshot.items()
    ]


metrics.register_gauge(Metric.INTEGRATION_HEALTH_STATUS, _observe)


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
    metrics.add(
        Metric.INTEGRATION_HEALTH_CHECKS, integration=integration, status=status, source=source
    )

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

    metrics.add(
        Metric.INTEGRATION_HEALTH_TRANSITIONS,
        integration=integration,
        from_status=previous or "unknown",
        status=status,
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

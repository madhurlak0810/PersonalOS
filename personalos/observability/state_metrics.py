"""The operational gauges: the latest `OperationalSnapshot`, published as metrics.

`StateMetricsSampler` takes a snapshot (from a callable the composition root
hands it -- in production `OperationalStateReader.snapshot`) every
`METRICS_STATE_INTERVAL_SECONDS`, on a thread, and keeps the newest. The
gauges it registers read that copy at export time and never touch the
database themselves, so a slow query cannot stall the exporter.

The API runs it (`apps.api.main`), not the worker: queue depth and stalled
workflows matter most when the worker is the thing that is down.

A sample that fails is logged, and the gauges go quiet -- they report nothing
rather than a stale picture -- until the next sample succeeds. Prometheus then
sees the series disappear, which `deploy/prometheus/rules/personalos.yml`
alerts on as `OperationalStateMissing`.

What each gauge reports:

- `personalos.worker.queue.depth{state}` and `personalos.outbox.backlog{state}`
  for `pending` and `in_progress`, zeros included;
- `personalos.outbox.oldest_pending.age`, 0 with nothing pending;
- `personalos.outbox.parked{status}` for `dead_letter` and `quarantined`;
- `personalos.workflows.active`;
- `personalos.workflows.stalled{reason}` for every reason, zeros included, and
  `personalos.workflow.stalled_for{workflow_id, reason}` for each stalled
  workflow only -- the series an alert names the workflow from;
- `personalos.integration.health.status{integration}` for each stored
  integration, beside what the worker reports from memory.
"""

import asyncio
import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from personalos.domain.operational_state import OperationalSnapshot, StallReason
from personalos.observability import metrics
from personalos.observability.health import STATUS_VALUES
from personalos.observability.metrics import Metric

logger = logging.getLogger(__name__)

Sample = Iterable[tuple[float, Mapping[str, Any]]]


class StateMetricsSampler:
    """Samples operational state on an interval and publishes the newest as gauges."""

    def __init__(self, take_snapshot: Callable[[], OperationalSnapshot]):
        """Take the function that reads one snapshot (it may block; it runs on a thread)."""
        self.take_snapshot = take_snapshot
        self._lock = threading.Lock()
        self._latest: OperationalSnapshot | None = None
        self._unregister: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @property
    def latest(self) -> OperationalSnapshot | None:
        """The newest snapshot, or `None` before the first or after a failed sample."""
        with self._lock:
            return self._latest

    def sample(self) -> OperationalSnapshot | None:
        """Take one snapshot now and publish it; `None` (and the gauges cleared) if it failed."""
        try:
            snapshot = self.take_snapshot()
        except Exception:
            logger.exception(
                "could not sample operational state; the gauges are empty until the next sample"
            )
            snapshot = None
        with self._lock:
            self._latest = snapshot
        return snapshot

    async def run(self, stop: asyncio.Event, *, interval_seconds: float) -> None:
        """Sample, wait `interval_seconds`, repeat, until `stop` is set."""
        while not stop.is_set():
            await asyncio.to_thread(self.sample)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------
    # Gauges
    # ------------------------------------------------------------------

    def install(self) -> "StateMetricsSampler":
        """Register this sampler's gauges. Idempotent; `uninstall` removes them."""
        if self._unregister:
            return self
        sources: list[tuple[metrics.MetricSpec, Callable[[], Sample]]] = [
            (Metric.WORKER_QUEUE_DEPTH, self._queue_depth),
            (Metric.OUTBOX_BACKLOG, self._backlog),
            (Metric.OUTBOX_OLDEST_PENDING_AGE, self._oldest),
            (Metric.OUTBOX_PARKED, self._parked),
            (Metric.WORKFLOWS_ACTIVE, self._active),
            (Metric.WORKFLOWS_STALLED, self._stalled_counts),
            (Metric.WORKFLOW_STALLED_FOR, self._stalled_workflows),
            (Metric.INTEGRATION_HEALTH_STATUS, self._integrations),
        ]
        self._unregister = [metrics.register_gauge(spec, source) for spec, source in sources]
        return self

    def uninstall(self) -> None:
        """Remove this sampler's gauges."""
        for unregister in self._unregister:
            unregister()
        self._unregister = []

    def _queue_depth(self) -> Sample:
        snapshot = self.latest
        return [] if snapshot is None else _by("state", snapshot.queue_depth)

    def _backlog(self) -> Sample:
        snapshot = self.latest
        return [] if snapshot is None else _by("state", snapshot.outbox_backlog)

    def _oldest(self) -> Sample:
        snapshot = self.latest
        return [] if snapshot is None else [(snapshot.oldest_pending_age_seconds, {})]

    def _parked(self) -> Sample:
        snapshot = self.latest
        return [] if snapshot is None else _by("status", snapshot.parked)

    def _active(self) -> Sample:
        snapshot = self.latest
        return [] if snapshot is None else [(snapshot.active_workflows, {})]

    def _stalled_counts(self) -> Sample:
        snapshot = self.latest
        if snapshot is None:
            return []
        return [(snapshot.stalled_count(reason), {"reason": reason}) for reason in StallReason]

    def _stalled_workflows(self) -> Sample:
        snapshot = self.latest
        if snapshot is None:
            return []
        return [
            (item.stalled_for_seconds, {"workflow_id": item.workflow_id, "reason": item.reason})
            for item in snapshot.stalled
        ]

    def _integrations(self) -> Sample:
        snapshot = self.latest
        if snapshot is None:
            return []
        return [
            (STATUS_VALUES.get(status, 2), {"integration": integration})
            for integration, status in snapshot.integrations.items()
        ]


def _by(label: str, counts: Mapping[str, int]) -> Sample:
    return [(count, {label: key}) for key, count in counts.items()]


__all__ = ["StateMetricsSampler"]

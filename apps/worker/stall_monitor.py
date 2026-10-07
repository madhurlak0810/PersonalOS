"""Sweeps applications that have gone quiet: the scheduled check behind STALLED.

An application does not notice that nothing has happened to it. Every other
status is entered because something occurred -- a draft was approved, a
recruiter wrote back -- and so has an event to hang the transition on. STALLED
is entered because nothing did, which means something has to come and look.
This module is that something, in the same shape as
`apps.worker.checkpoint_monitor` and for the same reason.

It runs on a schedule and reads `applications.last_activity_at`. It is not
called from a chat turn and never consults a conversation: an application
whose user has not opened the assistant in a month stalls exactly as one whose
user is talking to it right now about something else.

**Exactly once** rests on two things in `ApplicationLifecycleStore`, not on
anything here. An already-STALLED application is not selected, so a second
sweep does nothing; and the write is a guarded `UPDATE` on the status and
`last_activity_at` the sweep saw, so of two monitors running the same minute
one moves the row and the other is told it lost. Only the winner's transition
emits an event.

`now` is taken once per sweep and the cutoff derived from it once, so every
application in a sweep is measured against the same instant.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from personalos.config import settings
from personalos.persistence.application_lifecycle import (
    DEFAULT_STALL_SWEEP_LIMIT,
    ApplicationLifecycleStore,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StallSweepReport:
    """What one stall sweep did."""

    #: Moved to STALLED by this sweep.
    stalled: tuple[UUID, ...] = ()
    #: Selected as quiet, then found active, moved or already stalled by the
    #: time it was written. Not an error.
    skipped: tuple[UUID, ...] = ()


class StalledApplicationMonitor:
    """Moves applications with no activity past the stall window to STALLED."""

    def __init__(
        self,
        *,
        store: ApplicationLifecycleStore,
        window: timedelta | None = None,
        clock: Callable[[], datetime] = datetime.utcnow,
    ):
        """Wire the lifecycle store and the window; `window` defaults to the configured one."""
        if store is None:
            raise ValueError("StalledApplicationMonitor requires an ApplicationLifecycleStore")
        window = (
            window
            if window is not None
            else timedelta(days=settings.application_stall_window_days)
        )
        if window <= timedelta(0):
            raise ValueError("the stall window must be positive")
        self.store = store
        self.window = window
        self.clock = clock

    def sweep(
        self, *, now: datetime | None = None, limit: int = DEFAULT_STALL_SWEEP_LIMIT
    ) -> StallSweepReport:
        """Stall every application whose last activity is a full window old."""
        now = now or self.clock()
        cutoff = now - self.window
        stalled: list[UUID] = []
        skipped: list[UUID] = []

        for application_id in self.store.quiet_since(cutoff, limit=limit):
            if self.store.mark_stalled(application_id, quiet_since=cutoff, now=now):
                stalled.append(application_id)
            else:
                skipped.append(application_id)

        if stalled:
            logger.info(
                "stall sweep at %s: %s application(s) stalled after %s without activity",
                now.isoformat(),
                len(stalled),
                self.window,
            )
        return StallSweepReport(stalled=tuple(stalled), skipped=tuple(skipped))

    async def run_forever(self, *, interval_seconds: float | None = None) -> None:
        """Sweep, sleep, repeat. One failed sweep is logged and does not end the loop."""
        interval = (
            interval_seconds
            if interval_seconds is not None
            else settings.application_stall_check_interval_seconds
        )
        while True:
            try:
                await asyncio.to_thread(self.sweep)
            except Exception:
                logger.exception("stall sweep failed; retrying in %s seconds", interval)
            await asyncio.sleep(interval)


__all__ = ["StallSweepReport", "StalledApplicationMonitor"]

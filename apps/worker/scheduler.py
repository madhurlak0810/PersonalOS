"""Runs the worker's scheduled sweeps: the clock behind follow-ups, STALLED and expiry.

Three things in this system happen because time passed rather than because
something occurred: a follow-up checkpoint comes due, an application goes quiet
long enough to stall, an approval request runs out of time. Each has a sweep
that knows how to find what is due -- `PendingCheckpointMonitor.sweep`,
`StalledApplicationMonitor.sweep`, `ApprovalQueue.expire_overdue` -- and this
module is what calls them.

**Nothing is scheduled in memory.** What is due is read from Postgres on every
pass: `pending_checkpoints.trigger_at` and `expires_at`,
`applications.last_activity_at`, an approval request's `expires_at`. The only
thing this process keeps between passes is when it next intends to look, and
losing that costs nothing -- a worker that was down for a week sweeps at
startup, finds everything that came due while it was gone, and acts on each of
them once. Once, because every sweep claims what it acts on with a guarded
write (`store.close(..., FIRED)`, `mark_stalled`, the approval's primary key),
so two workers sweeping the same minute, or a worker restarted mid-sweep, do
not act twice.

An interval is therefore a polling cadence, not a deadline: a follow-up due at
09:00 swept every five minutes fires by 09:05, and one due while no worker was
running fires as soon as one is.

One failing sweep is logged and retried at its next interval; it does not stop
the others, and it does not stop the worker.
"""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledSweep:
    """One sweep and how often to run it.

    `run` may be a coroutine function or a plain one; a plain one is run in a
    thread, since every sweep here is a database round trip and the worker's
    event loop is also delivering events and running workflows.
    """

    name: str
    interval_seconds: float
    run: Callable[[], Awaitable[Any] | Any]

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError(f"sweep '{self.name}' needs a positive interval")


class Scheduler:
    """Runs each `ScheduledSweep` at startup and then every `interval_seconds`."""

    def __init__(self, sweeps: Sequence[ScheduledSweep]):
        """Initialize with the sweeps to run; names must be unique, for the logs."""
        names = [sweep.name for sweep in sweeps]
        if len(names) != len(set(names)):
            raise ValueError(f"sweep names must be unique: {names}")
        self.sweeps = tuple(sweeps)

    async def run_once(self) -> dict[str, Any]:
        """Run every sweep one time, in order. Returns each one's result, or its error."""
        results: dict[str, Any] = {}
        for sweep in self.sweeps:
            results[sweep.name] = await self._run(sweep)
        return results

    async def run(self, stop: asyncio.Event) -> None:
        """Run every sweep now, then each on its own interval, until `stop` is set."""
        await asyncio.gather(*(self._loop(sweep, stop) for sweep in self.sweeps))

    async def _loop(self, sweep: ScheduledSweep, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self._run(sweep)
            try:
                await asyncio.wait_for(stop.wait(), timeout=sweep.interval_seconds)
            except asyncio.TimeoutError:
                pass

    @staticmethod
    async def _run(sweep: ScheduledSweep) -> Any:
        try:
            if inspect.iscoroutinefunction(sweep.run):
                return await sweep.run()
            return await asyncio.to_thread(sweep.run)
        except Exception as error:
            logger.exception(
                "scheduled sweep '%s' failed; retrying in %s seconds",
                sweep.name,
                sweep.interval_seconds,
            )
            return error


__all__ = ["ScheduledSweep", "Scheduler"]

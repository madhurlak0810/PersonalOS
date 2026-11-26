"""The worker's liveness signal: a file it touches while its event loop runs.

The worker has no port to probe, so its container health check
(`personalos.cli worker-health`) looks at how old this file is instead. The
heartbeat is an asyncio task on the same loop as the command, outbox and
scheduler loops, so a loop that is blocked or gone stops it too. It says the
worker is alive; whether its integrations are healthy is
`GET /v1/integrations`.
"""

import asyncio
import contextlib
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class Heartbeat:
    """Touches `path` every `interval` seconds until stopped."""

    def __init__(self, path: Path | str, interval: float = 10.0):
        """Bind to a file; nothing is written until `run`."""
        self.path = Path(path)
        self.interval = interval

    def beat(self) -> None:
        """Write the current time to the file (atomically)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.tmp")
        temp.write_text(f"{time.time():.3f}\n", encoding="utf-8")
        os.replace(temp, self.path)

    async def run(self, stop: asyncio.Event) -> None:
        """Beat until `stop` is set, then remove the file: a stopped worker is not live."""
        try:
            while not stop.is_set():
                try:
                    self.beat()
                except OSError as error:
                    logger.warning("worker heartbeat not written: %s", error)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self.interval)
        finally:
            with contextlib.suppress(OSError):
                self.path.unlink()


def heartbeat_age(path: Path | str, *, now: float | None = None) -> float | None:
    """Seconds since the last beat, or `None` if there has been none."""
    try:
        written = Path(path).stat().st_mtime
    except FileNotFoundError:
        return None
    return (time.time() if now is None else now) - written


__all__ = ["Heartbeat", "heartbeat_age"]

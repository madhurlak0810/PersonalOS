"""Domain events a graph emits, staged in the outbox.

`OutboxEventEmitter` is what `JobSearchGraph`'s `EventEmitter` port is bound
to. Each event is its own short transaction, staged exactly as
`SqlRecruiterEventStore` stages the events it writes alongside a row: by type,
payload and `dedupe_key`, skipping a key already in the outbox, so a step
re-run after a crash emits the same event rather than a second copy of it.
"""

from collections.abc import Callable

from sqlalchemy.orm import Session

from personalos.domain.job_search import EmittedEvent
from personalos.persistence.event_outbox import stage_raw_event


class OutboxEventEmitter:
    """Binds the graph's `EventEmitter` port to `outbox_events`."""

    def __init__(self, session_factory: Callable[[], Session]):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory

    async def emit(self, event: EmittedEvent) -> None:
        """Stage and commit the event, unless its `dedupe_key` is already staged."""
        session = self.session_factory()
        try:
            stage_raw_event(
                session,
                type=event.type.value,
                payload=event.payload,
                dedupe_key=event.dedupe_key,
                skip_duplicate=True,
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


__all__ = ["OutboxEventEmitter"]

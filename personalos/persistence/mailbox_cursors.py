"""Where the inbox poller is in a mailbox's history, moved together with what it found.

`SqlMailboxCursorStore.advance` is the poller's one write. In a single
transaction it stages every `email.received` envelope a read of the history
produced (`stage_new_event`, which skips a `dedupe_key` already in the outbox) and
moves the cursor past that read. So a poller that dies anywhere between
fetching a history page and recording the cursor leaves neither -- the next
poll reads the same page again and stages the same envelopes for the first
time -- and one that dies after the commit leaves both.

Exactly one event per message does not rest on that alone. A cursor can
legitimately be behind events already staged: a resync after an expired
cursor or a change of rules re-reads mail it may have seen, and two pollers
racing over one mailbox read the same page. Each of those stages envelopes
under the message's own `dedupe_key`, and the outbox keeps the first.

The cursor never moves backwards. A slower racing poller that commits an
older position than the one stored leaves the stored one in place.
"""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.events import EventEnvelope
from personalos.domain.mailbox import MailboxCursor
from personalos.persistence.event_outbox import stage_new_event
from personalos.persistence.models import MailboxCursorModel

logger = logging.getLogger(__name__)


class SqlMailboxCursorStore:
    """`mailbox_cursors`, and the outbox writes that move with it."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        clock: Callable[[], datetime] = datetime.utcnow,
    ):
        """Take the session factory; `clock` stamps `updated_at`."""
        self.session_factory = session_factory
        self.clock = clock

    def get(self, provider: str, account: str) -> MailboxCursor | None:
        """The stored cursor for one mailbox, if it has ever been polled."""
        session = self.session_factory()
        try:
            row = session.get(MailboxCursorModel, (provider, account.lower()))
            return _cursor(row) if row is not None else None
        finally:
            session.close()

    def advance(self, cursor: MailboxCursor, envelopes: Sequence[EventEnvelope]) -> int:
        """Stage `envelopes` and store `cursor`, in one transaction. Returns how many were new.

        Raises whatever the database raises, having written nothing.
        """
        session = self.session_factory()
        try:
            staged = self._stage(session, envelopes)
            self._store(session, cursor)
            session.commit()
            return staged
        except IntegrityError:
            # A racing poller committed the same message (or the same first
            # cursor) between our read and our insert. Its copy counts; ours
            # is discarded whole, and the next poll moves the cursor.
            session.rollback()
            logger.info(
                "a concurrent poll of %s/%s committed first; this read is dropped",
                cursor.provider,
                cursor.account,
            )
            return 0
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _stage(session: Session, envelopes: Sequence[EventEnvelope]) -> int:
        staged = 0
        seen: set[str] = set()
        for envelope in envelopes:
            # Twice in one read (two history records for one message) is
            # once: the outbox lookup does not see rows not yet flushed.
            if envelope.dedupe_key in seen:
                continue
            seen.add(envelope.dedupe_key)
            if stage_new_event(session, envelope):
                staged += 1
        return staged

    def _store(self, session: Session, cursor: MailboxCursor) -> None:
        now = self.clock()
        key = (cursor.provider, cursor.account.lower())
        row = session.get(MailboxCursorModel, key)
        if row is None:
            session.add(
                MailboxCursorModel(
                    provider=key[0],
                    account=key[1],
                    history_id=cursor.history_id,
                    scope_fingerprint=cursor.scope_fingerprint,
                    polled_at=cursor.polled_at,
                    created_at=now,
                    updated_at=now,
                )
            )
            return
        rescoped = row.scope_fingerprint != cursor.scope_fingerprint
        if not rescoped and int(cursor.history_id) < int(row.history_id):
            logger.info(
                "not moving the %s/%s cursor back from %s to %s",
                key[0],
                key[1],
                row.history_id,
                cursor.history_id,
            )
            return
        row.history_id = cursor.history_id
        row.scope_fingerprint = cursor.scope_fingerprint
        row.polled_at = cursor.polled_at
        row.updated_at = now


def _cursor(row: MailboxCursorModel) -> MailboxCursor:
    return MailboxCursor(
        provider=row.provider,
        account=row.account,
        history_id=row.history_id,
        scope_fingerprint=row.scope_fingerprint,
        polled_at=row.polled_at,
    )


__all__ = ["SqlMailboxCursorStore"]

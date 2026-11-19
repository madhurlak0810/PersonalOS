"""Idempotent API requests: one answer per (actor, `Idempotency-Key`).

A client that retries `POST /v1/chat` with the `Idempotency-Key` it sent the
first time -- after a timeout, a dropped connection, a double click -- gets
the first answer back, and no second workflow is created. The answer is
stored in the `operations` ledger the executor already dedupes side effects
with (`OperationRepository`), under a key derived from the actor and the
client's key, so one actor's key can never collide with, or reveal, another's.

`begin` is the whole protocol:

- a key never seen is claimed, and the caller does the work, then calls
  `complete` (or `fail`, which leaves the key free to retry);
- a key already answered returns that answer (`Replay`);
- a key in flight, or reused for a *different* request body, is an
  `IdempotencyConflict` (409).
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from personalos.domain.errors import IdempotencyConflict
from personalos.domain.models import OperationStatus
from personalos.persistence.repositories import OperationRepository

#: The longest `Idempotency-Key` a client may send.
MAX_IDEMPOTENCY_KEY_LENGTH = 255


def _digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def request_fingerprint(body: Any) -> str:
    """A stable hash of a request body, to tell a replay from a reused key."""
    return _digest(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str))


@dataclass(frozen=True)
class Replay:
    """The stored answer to a request already made under this key."""

    result: dict[str, Any]


@dataclass(frozen=True)
class Claim:
    """This request owns the key: do the work, then `complete` or `fail` it."""

    key: str


class IdempotentRequests:
    """Claims, replays and settles idempotency keys for one API operation."""

    def __init__(self, session_factory: Callable[[], Session], *, operation: str):
        self.session_factory = session_factory
        self.operation = operation

    def ledger_key(self, actor_id: str, client_key: str) -> str:
        """The `operations` key: the operation, then a hash of actor and client key."""
        return f"api.{self.operation}:{_digest(actor_id, client_key)}"

    def begin(self, actor_id: str, client_key: str, fingerprint: str) -> Claim | Replay:
        key = self.ledger_key(actor_id, client_key)
        session = self.session_factory()
        try:
            record, claimed = OperationRepository(session).claim(
                key, f"api.{self.operation}", fingerprint
            )
        finally:
            session.close()
        if claimed:
            return Claim(key)
        if record.request_fingerprint != fingerprint:
            raise IdempotencyConflict(
                "this Idempotency-Key was already used for a different request; "
                "use a new key for a new request"
            )
        if record.status == OperationStatus.COMPLETED and record.result is not None:
            return Replay(record.result)
        raise IdempotencyConflict(
            "a request with this Idempotency-Key is still being processed; retry shortly"
        )

    def complete(self, claim: Claim, result: dict[str, Any]) -> None:
        session = self.session_factory()
        try:
            OperationRepository(session).complete(claim.key, result)
        finally:
            session.close()

    def fail(self, claim: Claim, error: str) -> None:
        session = self.session_factory()
        try:
            OperationRepository(session).fail(claim.key, error)
        finally:
            session.close()


__all__ = [
    "MAX_IDEMPOTENCY_KEY_LENGTH",
    "Claim",
    "IdempotentRequests",
    "Replay",
    "request_fingerprint",
]

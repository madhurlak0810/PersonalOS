"""Idempotency guardrails for mutating operations.

Every mutating action runs through an :class:`IdempotencyGuard`, which claims the
caller's idempotency key in an operation store before the side effect runs and
records the outcome afterwards. A replay of the same key returns the stored
result instead of executing the side effect again.
"""

import hashlib
import inspect
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from threading import RLock
from typing import Any
from uuid import uuid4

from personalos.domain.errors import ErrorCode, PersonalOSError
from personalos.domain.models import (
    OperationRecord,
    OperationStatus,
    validate_idempotency_key,
)
from personalos.persistence.repositories import OperationRepository

logger = logging.getLogger(__name__)


class IdempotencyError(PersonalOSError):
    """Base class for idempotency guardrail violations."""

    code = ErrorCode.IDEMPOTENCY_CONFLICT
    http_status = 409


class IdempotencyKeyReused(IdempotencyError):
    """The key was already used for an operation with different parameters."""


class OperationInProgress(IdempotencyError):
    """A concurrent attempt already owns this key and has not finished.

    Marked retryable: the caller did nothing wrong, the operation just has not
    settled yet.
    """

    code = ErrorCode.RETRYABLE
    retryable = True


def fingerprint_request(operation: str, params: dict[str, Any]) -> str:
    """Fingerprint an operation and its parameters.

    Canonical JSON keeps the hash stable across key ordering. Values that JSON
    cannot represent fall back to their repr so fingerprinting never raises and
    block an otherwise valid operation.
    """
    payload = json.dumps(
        {"operation": operation, "params": params},
        sort_keys=True,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OperationStore(ABC):
    """Storage for operation records, keyed by idempotency key."""

    @abstractmethod
    def get(self, idempotency_key: str) -> OperationRecord | None:
        """Get the record for a key, or None."""

    @abstractmethod
    def claim(
        self, idempotency_key: str, operation: str, request_fingerprint: str
    ) -> tuple[OperationRecord, bool]:
        """Claim a key. Returns (record, True) when the caller owns execution."""

    @abstractmethod
    def complete(self, idempotency_key: str, result: Any) -> OperationRecord:
        """Record a successful outcome."""

    @abstractmethod
    def fail(self, idempotency_key: str, error: str) -> OperationRecord:
        """Record a failed outcome, leaving the key retryable."""


class SqlOperationStore(OperationStore):
    """Operation store backed by the `operations` table.

    Takes a session factory rather than a session so each claim/complete cycle
    gets its own short-lived transaction, independent of the caller's session.
    """

    def __init__(self, session_factory: Callable[[], Any]):
        """Initialize with a callable returning a SQLAlchemy Session."""
        self.session_factory = session_factory

    def get(self, idempotency_key: str) -> OperationRecord | None:
        """Get the record for a key, or None."""
        with self._repository() as repo:
            return repo.get_by_key(idempotency_key)

    def claim(
        self, idempotency_key: str, operation: str, request_fingerprint: str
    ) -> tuple[OperationRecord, bool]:
        """Claim a key for execution."""
        with self._repository() as repo:
            return repo.claim(idempotency_key, operation, request_fingerprint)

    def complete(self, idempotency_key: str, result: Any) -> OperationRecord:
        """Record a successful outcome."""
        with self._repository() as repo:
            return repo.complete(idempotency_key, result)

    def fail(self, idempotency_key: str, error: str) -> OperationRecord:
        """Record a failed outcome."""
        with self._repository() as repo:
            return repo.fail(idempotency_key, error)

    class _RepositoryContext:
        def __init__(self, session_factory: Callable[[], Any]):
            self.session_factory = session_factory
            self.session = None

        def __enter__(self) -> OperationRepository:
            self.session = self.session_factory()
            return OperationRepository(self.session)

        def __exit__(self, exc_type, exc, tb):
            if exc_type is not None:
                self.session.rollback()
            self.session.close()
            return False

    def _repository(self) -> "_RepositoryContext":
        return self._RepositoryContext(self.session_factory)


class InMemoryOperationStore(OperationStore):
    """Process-local operation store.

    Deduplicates within a single process only — it does not survive a restart or
    coordinate across workers. Use :class:`SqlOperationStore` in production; this
    exists for tests and for single-process tools with no database configured.
    """

    def __init__(self):
        """Initialize an empty store."""
        self._records: dict[str, OperationRecord] = {}
        self._lock = RLock()

    def get(self, idempotency_key: str) -> OperationRecord | None:
        """Get the record for a key, or None."""
        with self._lock:
            record = self._records.get(idempotency_key)
            return record.model_copy(deep=True) if record else None

    def claim(
        self, idempotency_key: str, operation: str, request_fingerprint: str
    ) -> tuple[OperationRecord, bool]:
        """Claim a key for execution."""
        with self._lock:
            existing = self._records.get(idempotency_key)

            if existing is None:
                record = OperationRecord(
                    id=uuid4(),
                    idempotency_key=idempotency_key,
                    operation=operation,
                    request_fingerprint=request_fingerprint,
                    status=OperationStatus.IN_PROGRESS,
                )
                self._records[idempotency_key] = record
                return record.model_copy(deep=True), True

            # Only a failure of the *same* request is retryable; a key reused
            # for a different request is returned unchanged so the caller can
            # reject it rather than clobbering the original record.
            if (
                existing.status == OperationStatus.FAILED
                and existing.request_fingerprint == request_fingerprint
            ):
                existing.status = OperationStatus.IN_PROGRESS
                existing.attempts += 1
                existing.error = None
                existing.updated_at = datetime.utcnow()
                return existing.model_copy(deep=True), True

            return existing.model_copy(deep=True), False

    def complete(self, idempotency_key: str, result: Any) -> OperationRecord:
        """Record a successful outcome."""
        with self._lock:
            record = self._require(idempotency_key)
            now = datetime.utcnow()
            record.status = OperationStatus.COMPLETED
            record.result = result
            record.error = None
            record.updated_at = now
            record.completed_at = now
            return record.model_copy(deep=True)

    def fail(self, idempotency_key: str, error: str) -> OperationRecord:
        """Record a failed outcome."""
        with self._lock:
            record = self._require(idempotency_key)
            record.status = OperationStatus.FAILED
            record.error = error
            record.updated_at = datetime.utcnow()
            return record.model_copy(deep=True)

    def _require(self, idempotency_key: str) -> OperationRecord:
        record = self._records.get(idempotency_key)
        if record is None:
            raise ValueError(f"Operation '{idempotency_key}' not found")
        return record


class IdempotencyGuard:
    """Runs mutating operations at most once per idempotency key."""

    def __init__(self, store: OperationStore):
        """Initialize with the operation store to record against."""
        self.store = store

    async def run(
        self,
        operation: str,
        idempotency_key: str,
        params: dict[str, Any],
        handler: Callable[..., Any],
    ) -> tuple[Any, bool]:
        """Execute `handler(**params)` at most once for `idempotency_key`.

        Returns (result, replayed). `replayed` is True when the result came from
        a prior completed attempt and the side effect was not re-executed.

        Raises InvalidIdempotencyKey for a malformed key, IdempotencyKeyReused
        when the key was used for different parameters, and OperationInProgress
        when a concurrent attempt owns the key. A handler exception is recorded
        as a failure and re-raised, leaving the key retryable.
        """
        key = validate_idempotency_key(idempotency_key)
        fingerprint = fingerprint_request(operation, params)

        record, claimed = self.store.claim(key, operation, fingerprint)

        if record.request_fingerprint != fingerprint:
            raise IdempotencyKeyReused(
                f"idempotency_key '{key}' was already used for a different request "
                f"(operation '{record.operation}'); use a new key"
            )

        if not claimed:
            if record.is_replayable:
                logger.info(
                    f"Replaying prior result for '{operation}' (idempotency_key={key})"
                )
                return record.result, True
            raise OperationInProgress(
                f"operation '{operation}' with idempotency_key '{key}' is already "
                f"in progress; retry once it settles"
            )

        try:
            result = handler(**params)
            # Awaits any awaitable rather than testing the handler itself, so
            # callables whose __call__ is async are handled too — a coroutine
            # stored as a "result" would be replayed as an unusable object.
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:
            self.store.fail(key, str(e))
            raise

        self.store.complete(key, result)
        return result, False

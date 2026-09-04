"""Tests for idempotency guardrails on mutating operations."""

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from personalos.domain.models import (
    ActionTarget,
    Intent,
    InvalidIdempotencyKey,
    MutatingIntent,
    OperationStatus,
    ToolCallErrorCode,
    ToolCallRequest,
)
from personalos.mcp.base import MCPServer, ToolSchema
from personalos.persistence.idempotency import (
    IdempotencyGuard,
    IdempotencyKeyReused,
    InMemoryOperationStore,
    OperationInProgress,
    SqlOperationStore,
    fingerprint_request,
)
from personalos.persistence.models import OperationModel
from personalos.persistence.repositories import OperationRepository


def new_key() -> str:
    """Generate a fresh idempotency key."""
    return str(uuid4())


@pytest.fixture
def sqlite_session_factory(tmp_path):
    """Session factory over a file-backed SQLite database.

    Only the operations table is created - the other models use Postgres-native
    UUID columns and are not under test here. The database is file-backed rather
    than in-memory so each session gets its own connection and transaction,
    which is what makes the unique-constraint race meaningful.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path / 'operations.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    OperationModel.__table__.create(engine)
    yield sessionmaker(autocommit=False, autoflush=False, bind=engine)
    engine.dispose()


@pytest.fixture(params=["memory", "sql"])
def store(request, sqlite_session_factory):
    """Run store-level tests against both store implementations."""
    if request.param == "memory":
        return InMemoryOperationStore()
    return SqlOperationStore(sqlite_session_factory)


class CountingHandler:
    """Records how many times the side effect actually ran.

    Pass `handler.run` where a tool handler is expected - a bound coroutine
    method looks to `inspect` exactly like a real async tool handler.
    """

    def __init__(self, fail_times: int = 0):
        self.calls = 0
        self.fail_times = fail_times

    async def run(self, **kwargs):
        """Simulate a side effect, failing the first `fail_times` attempts."""
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"transient failure #{self.calls}")
        return {"saved": True, "call_number": self.calls, **kwargs}


# --- Intent contract -------------------------------------------------------


class SaveFavoriteIntent(MutatingIntent):
    """Test intent for a mutating action."""

    job_id: str
    notes: str = ""


def test_mutating_intent_requires_idempotency_key():
    """A mutating intent cannot be constructed without an idempotency key."""
    with pytest.raises(Exception) as exc_info:
        SaveFavoriteIntent(job_id="job_1")

    assert "idempotency_key" in str(exc_info.value)


def test_mutating_intent_rejects_low_entropy_key():
    """Keys too short to be unique are rejected."""
    with pytest.raises(Exception) as exc_info:
        SaveFavoriteIntent(job_id="job_1", idempotency_key="abc")

    assert "at least" in str(exc_info.value)


def test_mutating_intent_normalizes_key():
    """Surrounding whitespace is stripped so ' k ' and 'k' dedupe together."""
    key = new_key()
    intent = SaveFavoriteIntent(job_id="job_1", idempotency_key=f"  {key}  ")

    assert intent.idempotency_key == key


def test_side_effect_params_exclude_key():
    """The fingerprint covers the side effect, not the key identifying it."""
    intent = SaveFavoriteIntent(job_id="job_1", notes="hi", idempotency_key=new_key())

    assert intent.side_effect_params() == {"job_id": "job_1", "notes": "hi"}


def test_fingerprint_is_order_independent():
    """Key ordering must not change the request fingerprint."""
    a = fingerprint_request("jobs.save", {"job_id": "1", "notes": "x"})
    b = fingerprint_request("jobs.save", {"notes": "x", "job_id": "1"})

    assert a == b
    assert a != fingerprint_request("jobs.save", {"job_id": "2", "notes": "x"})


def test_fingerprint_survives_unserializable_values():
    """Fingerprinting never raises on values JSON cannot represent."""
    assert fingerprint_request("jobs.save", {"when": object()})


# --- Store / repository semantics -----------------------------------------


def test_claim_is_exclusive(store):
    """The first claim owns execution; a second claim of the same key does not."""
    key = new_key()
    fingerprint = fingerprint_request("jobs.save", {"job_id": "1"})

    first, claimed_first = store.claim(key, "jobs.save", fingerprint)
    second, claimed_second = store.claim(key, "jobs.save", fingerprint)

    assert claimed_first is True
    assert claimed_second is False
    assert first.status == OperationStatus.IN_PROGRESS
    assert second.id == first.id


def test_completed_operation_is_replayable(store):
    """A completed operation exposes its stored result for replay."""
    key = new_key()
    fingerprint = fingerprint_request("jobs.save", {"job_id": "1"})
    store.claim(key, "jobs.save", fingerprint)

    store.complete(key, {"saved": True})
    record = store.get(key)

    assert record.status == OperationStatus.COMPLETED
    assert record.is_replayable is True
    assert record.result == {"saved": True}
    assert record.completed_at is not None


def test_failed_operation_is_reclaimable(store):
    """A failure leaves the key retryable and bumps the attempt count."""
    key = new_key()
    fingerprint = fingerprint_request("jobs.save", {"job_id": "1"})
    store.claim(key, "jobs.save", fingerprint)
    store.fail(key, "boom")

    failed = store.get(key)
    assert failed.status == OperationStatus.FAILED
    assert failed.error == "boom"
    assert failed.is_replayable is False

    record, claimed = store.claim(key, "jobs.save", fingerprint)
    assert claimed is True
    assert record.attempts == 2
    assert record.error is None


def test_repository_rejects_unknown_key(sqlite_session_factory):
    """Completing an operation that was never claimed is an error."""
    session = sqlite_session_factory()
    try:
        repo = OperationRepository(session)
        with pytest.raises(ValueError):
            repo.complete(new_key(), {"saved": True})
    finally:
        session.close()


def test_concurrent_claims_yield_single_owner(sqlite_session_factory):
    """Under a race, the unique constraint lets exactly one caller claim the key."""
    key = new_key()
    fingerprint = fingerprint_request("jobs.save", {"job_id": "1"})
    store = SqlOperationStore(sqlite_session_factory)

    def attempt(_):
        return store.claim(key, "jobs.save", fingerprint)[1]

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(8)))

    assert outcomes.count(True) == 1

    session = sqlite_session_factory()
    try:
        rows = session.query(OperationModel).filter(
            OperationModel.idempotency_key == key
        ).count()
    finally:
        session.close()
    assert rows == 1


# --- Guard behaviour -------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_runs_side_effect_once(store):
    """Retrying with the same key replays the result without re-executing."""
    handler = CountingHandler()
    guard = IdempotencyGuard(store)
    key = new_key()
    params = {"job_id": "job_1"}

    first, replayed_first = await guard.run("jobs.save", key, params, handler.run)
    second, replayed_second = await guard.run("jobs.save", key, params, handler.run)

    assert handler.calls == 1
    assert replayed_first is False
    assert replayed_second is True
    assert second == first


@pytest.mark.asyncio
async def test_guard_distinct_keys_both_execute(store):
    """Different keys are different operations, even with identical params."""
    handler = CountingHandler()
    guard = IdempotencyGuard(store)
    params = {"job_id": "job_1"}

    await guard.run("jobs.save", new_key(), params, handler.run)
    await guard.run("jobs.save", new_key(), params, handler.run)

    assert handler.calls == 2


@pytest.mark.asyncio
async def test_guard_retries_after_failure(store):
    """A failed attempt is retryable, and the retry's result is then cached."""
    handler = CountingHandler(fail_times=1)
    guard = IdempotencyGuard(store)
    key = new_key()
    params = {"job_id": "job_1"}

    with pytest.raises(RuntimeError):
        await guard.run("jobs.save", key, params, handler.run)

    assert store.get(key).status == OperationStatus.FAILED

    result, replayed = await guard.run("jobs.save", key, params, handler.run)
    assert replayed is False
    assert result["call_number"] == 2

    replay, replayed_again = await guard.run("jobs.save", key, params, handler.run)
    assert replayed_again is True
    assert replay == result
    assert handler.calls == 2


@pytest.mark.asyncio
async def test_guard_rejects_key_reuse_with_different_params(store):
    """The same key with a different request is a caller bug, not a replay."""
    handler = CountingHandler()
    guard = IdempotencyGuard(store)
    key = new_key()

    await guard.run("jobs.save", key, {"job_id": "job_1"}, handler.run)

    with pytest.raises(IdempotencyKeyReused):
        await guard.run("jobs.save", key, {"job_id": "job_2"}, handler.run)

    assert handler.calls == 1


@pytest.mark.asyncio
async def test_key_reuse_after_failure_leaves_record_intact(store):
    """Reusing a failed key for a different request must not clobber the record."""
    handler = CountingHandler(fail_times=1)
    guard = IdempotencyGuard(store)
    key = new_key()

    with pytest.raises(RuntimeError):
        await guard.run("jobs.save", key, {"job_id": "job_1"}, handler.run)

    with pytest.raises(IdempotencyKeyReused):
        await guard.run("jobs.save", key, {"job_id": "job_2"}, handler.run)

    # Still recorded as the original failed request, not re-claimed under the
    # wrong parameters, so the genuine retry can still proceed.
    record = store.get(key)
    assert record.status == OperationStatus.FAILED
    assert record.attempts == 1
    assert handler.calls == 1

    result, replayed = await guard.run("jobs.save", key, {"job_id": "job_1"}, handler.run)
    assert replayed is False
    assert result["job_id"] == "job_1"


@pytest.mark.asyncio
async def test_guard_rejects_key_reuse_across_operations(store):
    """Keys are scoped globally, so reuse across operations is rejected too."""
    handler = CountingHandler()
    guard = IdempotencyGuard(store)
    key = new_key()

    await guard.run("jobs.save", key, {"job_id": "job_1"}, handler.run)

    with pytest.raises(IdempotencyKeyReused):
        await guard.run("jobs.delete", key, {"job_id": "job_1"}, handler.run)


@pytest.mark.asyncio
async def test_guard_rejects_concurrent_in_progress_operation(store):
    """A key still in progress is refused rather than executed twice."""
    handler = CountingHandler()
    guard = IdempotencyGuard(store)
    key = new_key()
    params = {"job_id": "job_1"}
    fingerprint = fingerprint_request("jobs.save", params)

    # Simulate an attempt that claimed the key and has not settled yet.
    store.claim(key, "jobs.save", fingerprint)

    with pytest.raises(OperationInProgress):
        await guard.run("jobs.save", key, params, handler.run)

    assert handler.calls == 0


@pytest.mark.asyncio
async def test_guard_requires_key(store):
    """A missing key is rejected before the handler runs."""
    handler = CountingHandler()
    guard = IdempotencyGuard(store)

    with pytest.raises(InvalidIdempotencyKey):
        await guard.run("jobs.save", None, {"job_id": "job_1"}, handler.run)

    assert handler.calls == 0


@pytest.mark.asyncio
async def test_guard_supports_sync_handlers(store):
    """Synchronous handlers are supported alongside coroutines."""
    calls = []

    def handler(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    guard = IdempotencyGuard(store)
    key = new_key()

    await guard.run("jobs.save", key, {"job_id": "job_1"}, handler)
    _, replayed = await guard.run("jobs.save", key, {"job_id": "job_1"}, handler)

    assert len(calls) == 1
    assert replayed is True


# --- MCP integration -------------------------------------------------------


class _DoMutateIntent(MutatingIntent):
    job_id: str


class _DoReadIntent(Intent):
    job_id: str


class _TestServer(MCPServer):
    """Minimal server exposing one mutating and one read-only tool."""

    def __init__(self, operation_store=None):
        super().__init__("test", "test server", operation_store=operation_store)
        self.mutating_handler = CountingHandler()
        self.read_handler = CountingHandler()
        self.initialize()

    def initialize(self):
        self.register_tool(
            ToolSchema(
                name="do_mutate",
                description="mutating tool",
                intent_type=_DoMutateIntent,
                parameters={"type": "object", "properties": {"job_id": {"type": "string"}}},
                required=["job_id"],
            ),
            self.mutating_handler.run,
        )
        self.register_tool(
            ToolSchema(
                name="do_read",
                description="read-only tool",
                intent_type=_DoReadIntent,
                parameters={"type": "object", "properties": {"job_id": {"type": "string"}}},
                required=["job_id"],
            ),
            self.read_handler.run,
        )


def _request(tool_name: str, **params) -> ToolCallRequest:
    """Build a ToolCallRequest against the `_TestServer`'s "test" server."""
    return ToolCallRequest(target=ActionTarget(server="test", tool=tool_name), params=params)


def test_mutating_tool_advertises_idempotency_key():
    """The key is part of the published contract for mutating tools only."""
    server = _TestServer(InMemoryOperationStore())

    mutating = server.get_tool_schema("do_mutate")
    read_only = server.get_tool_schema("do_read")

    assert mutating.mutating is True
    assert read_only.mutating is False
    assert "idempotency_key" in mutating.parameters["properties"]
    assert "idempotency_key" in mutating.required
    assert "idempotency_key" not in read_only.parameters["properties"]
    assert "idempotency_key" not in read_only.required


@pytest.mark.asyncio
async def test_mutating_tool_rejects_missing_key():
    """Calling a mutating tool without a key fails without side effects."""
    server = _TestServer(InMemoryOperationStore())

    result = await server.execute(_request("do_mutate", job_id="job_1"))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.VALIDATION_ERROR
    assert "idempotency_key" in result.error.message
    assert server.mutating_handler.calls == 0


@pytest.mark.asyncio
async def test_mutating_tool_requires_operation_store():
    """Without a store there is no dedup, so the call is refused."""
    server = _TestServer(operation_store=None)

    result = await server.execute(
        _request("do_mutate", job_id="job_1", idempotency_key=new_key())
    )

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.MISSING_OPERATION_STORE
    assert "operation store" in result.error.message
    assert server.mutating_handler.calls == 0


@pytest.mark.asyncio
async def test_mutating_tool_dedupes_retry(sqlite_session_factory):
    """A retried tool call returns the prior result and does not re-execute."""
    server = _TestServer(SqlOperationStore(sqlite_session_factory))
    key = new_key()

    first = await server.execute(_request("do_mutate", job_id="job_1", idempotency_key=key))
    second = await server.execute(_request("do_mutate", job_id="job_1", idempotency_key=key))

    assert first.ok is True
    assert first.replayed is False
    assert second.ok is True
    assert second.replayed is True
    assert second.result == first.result
    assert second.idempotency_key == key
    assert server.mutating_handler.calls == 1


@pytest.mark.asyncio
async def test_mutating_tool_surfaces_key_reuse_as_error():
    """Key reuse with different params surfaces as a typed failed tool result."""
    server = _TestServer(InMemoryOperationStore())
    key = new_key()

    await server.execute(_request("do_mutate", job_id="job_1", idempotency_key=key))
    result = await server.execute(_request("do_mutate", job_id="job_2", idempotency_key=key))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.IDEMPOTENCY_CONFLICT
    assert "already used for a different request" in result.error.message
    assert server.mutating_handler.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["do_mutate", "do_read"])
async def test_execute_awaits_async_callable_objects(tool_name):
    """A handler whose __call__ is async must be awaited, not returned raw.

    `inspect.iscoroutinefunction` is False for such objects, so testing the
    handler instead of its return value would surface an un-awaited coroutine
    as the tool's result.
    """

    class AsyncCallable:
        async def __call__(self, **kwargs):
            return {"awaited": True, **kwargs}

    server = _TestServer(InMemoryOperationStore())
    server.register_tool(server.get_tool_schema(tool_name), AsyncCallable())

    params = {"job_id": "job_1"}
    if tool_name == "do_mutate":
        params["idempotency_key"] = new_key()
    result = await server.execute(_request(tool_name, **params))

    assert result.ok is True
    assert result.result == {"awaited": True, "job_id": "job_1"}


@pytest.mark.asyncio
async def test_read_only_tool_needs_no_key():
    """Read-only tools are unaffected by the guardrail."""
    server = _TestServer(InMemoryOperationStore())

    first = await server.execute(_request("do_read", job_id="job_1"))
    second = await server.execute(_request("do_read", job_id="job_1"))

    assert first.ok is True
    assert second.ok is True
    assert first.replayed is None
    assert server.read_handler.calls == 2


# --- Typed contract validation ----------------------------------------------


def test_tool_call_request_rejects_blank_target():
    """ActionTarget requires non-blank server and tool names."""
    with pytest.raises(Exception):
        ActionTarget(server="", tool="do_read")


@pytest.mark.asyncio
async def test_unknown_tool_returns_typed_not_found():
    """A request for a tool the server doesn't have fails with a typed code."""
    server = _TestServer(InMemoryOperationStore())

    result = await server.execute(_request("does_not_exist"))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.TOOL_NOT_FOUND
    assert result.result is None


@pytest.mark.asyncio
async def test_missing_required_field_is_typed_validation_error():
    """A read-only tool called without its required field fails typed, not by exception."""
    server = _TestServer(InMemoryOperationStore())

    result = await server.execute(ToolCallRequest(target=ActionTarget(server="test", tool="do_read"), params={}))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.VALIDATION_ERROR
    assert result.error.details["errors"]


@pytest.mark.asyncio
async def test_unknown_field_is_typed_validation_error():
    """Extra, unrecognized params are rejected rather than silently dropped."""
    server = _TestServer(InMemoryOperationStore())

    result = await server.execute(_request("do_read", job_id="job_1", extra_field="nope"))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.VALIDATION_ERROR


@pytest.mark.asyncio
async def test_wrong_field_type_is_typed_validation_error():
    """A field of the wrong type fails typed validation before the handler runs."""
    server = _TestServer(InMemoryOperationStore())

    result = await server.execute(_request("do_read", job_id={"not": "a string"}))

    assert result.ok is False
    assert result.error.code == ToolCallErrorCode.VALIDATION_ERROR
    assert server.read_handler.calls == 0


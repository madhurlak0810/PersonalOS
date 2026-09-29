"""Contract tests for the workflow lease: the rule that keeps resumes exclusive.

The lease's job is one sentence -- at most one worker may be running a given
workflow -- and every test here is a way that sentence can be broken:

- two workers acquiring from nothing, or from an expired lease;
- a worker that died holding one, and whether its workflow is stranded;
- a stalled worker waking up after losing its lease, and whether it can release
  or renew the one that replaced it.

The clock is injected rather than slept through, so expiry is tested exactly
rather than approximately. The concurrency here is real threads over the same
file-backed SQLite database: the guarantee is supposed to come from a unique
constraint and a guarded `UPDATE` rather than from `SELECT ... FOR UPDATE` (which
SQLite ignores), so this is the dialect where a design that leaned on row locks
would be caught.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Barrier
from uuid import uuid4

import pytest

from personalos.persistence.checkpointer import WorkflowThreadRegistry
from personalos.persistence.leases import (
    LEASE_EXCLUSION_REASONS,
    LEASE_REASON_HELD,
    WorkflowLeaseLost,
    WorkflowLeaseStore,
    WorkflowLeaseUnavailable,
    default_owner,
)
from tests.fixtures.durable_workflow import session_factory


class FrozenClock:
    """A clock the test advances by hand, so expiry needs no sleeping."""

    def __init__(self, now: datetime | None = None):
        self.now = now or datetime(2026, 9, 29, 12, 0, 0)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _workflow(db_path):
    """A registered workflow to take leases against, and its session factory."""
    factory = session_factory(db_path)
    thread = WorkflowThreadRegistry(factory).register(thread_id="t-1", workflow_name="job_search")
    return thread, factory


# --- Exclusion ---------------------------------------------------------------


def test_a_lease_is_granted_once_and_refused_to_everyone_else(tmp_path):
    thread, factory = _workflow(tmp_path / "lease.db")
    store = WorkflowLeaseStore(factory)

    lease = store.acquire(thread.workflow_id, owner="worker-a")

    assert lease.owner == "worker-a"
    with pytest.raises(WorkflowLeaseUnavailable) as refusal:
        store.acquire(thread.workflow_id, owner="worker-b")
    assert refusal.value.details == {"reason": LEASE_REASON_HELD, "holder": "worker-a"}


def test_releasing_a_lease_frees_the_workflow_for_the_next_worker(tmp_path):
    thread, factory = _workflow(tmp_path / "lease.db")
    store = WorkflowLeaseStore(factory)

    first = store.acquire(thread.workflow_id, owner="worker-a")
    assert store.release(first) is True

    second = store.acquire(thread.workflow_id, owner="worker-b")
    assert second.owner == "worker-b"
    assert second.token != first.token


def test_current_reports_the_holder_and_nothing_once_released(tmp_path):
    thread, factory = _workflow(tmp_path / "lease.db")
    store = WorkflowLeaseStore(factory)

    assert store.current(thread.workflow_id) is None
    lease = store.acquire(thread.workflow_id, owner="worker-a")
    assert store.current(thread.workflow_id).owner == "worker-a"
    store.release(lease)
    assert store.current(thread.workflow_id) is None


def test_the_hold_context_releases_even_when_the_block_raises(tmp_path):
    """A failed resume must not keep its lease until the TTL runs out.

    Otherwise the retry that is supposed to follow a failure is locked out by the
    failure itself.
    """
    thread, factory = _workflow(tmp_path / "lease.db")
    store = WorkflowLeaseStore(factory)

    with pytest.raises(RuntimeError, match="resume blew up"):
        with store.hold(thread.workflow_id, owner="worker-a"):
            raise RuntimeError("resume blew up")

    assert store.current(thread.workflow_id) is None
    assert store.acquire(thread.workflow_id, owner="worker-b").owner == "worker-b"


def test_only_one_of_many_simultaneous_acquisitions_succeeds(tmp_path):
    """Eight threads race for one lease from nothing; exactly one gets it.

    Racing from *no* lease row at all is the case the unique constraint on
    `workflow_leases.workflow_id` decides, which is the half of the design that
    does not depend on the dialect supporting row locks.
    """
    thread, _factory = _workflow(tmp_path / "lease.db")
    workers = 8
    barrier = Barrier(workers)

    def acquire(index: int):
        store = WorkflowLeaseStore(session_factory(tmp_path / "lease.db"))
        barrier.wait(timeout=30)
        try:
            return store.acquire(thread.workflow_id, owner=f"worker-{index}")
        except WorkflowLeaseUnavailable as exc:
            return exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(acquire, range(workers)))

    granted = [result for result in results if not isinstance(result, Exception)]
    refused = [result for result in results if isinstance(result, Exception)]

    assert len(granted) == 1, f"{len(granted)} workers were granted the same lease"
    assert len(refused) == workers - 1
    assert all(
        exc.details["reason"] in LEASE_EXCLUSION_REASONS | {"database_locked"} for exc in refused
    )


def test_only_one_of_many_simultaneous_takeovers_of_an_expired_lease_succeeds(tmp_path):
    """The same race, but from an expired lease, which the guarded UPDATE decides.

    A different code path from the empty case and the one a real orphaned
    workflow hits: several workers all notice the same dead holder at once.
    """
    thread, factory = _workflow(tmp_path / "lease.db")
    clock = FrozenClock()
    WorkflowLeaseStore(factory, ttl_seconds=60, clock=clock).acquire(
        thread.workflow_id, owner="worker-dead"
    )
    clock.advance(120)
    later = clock.now

    workers = 8
    barrier = Barrier(workers)

    def take_over(index: int):
        store = WorkflowLeaseStore(
            session_factory(tmp_path / "lease.db"), ttl_seconds=60, clock=lambda: later
        )
        barrier.wait(timeout=30)
        try:
            return store.acquire(thread.workflow_id, owner=f"worker-{index}")
        except WorkflowLeaseUnavailable as exc:
            return exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(take_over, range(workers)))

    granted = [result for result in results if not isinstance(result, Exception)]
    assert len(granted) == 1, f"{len(granted)} workers took over the same expired lease"


# --- Expiry and fencing ------------------------------------------------------


def test_an_expired_lease_can_be_taken_over(tmp_path):
    """A workflow orphaned by a hard kill becomes resumable rather than stranded."""
    thread, factory = _workflow(tmp_path / "lease.db")
    clock = FrozenClock()
    store = WorkflowLeaseStore(factory, ttl_seconds=60, clock=clock)

    dead = store.acquire(thread.workflow_id, owner="worker-killed")
    clock.advance(59)
    with pytest.raises(WorkflowLeaseUnavailable):
        store.acquire(thread.workflow_id, owner="worker-b")

    clock.advance(2)
    taken = store.acquire(thread.workflow_id, owner="worker-b")

    assert taken.owner == "worker-b"
    assert taken.token != dead.token


def test_a_renewed_lease_does_not_expire_under_its_holder(tmp_path):
    """Renewal is how a long step keeps its lease without a long TTL."""
    thread, factory = _workflow(tmp_path / "lease.db")
    clock = FrozenClock()
    store = WorkflowLeaseStore(factory, ttl_seconds=60, clock=clock)

    lease = store.acquire(thread.workflow_id, owner="worker-a")
    clock.advance(50)
    renewed = store.renew(lease)
    clock.advance(50)

    assert renewed.expires_at > clock.now
    with pytest.raises(WorkflowLeaseUnavailable):
        store.acquire(thread.workflow_id, owner="worker-b")


def test_a_worker_that_lost_its_lease_cannot_release_its_successors(tmp_path):
    """The fencing token, which is the whole reason expiry is safe.

    A worker that stalled past its TTL and then woke up still holds a lease
    object. If its release were honoured, it would hand the workflow to a third
    worker while its successor was still running -- two workers on one workflow,
    arrived at by being careful rather than careless.
    """
    thread, factory = _workflow(tmp_path / "lease.db")
    clock = FrozenClock()
    store = WorkflowLeaseStore(factory, ttl_seconds=60, clock=clock)

    stalled = store.acquire(thread.workflow_id, owner="worker-stalled")
    clock.advance(120)
    successor = store.acquire(thread.workflow_id, owner="worker-successor")

    assert store.release(stalled) is False
    assert store.current(thread.workflow_id).owner == "worker-successor"

    with pytest.raises(WorkflowLeaseUnavailable):
        store.acquire(thread.workflow_id, owner="worker-third")
    assert store.release(successor) is True


def test_renewing_a_lost_lease_raises_rather_than_quietly_extending_it(tmp_path):
    """A stalled worker is told it has lost the work, not allowed to keep it.

    Raising rather than returning `False`: whatever that worker has in flight
    must stop, and a boolean it might not check would let it carry on.
    """
    thread, factory = _workflow(tmp_path / "lease.db")
    clock = FrozenClock()
    store = WorkflowLeaseStore(factory, ttl_seconds=60, clock=clock)

    stalled = store.acquire(thread.workflow_id, owner="worker-stalled")
    clock.advance(120)
    store.acquire(thread.workflow_id, owner="worker-successor")

    with pytest.raises(WorkflowLeaseLost, match="no longer held"):
        store.renew(stalled)


def test_leases_on_different_workflows_do_not_block_each_other(tmp_path):
    """Exclusion is per workflow: two business processes run concurrently."""
    factory = session_factory(tmp_path / "lease.db")
    registry = WorkflowThreadRegistry(factory)
    a = registry.register(thread_id="t-a", workflow_name="job_search")
    b = registry.register(thread_id="t-b", workflow_name="other_process")
    store = WorkflowLeaseStore(factory)

    assert store.acquire(a.workflow_id, owner="worker-a").workflow_id == a.workflow_id
    assert store.acquire(b.workflow_id, owner="worker-b").workflow_id == b.workflow_id
    assert a.workflow_id != b.workflow_id


# --- Construction ------------------------------------------------------------


def test_a_lease_must_name_its_owner(tmp_path):
    """An anonymous lease is one nobody can investigate when it goes stale.

    Surfaces as pydantic's `ValidationError` wrapping the domain error, which is
    how every other field contract in `personalos.domain` reports -- see
    `tests/unit/test_job_search_nodes.py`.
    """
    from pydantic import ValidationError

    from personalos.domain.workflow import WorkflowLease

    with pytest.raises(ValidationError, match="must name its owner"):
        WorkflowLease(
            workflow_id=uuid4(),
            owner="  ",
            token=uuid4(),
            acquired_at=datetime.utcnow(),
            expires_at=datetime.utcnow(),
        )


def test_a_non_positive_ttl_is_rejected_at_construction(tmp_path):
    """A zero or negative TTL is a lease that is expired the moment it is taken."""
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        WorkflowLeaseStore(session_factory(tmp_path / "lease.db"), ttl_seconds=0)


def test_the_lease_table_declares_its_uniqueness_exactly_once():
    """One UNIQUE clause on `workflow_id`, so the ORM and the migration agree.

    Declaring it both on the column and as a named table constraint emits a
    second, anonymous UNIQUE in `create_all` that the migration does not create.
    Harmless in behaviour and not harmless in kind: the schema every test builds
    would differ from the one production runs, which is how a constraint that
    matters gets verified against the wrong database.
    """
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    from personalos.persistence.models import WorkflowLeaseModel

    ddl = str(
        CreateTable(WorkflowLeaseModel.__table__).compile(dialect=postgresql.dialect())
    )

    assert ddl.count("UNIQUE") == 1, ddl
    assert "CONSTRAINT uq_workflow_leases_workflow_id UNIQUE (workflow_id)" in ddl


def test_the_default_owner_identifies_this_process(tmp_path):
    """`host:pid`, which is what someone chasing a stale lease needs."""
    import os

    owner = default_owner()

    assert owner.endswith(f":{os.getpid()}")
    assert len(owner.split(":")[0]) > 0

"""Tests for the outbox_events, event_log, and application_status_view schema.

Covers the acceptance criteria from the schema migration: the three tables
exist with the documented shape, a domain mutation and its outbox event are
written (and rolled back) atomically in one DB transaction, a dispatch
worker can claim a pending outbox row exactly once even under concurrency,
event_log is append-only, and application_status_view is a projection
recomputed from event_log rather than written directly.
"""

import threading
from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from personalos.domain.models import ApplicationStatus, OutboxEventStatus
from personalos.persistence.models import Base, OutboxEventModel, UserModel
from personalos.persistence.repositories import (
    ApplicationRepository,
    ApplicationStatusViewRepository,
    EventLogRepository,
    JobPostingRepository,
    OutboxEventRepository,
)


def _open(db_path):
    """A fresh engine + session over a file-backed SQLite database."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return factory(), engine


def _open_shared(db_path):
    """A session factory + engine over a file-backed SQLite database.

    A generous busy timeout makes two connections hitting the same file at
    once serialize on SQLite's file lock instead of raising "database is
    locked", so a concurrency test can drive two real, independent sessions
    against the same data instead of simulating the race sequentially.
    """
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return factory, engine


def _make_application(session):
    user = UserModel(email=f"candidate-{id(session)}@example.com")
    session.add(user)
    session.commit()
    posting = JobPostingRepository(session).create(
        source="linkedin",
        title="Staff Engineer",
        company="Acme",
        description_hash="a" * 64,
        dedupe_key=f"acme:staff-engineer:{id(session)}",
    )
    return ApplicationRepository(session).create(job_posting_id=posting.id, user_id=user.id)


# ----------------------------------------------------------------------
# Migration shape: tables and the documented indexes/constraints exist.
# ----------------------------------------------------------------------


def test_schema_creates_all_three_tables_with_documented_shape(tmp_path):
    """outbox_events, event_log, and application_status_view all exist with their key columns."""
    _, engine = _open(tmp_path / "outbox.db")
    try:
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert {"outbox_events", "event_log", "application_status_view"} <= table_names

        outbox_columns = {c["name"] for c in inspector.get_columns("outbox_events")}
        assert outbox_columns == {
            "id",
            "type",
            "payload_json",
            "dedupe_key",
            "status",
            "created_at",
            "dispatched_at",
        }
        outbox_indexed_columns = {
            column
            for index in inspector.get_indexes("outbox_events")
            for column in index["column_names"]
        }
        assert "status" in outbox_indexed_columns
        outbox_unique_columns = {
            column
            for uc in inspector.get_unique_constraints("outbox_events")
            for column in uc["column_names"]
        }
        assert "dedupe_key" in outbox_unique_columns

        event_log_columns = {c["name"] for c in inspector.get_columns("event_log")}
        assert {
            "id",
            "aggregate_type",
            "aggregate_id",
            "event_type",
            "payload_json",
            "occurred_at",
            "created_at",
        } <= event_log_columns
        event_log_indexed_columns = {
            column
            for index in inspector.get_indexes("event_log")
            for column in index["column_names"]
        }
        assert "aggregate_id" in event_log_indexed_columns

        view_columns = {c["name"] for c in inspector.get_columns("application_status_view")}
        assert view_columns == {"application_id", "status", "last_event_id", "updated_at"}
    finally:
        engine.dispose()


def test_outbox_event_dedupe_key_is_unique(tmp_path):
    """The schema itself rejects a duplicate dedupe_key insert."""
    session, engine = _open(tmp_path / "outbox.db")
    try:
        repo = OutboxEventRepository(session)
        repo.create(type="application.status_changed", payload={}, dedupe_key="dup-key")
        with pytest.raises(IntegrityError):
            repo.create(type="application.status_changed", payload={}, dedupe_key="dup-key")
    finally:
        session.close()
        engine.dispose()


# ----------------------------------------------------------------------
# Outbox rows are written atomically with the domain mutation.
# ----------------------------------------------------------------------


def test_domain_mutation_and_outbox_event_are_committed_atomically(tmp_path):
    """Staging both writes with commit=False and committing once persists both together."""
    session, engine = _open(tmp_path / "outbox.db")
    try:
        application = _make_application(session)
        app_repo = ApplicationRepository(session)
        outbox_repo = OutboxEventRepository(session)

        app_repo.update_status(application.id, ApplicationStatus.SAVED, commit=False)
        outbox_repo.create(
            type="application.status_changed",
            payload={"application_id": str(application.id), "status": "saved"},
            commit=False,
        )
        session.commit()

        refetched = app_repo.get_by_id(application.id)
        assert refetched.status == ApplicationStatus.SAVED.value
        outbox_rows = session.query(OutboxEventModel).all()
        assert len(outbox_rows) == 1
        assert outbox_rows[0].payload_json["status"] == "saved"
    finally:
        session.close()
        engine.dispose()


def test_rolled_back_transaction_leaves_neither_mutation_nor_outbox_event(tmp_path):
    """If the shared transaction never commits, neither write survives."""
    session, engine = _open(tmp_path / "outbox.db")
    try:
        application = _make_application(session)
        app_repo = ApplicationRepository(session)
        outbox_repo = OutboxEventRepository(session)

        app_repo.update_status(application.id, ApplicationStatus.SAVED, commit=False)
        outbox_repo.create(
            type="application.status_changed",
            payload={"application_id": str(application.id), "status": "saved"},
            commit=False,
        )
        session.rollback()

        refetched = app_repo.get_by_id(application.id)
        assert refetched.status == ApplicationStatus.DISCOVERED.value
        assert session.query(OutboxEventModel).count() == 0
    finally:
        session.close()
        engine.dispose()


# ----------------------------------------------------------------------
# Worker claim semantics: exactly once.
# ----------------------------------------------------------------------


def test_worker_claims_pending_event_and_marks_it_dispatched(tmp_path):
    """The happy path: claim moves PENDING -> IN_PROGRESS, then mark_dispatched finishes it."""
    session, engine = _open(tmp_path / "outbox.db")
    try:
        repo = OutboxEventRepository(session)
        created = repo.create(type="application.status_changed", payload={"x": 1})

        claimed = repo.claim_next()
        assert claimed is not None
        assert claimed.id == created.id
        assert claimed.status == OutboxEventStatus.IN_PROGRESS.value

        dispatched = repo.mark_dispatched(claimed.id)
        assert dispatched.status == OutboxEventStatus.DISPATCHED.value
        assert dispatched.dispatched_at is not None
    finally:
        session.close()
        engine.dispose()


def test_claim_next_returns_none_once_the_only_row_is_already_claimed(tmp_path):
    """A second claim attempt on an already-claimed row finds nothing pending left."""
    session, engine = _open(tmp_path / "outbox.db")
    try:
        repo = OutboxEventRepository(session)
        repo.create(type="application.status_changed", payload={"x": 1})

        first = repo.claim_next()
        assert first is not None

        second = repo.claim_next()
        assert second is None
    finally:
        session.close()
        engine.dispose()


def test_worker_claims_outbox_event_exactly_once_under_concurrency(tmp_path):
    """Two independent sessions racing to claim the same row: exactly one wins.

    Simulates two worker processes polling the same outbox table
    concurrently. `claim_next`'s atomic conditional UPDATE (the SQLite
    equivalent of Postgres's `SELECT ... FOR UPDATE SKIP LOCKED`) must let
    only one of them transition the row out of PENDING.
    """
    factory, engine = _open_shared(tmp_path / "outbox.db")
    try:
        setup_session = factory()
        created = OutboxEventRepository(setup_session).create(
            type="application.status_changed", payload={"x": 1}
        )
        event_id = created.id
        setup_session.close()

        results: list = []
        errors: list = []
        barrier = threading.Barrier(2)

        def worker():
            worker_session = factory()
            try:
                barrier.wait(timeout=5)
                results.append(OutboxEventRepository(worker_session).claim_next())
            except Exception as exc:  # pragma: no cover - surfaced via assertion below
                errors.append(exc)
            finally:
                worker_session.close()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        claimed_rows = [r for r in results if r is not None]
        assert len(claimed_rows) == 1
        assert claimed_rows[0].id == event_id

        check_session = factory()
        try:
            row = check_session.get(OutboxEventModel, event_id)
            assert row.status == OutboxEventStatus.IN_PROGRESS.value
        finally:
            check_session.close()
    finally:
        engine.dispose()


# ----------------------------------------------------------------------
# event_log: append-only.
# ----------------------------------------------------------------------


def test_event_log_is_queryable_by_aggregate_id_oldest_first(tmp_path):
    """Appended events for an aggregate come back in occurrence order."""
    session, engine = _open(tmp_path / "outbox.db")
    try:
        application = _make_application(session)
        repo = EventLogRepository(session)

        repo.append(
            aggregate_type="application",
            aggregate_id=application.id,
            event_type="application.discovered",
            payload={},
            occurred_at=datetime(2026, 9, 10, 9, 0),
        )
        repo.append(
            aggregate_type="application",
            aggregate_id=application.id,
            event_type="application.saved",
            payload={},
            occurred_at=datetime(2026, 9, 10, 10, 0),
        )

        events = repo.get_by_aggregate_id(application.id)
        assert [e.event_type for e in events] == ["application.discovered", "application.saved"]
    finally:
        session.close()
        engine.dispose()


def test_event_log_repository_exposes_no_update_or_delete(tmp_path):
    """The event log cannot be rewritten: only append and reads exist."""
    assert not hasattr(EventLogRepository, "update")
    assert not hasattr(EventLogRepository, "delete")


# ----------------------------------------------------------------------
# application_status_view: a projection, recomputed from event_log.
# ----------------------------------------------------------------------


def test_projection_is_recomputed_not_accumulated(tmp_path):
    """Recomputing the projection twice overwrites the single row, not append to it."""
    session, engine = _open(tmp_path / "outbox.db")
    try:
        application = _make_application(session)
        event_repo = EventLogRepository(session)
        view_repo = ApplicationStatusViewRepository(session)

        first_event = event_repo.append(
            aggregate_type="application",
            aggregate_id=application.id,
            event_type="application.discovered",
            payload={},
        )
        view_repo.recompute(
            application_id=application.id,
            status=ApplicationStatus.DISCOVERED.value,
            last_event_id=first_event.id,
        )

        second_event = event_repo.append(
            aggregate_type="application",
            aggregate_id=application.id,
            event_type="application.saved",
            payload={},
        )
        view_repo.recompute(
            application_id=application.id,
            status=ApplicationStatus.SAVED.value,
            last_event_id=second_event.id,
        )

        view = view_repo.get_by_application_id(application.id)
        assert view.status == ApplicationStatus.SAVED.value
        assert view.last_event_id == second_event.id

        from personalos.persistence.models import ApplicationStatusViewModel

        assert (
            session.query(ApplicationStatusViewModel)
            .filter(ApplicationStatusViewModel.application_id == application.id)
            .count()
            == 1
        )
    finally:
        session.close()
        engine.dispose()

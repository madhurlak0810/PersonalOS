"""Tests for the workflow orchestration persistence schema.

Exercises the tables a LangGraph-based workflow needs to survive a process
restart: a workflow_run must be re-fetchable by workflow_id, and its latest
checkpoint must still be there, using a fresh engine/session over the same
on-disk database rather than the one that wrote the data.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from personalos.persistence.models import Base
from personalos.persistence.repositories import (
    CheckpointRepository,
    WorkflowRepository,
    WorkflowRunRepository,
)


def _open(db_path):
    """A fresh engine + session over a file-backed SQLite database."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return factory(), engine


def test_workflow_run_is_checkpointed_and_re_fetchable_after_a_restart(tmp_path):
    """A workflow_run and its checkpoint survive a simulated process restart."""
    db_path = tmp_path / "workflows.db"

    session, engine = _open(db_path)
    try:
        workflow = WorkflowRepository(session).create(name="job_search")
        run = WorkflowRunRepository(session).create(workflow_id=workflow.id, thread_id="thread-1")
        CheckpointRepository(session).save(
            workflow_id=workflow.id,
            workflow_run_id=run.id,
            thread_id=run.thread_id,
            checkpoint={"step": "search", "state": {"found": 3}},
        )
        workflow_id, run_id = workflow.id, run.id
    finally:
        session.close()
        engine.dispose()

    # Simulate a restart: a new engine and session over the same file, with no
    # in-memory identity map or connection carried over from the write above.
    session, engine = _open(db_path)
    try:
        runs = WorkflowRunRepository(session).get_by_workflow_id(workflow_id)
        assert len(runs) == 1
        assert runs[0].id == run_id
        assert runs[0].thread_id == "thread-1"
        assert runs[0].status == "pending"

        checkpoint = CheckpointRepository(session).get_latest_by_workflow_id(workflow_id)
        assert checkpoint is not None
        assert checkpoint.workflow_run_id == run_id
        assert checkpoint.checkpoint == {"step": "search", "state": {"found": 3}}
    finally:
        session.close()
        engine.dispose()


def test_workflow_run_defaults_to_a_generated_thread_id(tmp_path):
    """A run created without an explicit thread_id still gets a unique one."""
    session, engine = _open(tmp_path / "workflows.db")
    try:
        workflow = WorkflowRepository(session).create(name="job_search")
        run_a = WorkflowRunRepository(session).create(workflow_id=workflow.id)
        run_b = WorkflowRunRepository(session).create(workflow_id=workflow.id)

        assert run_a.thread_id
        assert run_b.thread_id
        assert run_a.thread_id != run_b.thread_id
    finally:
        session.close()
        engine.dispose()

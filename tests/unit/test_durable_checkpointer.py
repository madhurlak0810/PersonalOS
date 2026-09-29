"""Contract tests for the database-backed LangGraph checkpointer.

`tests/graph_scenarios/test_durable_resume.py` proves the checkpointer works
where it counts -- a killed worker resumes at the right step. These tests cover
the saver's own contract, the parts a graph run exercises only incidentally:
that a `put` round-trips through a *fresh* engine, that `get_tuple` means
"latest" when no checkpoint id is named, that writes are kept or overwritten
according to LangGraph's reserved-channel rule, and that a thread nobody bound
to a workflow is refused rather than checkpointed under a guess.

Each test opens its own engine over a file-backed SQLite database, and any test
about durability opens a second one to read with: a read through the engine that
did the writing can be answered from an identity map, which is exactly the thing
a restart does not have.
"""

from datetime import datetime
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint

from personalos.domain.errors import NotFound
from personalos.persistence.checkpointer import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_RUNNING,
    SqlAlchemyCheckpointSaver,
    UnregisteredWorkflowThread,
    WorkflowThreadRegistry,
)
from personalos.persistence.models import CheckpointModel, WorkflowRunModel
from tests.fixtures.durable_workflow import session_factory

WORKFLOW = "job_search"


def _saver(db_path):
    """A registry and saver over their own engine, as one process would have."""
    factory = session_factory(db_path)
    registry = WorkflowThreadRegistry(factory)
    return SqlAlchemyCheckpointSaver(factory, registry), registry, factory


def _checkpoint(step: int, **channel_values) -> Checkpoint:
    """A checkpoint carrying channel values, versioned as LangGraph would."""
    checkpoint = empty_checkpoint()
    checkpoint["id"] = f"1f000000-0000-6000-8000-{step:012d}"
    checkpoint["ts"] = datetime.utcnow().isoformat()
    checkpoint["channel_values"] = dict(channel_values)
    checkpoint["channel_versions"] = dict.fromkeys(channel_values, step + 1)
    return checkpoint


def _config(thread_id: str, checkpoint_id: str | None = None) -> dict:
    configurable = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


# --- Round-tripping ----------------------------------------------------------


def test_a_checkpoint_survives_a_process_restart(tmp_path):
    """A stored checkpoint reads back through an engine that never wrote it."""
    db_path = tmp_path / "cp.db"
    saver, registry, _factory = _saver(db_path)
    registry.register(thread_id="t-1", workflow_name=WORKFLOW)

    saver.put(
        _config("t-1"),
        _checkpoint(0, shortlist=[{"rank": 1}], query="python"),
        {"source": "loop", "step": 0},
        {"shortlist": 1, "query": 1},
    )

    # A second process: new engine, new session, new saver, nothing shared.
    restarted, _registry2, _factory2 = _saver(db_path)
    loaded = restarted.get_tuple(_config("t-1"))

    assert loaded is not None
    assert loaded.checkpoint["channel_values"] == {"shortlist": [{"rank": 1}], "query": "python"}
    assert loaded.metadata["step"] == 0
    assert loaded.config["configurable"]["checkpoint_id"] == _checkpoint(0)["id"]


def test_get_tuple_without_a_checkpoint_id_returns_the_latest(tmp_path):
    """ "Which checkpoint?" defaults to the newest, which is what a resume wants."""
    saver, registry, _factory = _saver(tmp_path / "cp.db")
    registry.register(thread_id="t-1", workflow_name=WORKFLOW)

    for step in range(3):
        saver.put(
            _config("t-1", _checkpoint(step - 1)["id"] if step else None),
            _checkpoint(step, counter=step),
            {"source": "loop", "step": step},
            {"counter": step + 1},
        )

    latest = saver.get_tuple(_config("t-1"))

    assert latest is not None
    assert latest.checkpoint["channel_values"] == {"counter": 2}
    assert latest.metadata["step"] == 2
    # And the parent chain is intact, which is what time-travel reads follow.
    assert latest.parent_config["configurable"]["checkpoint_id"] == _checkpoint(1)["id"]


def test_a_named_checkpoint_is_returned_even_when_a_newer_one_exists(tmp_path):
    """An explicit checkpoint id is honoured, not silently upgraded to the latest."""
    saver, registry, _factory = _saver(tmp_path / "cp.db")
    registry.register(thread_id="t-1", workflow_name=WORKFLOW)
    for step in range(3):
        saver.put(
            _config("t-1"),
            _checkpoint(step, counter=step),
            {"source": "loop", "step": step},
            {"counter": step + 1},
        )

    loaded = saver.get_tuple(_config("t-1", _checkpoint(1)["id"]))

    assert loaded.checkpoint["channel_values"] == {"counter": 1}


def test_an_unknown_thread_reads_back_as_nothing_rather_than_raising(tmp_path):
    """A thread with no checkpoints is `None`: a run that has not started yet."""
    saver, registry, _factory = _saver(tmp_path / "cp.db")
    registry.register(thread_id="t-1", workflow_name=WORKFLOW)

    assert saver.get_tuple(_config("t-1")) is None


def test_threads_do_not_see_each_others_checkpoints(tmp_path):
    """Isolation by `thread_id`, which is what makes two runs independent."""
    saver, registry, _factory = _saver(tmp_path / "cp.db")
    registry.register(thread_id="t-a", workflow_name=WORKFLOW)
    registry.register(thread_id="t-b", workflow_name=WORKFLOW)

    saver.put(_config("t-a"), _checkpoint(0, who="a"), {"source": "loop", "step": 0}, {"who": 1})

    assert saver.get_tuple(_config("t-b")) is None
    assert saver.get_tuple(_config("t-a")).checkpoint["channel_values"] == {"who": "a"}


def test_re_putting_a_checkpoint_id_overwrites_it(tmp_path):
    """A forked or manually updated state replaces its row instead of duplicating it."""
    saver, registry, factory = _saver(tmp_path / "cp.db")
    registry.register(thread_id="t-1", workflow_name=WORKFLOW)

    saver.put(_config("t-1"), _checkpoint(0, v="first"), {"source": "loop", "step": 0}, {"v": 1})
    saver.put(_config("t-1"), _checkpoint(0, v="second"), {"source": "update", "step": 0}, {"v": 2})

    assert saver.get_tuple(_config("t-1")).checkpoint["channel_values"] == {"v": "second"}
    session = factory()
    try:
        assert (
            session.query(CheckpointModel).filter(CheckpointModel.thread_id == "t-1").count() == 1
        )
    finally:
        session.close()


# --- Writes ------------------------------------------------------------------


def test_pending_writes_come_back_with_the_checkpoint_they_belong_to(tmp_path):
    """The writes a finished task produced are what a resumed run must not redo."""
    db_path = tmp_path / "cp.db"
    saver, registry, _factory = _saver(db_path)
    registry.register(thread_id="t-1", workflow_name=WORKFLOW)
    saver.put(_config("t-1"), _checkpoint(0), {"source": "loop", "step": 0}, {})

    saver.put_writes(
        _config("t-1", _checkpoint(0)["id"]),
        [("shortlist", [{"rank": 1}]), ("query", "python")],
        task_id="task-1",
    )

    restarted, _registry2, _factory2 = _saver(db_path)
    loaded = restarted.get_tuple(_config("t-1"))

    assert loaded.pending_writes == [
        ("task-1", "shortlist", [{"rank": 1}]),
        ("task-1", "query", "python"),
    ]


def test_a_replayed_task_does_not_duplicate_its_ordinary_writes(tmp_path):
    """Re-writing the same task's channel writes is idempotent.

    A task that is retried emits the same writes again; keeping the first ones is
    what stops a resumed run from appending a second copy of everything the first
    attempt produced.
    """
    saver, registry, _factory = _saver(tmp_path / "cp.db")
    registry.register(thread_id="t-1", workflow_name=WORKFLOW)
    saver.put(_config("t-1"), _checkpoint(0), {"source": "loop", "step": 0}, {})
    config = _config("t-1", _checkpoint(0)["id"])

    saver.put_writes(config, [("shortlist", ["first"])], task_id="task-1")
    saver.put_writes(config, [("shortlist", ["second"])], task_id="task-1")

    assert saver.get_tuple(_config("t-1")).pending_writes == [("task-1", "shortlist", ["first"])]


def test_a_reserved_channel_write_is_overwritten_not_kept(tmp_path):
    """`__error__` and friends take the newest value, per `WRITES_IDX_MAP`.

    The opposite rule to ordinary channels, and deliberately so: the latest error
    or interrupt is the true one, while the latest copy of a channel write is a
    duplicate.
    """
    saver, registry, _factory = _saver(tmp_path / "cp.db")
    registry.register(thread_id="t-1", workflow_name=WORKFLOW)
    saver.put(_config("t-1"), _checkpoint(0), {"source": "loop", "step": 0}, {})
    config = _config("t-1", _checkpoint(0)["id"])

    saver.put_writes(config, [("__error__", "first failure")], task_id="task-1")
    saver.put_writes(config, [("__error__", "second failure")], task_id="task-1")

    assert saver.get_tuple(_config("t-1")).pending_writes == [
        ("task-1", "__error__", "second failure")
    ]


def test_deleting_a_thread_removes_its_checkpoints_and_writes(tmp_path):
    """Thread deletion is complete, so a deleted run cannot half-resume."""
    saver, registry, _factory = _saver(tmp_path / "cp.db")
    registry.register(thread_id="t-1", workflow_name=WORKFLOW)
    registry.register(thread_id="t-2", workflow_name=WORKFLOW)
    saver.put(_config("t-1"), _checkpoint(0), {"source": "loop", "step": 0}, {})
    saver.put_writes(_config("t-1", _checkpoint(0)["id"]), [("a", 1)], task_id="task-1")
    saver.put(_config("t-2"), _checkpoint(0), {"source": "loop", "step": 0}, {})

    saver.delete_thread("t-1")

    assert saver.get_tuple(_config("t-1")) is None
    assert saver.get_tuple(_config("t-2")) is not None


# --- Listing -----------------------------------------------------------------


def test_list_returns_checkpoints_newest_first(tmp_path):
    saver, registry, _factory = _saver(tmp_path / "cp.db")
    registry.register(thread_id="t-1", workflow_name=WORKFLOW)
    for step in range(3):
        saver.put(
            _config("t-1"),
            _checkpoint(step, counter=step),
            {"source": "loop", "step": step},
            {"counter": step + 1},
        )

    steps = [item.metadata["step"] for item in saver.list(_config("t-1"))]

    assert steps == [2, 1, 0]


def test_list_honours_limit_before_and_a_metadata_filter(tmp_path):
    """The three narrowings LangGraph's history and time-travel reads rely on."""
    saver, registry, _factory = _saver(tmp_path / "cp.db")
    registry.register(thread_id="t-1", workflow_name=WORKFLOW)
    for step in range(4):
        saver.put(
            _config("t-1"),
            _checkpoint(step, counter=step),
            {"source": "update" if step == 2 else "loop", "step": step},
            {"counter": step + 1},
        )

    assert [item.metadata["step"] for item in saver.list(_config("t-1"), limit=2)] == [3, 2]
    assert [
        item.metadata["step"]
        for item in saver.list(_config("t-1"), before=_config("t-1", _checkpoint(2)["id"]))
    ] == [1, 0]
    assert [
        item.metadata["step"] for item in saver.list(_config("t-1"), filter={"source": "update"})
    ] == [2]


# --- Registration is a precondition -----------------------------------------


def test_checkpointing_an_unregistered_thread_is_refused(tmp_path):
    """No workflow binding, no checkpoint.

    The alternative would be a row whose `workflow_id` was invented, which an
    operator resuming that workflow would never find -- so the write is refused
    where the mistake was made.
    """
    saver, _registry, _factory = _saver(tmp_path / "cp.db")

    with pytest.raises(UnregisteredWorkflowThread, match="not bound to a workflow"):
        saver.put(
            _config("never-registered"),
            _checkpoint(0),
            {"source": "loop", "step": 0},
            {},
        )


def test_checkpoints_are_cross_indexed_by_workflow(tmp_path):
    """Every checkpoint carries the workflow and run it belongs to.

    This is what makes "resume workflow X" answerable in SQL, without walking
    threads or deserializing state.
    """
    saver, registry, factory = _saver(tmp_path / "cp.db")
    thread = registry.register(thread_id="t-1", workflow_name=WORKFLOW)
    saver.put(_config("t-1"), _checkpoint(0), {"source": "loop", "step": 0}, {})

    session = factory()
    try:
        row = session.query(CheckpointModel).filter(CheckpointModel.thread_id == "t-1").one()
        assert row.workflow_id == thread.workflow_id
        assert row.workflow_run_id == thread.workflow_run_id
    finally:
        session.close()


# --- The registry ------------------------------------------------------------


def test_registering_the_same_thread_twice_returns_the_same_run(tmp_path):
    """Registration is idempotent, which is what makes a restart rejoin its run."""
    _saver_, registry, factory = _saver(tmp_path / "cp.db")

    first = registry.register(thread_id="t-1", workflow_name=WORKFLOW)
    second = WorkflowThreadRegistry(factory).register(thread_id="t-1", workflow_name=WORKFLOW)

    assert second.workflow_run_id == first.workflow_run_id
    assert second.workflow_id == first.workflow_id


def test_one_workflow_definition_is_reused_across_its_threads(tmp_path):
    """Two threads of one workflow name share the workflow row they hang off."""
    _saver_, registry, _factory = _saver(tmp_path / "cp.db")

    a = registry.register(thread_id="t-a", workflow_name=WORKFLOW)
    b = registry.register(thread_id="t-b", workflow_name=WORKFLOW)

    assert a.workflow_id == b.workflow_id
    assert {thread.thread_id for thread in registry.threads_for_workflow(a.workflow_id)} == {
        "t-a",
        "t-b",
    }


def test_registering_against_a_workflow_that_does_not_exist_is_refused(tmp_path):
    """An explicit `workflow_id` must name a real workflow, not create one."""
    _saver_, registry, _factory = _saver(tmp_path / "cp.db")

    with pytest.raises(NotFound, match="does not exist"):
        registry.register(thread_id="t-1", workflow_name=WORKFLOW, workflow_id=uuid4())


def test_run_status_transitions_are_recorded_with_their_timestamps(tmp_path):
    """`running` then `completed`, with `started_at` set once and `completed_at` set."""
    _saver_, registry, factory = _saver(tmp_path / "cp.db")
    registry.register(thread_id="t-1", workflow_name=WORKFLOW)

    registry.mark_status("t-1", RUN_STATUS_RUNNING, started=True)
    session = factory()
    try:
        run = session.query(WorkflowRunModel).filter(WorkflowRunModel.thread_id == "t-1").one()
        assert run.status == RUN_STATUS_RUNNING
        started_at = run.started_at
        assert started_at is not None
        assert run.completed_at is None
    finally:
        session.close()

    registry.mark_status("t-1", RUN_STATUS_RUNNING, started=True)
    registry.mark_status("t-1", RUN_STATUS_COMPLETED, finished=True)

    session = factory()
    try:
        run = session.query(WorkflowRunModel).filter(WorkflowRunModel.thread_id == "t-1").one()
        assert run.status == RUN_STATUS_COMPLETED
        # Re-marking `started` does not move the original start time.
        assert run.started_at == started_at
        assert run.completed_at is not None
    finally:
        session.close()


def test_marking_the_status_of_an_unregistered_thread_is_refused(tmp_path):
    _saver_, registry, _factory = _saver(tmp_path / "cp.db")

    with pytest.raises(UnregisteredWorkflowThread, match="unregistered thread"):
        registry.mark_status("nope", RUN_STATUS_RUNNING)

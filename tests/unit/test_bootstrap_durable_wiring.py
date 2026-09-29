"""The composition root's durable-orchestration wiring.

Each builder here is one line of construction, which is exactly why they are
worth testing: they are the *only* place the durable implementations are chosen
over the in-memory defaults the graphs fall back to. A builder that quietly
handed back an `InMemorySaver` would leave every graph compiling, every test
passing, and every workflow unresumable after a restart.

The derive-and-register pair gets the most attention, because it is the one that
has to hold across processes: the same candidate must always produce the same
thread id, and registering it twice must rejoin one run rather than fork two.
"""

from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from personalos.bootstrap import (
    JOB_SEARCH_WORKFLOW,
    SUPERVISOR_WORKFLOW,
    build_durable_checkpointer,
    build_journaled_action_executor,
    build_workflow_lease_store,
    build_workflow_thread_registry,
    register_job_search_thread,
    register_supervisor_thread,
)
from personalos.domain.workflow import job_search_thread_id, supervisor_thread_id
from personalos.persistence.action_journal import JournaledActionExecutor
from personalos.persistence.checkpointer import (
    SqlAlchemyCheckpointSaver,
    WorkflowThreadRegistry,
)
from personalos.persistence.leases import WorkflowLeaseStore
from tests.fixtures.durable_workflow import session_factory

USER = uuid4()


def test_the_built_checkpointer_is_the_durable_one_not_the_in_memory_default():
    """The whole point of the builder: durable is chosen here, once."""
    factory = session_factory(":memory:", create=False)

    checkpointer = build_durable_checkpointer(factory)

    assert isinstance(checkpointer, SqlAlchemyCheckpointSaver)
    assert not isinstance(checkpointer, InMemorySaver)
    assert isinstance(checkpointer.registry, WorkflowThreadRegistry)


def test_a_shared_registry_is_reused_rather_than_rebuilt(tmp_path):
    """One registry across the graphs, so its thread cache is shared.

    Also the reason it is a parameter at all: a checkpointer that built its own
    would re-query `workflow_runs` on every super-step of every graph.
    """
    factory = session_factory(tmp_path / "boot.db")
    registry = build_workflow_thread_registry(factory)

    checkpointer = build_durable_checkpointer(factory, registry)

    assert checkpointer.registry is registry


def test_the_built_lease_store_uses_the_default_ttl_unless_told_otherwise(tmp_path):
    factory = session_factory(tmp_path / "boot.db")

    default = build_workflow_lease_store(factory)
    shorter = build_workflow_lease_store(factory, ttl_seconds=30)

    assert isinstance(default, WorkflowLeaseStore)
    assert default.ttl.total_seconds() == 300
    assert shorter.ttl.total_seconds() == 30


def test_an_action_executor_is_wrapped_in_the_journal(tmp_path):
    """Every executor a real deployment hands a graph goes through the journal."""

    class Inner:
        async def execute(self, intent, decision):  # pragma: no cover - not called here
            raise AssertionError("not reached")

    factory = session_factory(tmp_path / "boot.db")
    inner = Inner()
    workflow_id = uuid4()

    wrapped = build_journaled_action_executor(inner, factory, workflow_id=workflow_id)

    assert isinstance(wrapped, JournaledActionExecutor)
    assert wrapped.inner is inner
    assert wrapped.workflow_id == workflow_id


# --- Derive and register together --------------------------------------------


def test_the_same_candidate_always_registers_the_same_job_search_thread(tmp_path):
    """Recomputable from the candidate alone, which is what a restart has."""
    registry = build_workflow_thread_registry(session_factory(tmp_path / "boot.db"))

    first = register_job_search_thread(user_id=USER, registry=registry)
    # A second process, a fresh registry over the same database.
    second = register_job_search_thread(
        user_id=USER,
        registry=build_workflow_thread_registry(session_factory(tmp_path / "boot.db")),
    )

    assert second.thread_id == first.thread_id == job_search_thread_id(USER)
    assert second.workflow_run_id == first.workflow_run_id
    assert first.workflow_name == JOB_SEARCH_WORKFLOW


def test_a_search_key_separates_two_concurrent_searches_for_one_candidate(tmp_path):
    """Two searches, two threads -- and two runs, so neither resumes the other."""
    registry = build_workflow_thread_registry(session_factory(tmp_path / "boot.db"))

    first = register_job_search_thread(user_id=USER, registry=registry, search_key="backend")
    second = register_job_search_thread(user_id=USER, registry=registry, search_key="platform")

    assert first.thread_id != second.thread_id
    assert first.workflow_run_id != second.workflow_run_id


def test_a_conversation_can_be_registered_into_an_existing_workflow(tmp_path):
    """A Supervisor thread and the domain run it delegates to are one process.

    Passing the job search's `workflow_id` is what makes them one resumable
    business process with two threads instead of two unrelated ones.
    """
    registry = build_workflow_thread_registry(session_factory(tmp_path / "boot.db"))
    job = register_job_search_thread(user_id=USER, registry=registry)

    conversation = register_supervisor_thread(
        conversation_key="conv-1", registry=registry, workflow_id=job.workflow_id
    )

    assert conversation.workflow_id == job.workflow_id
    assert conversation.thread_id != job.thread_id
    assert conversation.thread_id == supervisor_thread_id("conv-1")
    assert {thread.thread_id for thread in registry.threads_for_workflow(job.workflow_id)} == {
        job.thread_id,
        conversation.thread_id,
    }


def test_a_conversation_registered_on_its_own_gets_the_supervisor_workflow(tmp_path):
    """Without a workflow_id, a conversation is its own process."""
    registry = build_workflow_thread_registry(session_factory(tmp_path / "boot.db"))

    job = register_job_search_thread(user_id=USER, registry=registry)
    conversation = register_supervisor_thread(conversation_key="conv-1", registry=registry)

    assert conversation.workflow_name == SUPERVISOR_WORKFLOW
    assert conversation.workflow_id != job.workflow_id


def test_the_subgraph_runners_default_thread_is_the_one_bootstrap_registers(tmp_path):
    """The runner's default and the registered id must be the same string.

    They were not, once: the registration passed the search key and the runner's
    default did not, so a runner built without an explicit `thread_id` ran on a
    thread nobody had bound to a workflow -- which the durable saver refuses, and
    which an in-memory saver happily accepts. Both now go through
    `job_search_thread_id`, and this pins that they agree.
    """
    from personalos.graphs.job_search import JobSearchSubgraphRunner

    registry = build_workflow_thread_registry(session_factory(tmp_path / "boot.db"))
    registered = register_job_search_thread(user_id=USER, registry=registry)

    runner = JobSearchSubgraphRunner(object(), user_id=USER)

    assert runner.thread_id == registered.thread_id
    assert registry.resolve(runner.thread_id) is not None

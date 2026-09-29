"""Contract tests for durable workflow identity.

Two identifiers with different jobs, and the tests here are about keeping them
apart: a `thread_id` that is *stable* (so a restarted process can recompute it
rather than having to have kept it) and a `workflow_id` that names the business
process a thread belongs to.

The property that makes the whole resume path work is derivation: the same facts
must always produce the same thread id, in this process and in the one that
replaces it. Everything else here -- the length bound, the reserved separator --
exists so a derived id cannot be quietly rejected later, half way through a run,
by a column it does not fit.
"""

from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from personalos.domain.workflow import (
    MAX_THREAD_ID_LENGTH,
    InvalidWorkflowIdentity,
    WorkflowLease,
    WorkflowResumeState,
    WorkflowThread,
    derive_thread_id,
)

USER = UUID("11111111-1111-1111-1111-111111111111")


# --- Deriving a stable thread id ---------------------------------------------


def test_the_same_facts_always_derive_the_same_thread_id():
    """The point of deriving one: a restart recomputes it instead of recalling it."""
    assert derive_thread_id("job_search", USER) == derive_thread_id("job_search", USER)


def test_different_facts_derive_different_thread_ids():
    assert derive_thread_id("job_search", USER) != derive_thread_id("job_search", uuid4())
    assert derive_thread_id("job_search", USER) != derive_thread_id("job_search", USER, "second")


def test_the_namespace_stays_readable_in_the_derived_id():
    """So a thread id is greppable in a log line rather than an opaque hash."""
    thread_id = derive_thread_id("job_search", USER)

    assert thread_id.startswith("job_search:")
    assert len(thread_id) <= MAX_THREAD_ID_LENGTH


def test_an_over_long_part_does_not_produce_an_over_long_thread_id():
    """The varying parts are hashed, so no input can overflow the column.

    A thread id assembled by concatenation would be rejected by the database
    mid-run, and only for the users whose inputs happened to be long.
    """
    thread_id = derive_thread_id("job_search", USER, "x" * 10_000)

    assert len(thread_id) <= MAX_THREAD_ID_LENGTH


def test_parts_are_separated_so_they_cannot_be_confused_with_each_other():
    """`("ab", "c")` and `("a", "bc")` are different threads, and derive differently."""
    assert derive_thread_id("job_search", "ab", "c") != derive_thread_id("job_search", "a", "bc")


def test_a_blank_or_colon_bearing_namespace_is_refused():
    """The colon is the separator, so a namespace containing one is ambiguous."""
    with pytest.raises(InvalidWorkflowIdentity, match="must not be blank"):
        derive_thread_id("   ", USER)
    with pytest.raises(InvalidWorkflowIdentity, match="must not contain ':'"):
        derive_thread_id("job:search", USER)


# --- WorkflowThread ----------------------------------------------------------


def test_a_thread_carries_the_workflow_it_belongs_to_into_the_graph_config():
    """`config()` is what binds an invocation to both identifiers at once."""
    workflow_id = uuid4()
    thread = WorkflowThread(workflow_id=workflow_id, thread_id="job_search:abc")

    assert thread.config() == {
        "configurable": {"thread_id": "job_search:abc", "workflow_id": str(workflow_id)}
    }


def test_extra_configurable_values_are_merged_into_the_config():
    thread = WorkflowThread(workflow_id=uuid4(), thread_id="job_search:abc")

    config = thread.config(actor_id="user-7")

    assert config["configurable"]["actor_id"] == "user-7"
    assert config["configurable"]["thread_id"] == "job_search:abc"


def test_a_blank_or_over_long_thread_id_is_refused():
    """Refused where it is constructed, not on the first checkpoint write."""
    with pytest.raises(ValidationError, match="must not be blank"):
        WorkflowThread(workflow_id=uuid4(), thread_id="   ")
    with pytest.raises(ValidationError, match="the maximum is"):
        WorkflowThread(workflow_id=uuid4(), thread_id="x" * (MAX_THREAD_ID_LENGTH + 1))


def test_a_thread_is_immutable_and_closed():
    """A resume path that could rewrite its workflow_id could lease one process
    and run another."""
    thread = WorkflowThread(workflow_id=uuid4(), thread_id="job_search:abc")

    with pytest.raises(ValidationError):
        thread.workflow_id = uuid4()
    with pytest.raises(ValidationError):
        WorkflowThread(workflow_id=uuid4(), thread_id="t", unexpected="value")


# --- WorkflowLease -----------------------------------------------------------


def test_a_lease_knows_when_it_has_expired():
    now = datetime(2026, 9, 29, 12, 0, 0)
    lease = WorkflowLease(
        workflow_id=uuid4(),
        owner="worker-a",
        token=uuid4(),
        acquired_at=now,
        expires_at=now + timedelta(seconds=60),
    )

    assert lease.is_expired(now + timedelta(seconds=59)) is False
    assert lease.is_expired(now + timedelta(seconds=60)) is True


# --- WorkflowResumeState -----------------------------------------------------


def test_a_thread_is_resumable_only_with_both_stored_state_and_a_step_left():
    """The two ways "resume" is not the right verb, kept distinct.

    No checkpoint means the run has to be *started*. No next step means it has
    already finished. Collapsing either into "resumable" would re-run work.
    """
    thread = WorkflowThread(workflow_id=uuid4(), thread_id="job_search:abc")

    assert not WorkflowResumeState(thread=thread).is_resumable
    assert not WorkflowResumeState(thread=thread, checkpoint_id="cp-1").is_resumable
    assert not WorkflowResumeState(thread=thread, next=("approval",)).is_resumable
    assert WorkflowResumeState(thread=thread, checkpoint_id="cp-1", next=("approval",)).is_resumable

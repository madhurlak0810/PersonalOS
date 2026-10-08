"""Durable workflow identity: what a run is resumable *by*.

Two identifiers, deliberately distinct, because they answer different
questions:

- **`thread_id`** names one conversational/orchestration thread. It is the key
  LangGraph's checkpointer stores and loads state under, so it has to be
  *stable*: a thread id minted fresh on every invocation produces a graph that
  checkpoints diligently and can never resume, because nothing ever looks the
  state up again. `derive_thread_id` exists so a caller can recompute the same
  id from the same facts after a process restart instead of having to have kept
  it.
- **`workflow_id`** names one long-running business process -- one pursuit of
  one job search, which may span many threads (the Supervisor's conversation
  thread, the Job Search subgraph's own thread) and many process lifetimes. It
  is what an operator resumes, what leases are taken against, and what
  `checkpoints.workflow_id` cross-indexes.

`WorkflowThread` is the pair, and it is what everything that resumes work
passes around: `personalos.persistence.checkpointer` resolves a `thread_id` to
its workflow through it, and `personalos.persistence.leases` takes a lease on
the `workflow_id` so two workers cannot resume the same process at once.

This module is `domain`: it holds the identity and its invariants and knows
nothing about how either is stored.
"""

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from personalos.domain.errors import ValidationFailed

#: Upper bound on a thread id, matching `workflow_runs.thread_id` and
#: `checkpoints.thread_id` (`String(255)`). Enforced here rather than left to
#: the database so an over-long id fails where it is constructed, not on the
#: first checkpoint write half way through a run.
MAX_THREAD_ID_LENGTH = 255

#: How many hex characters of the digest a derived thread id carries. 32 is
#: half a SHA-256 and leaves collisions well below anything a single user's
#: thread count makes reachable, while keeping the id short enough to read in
#: a log line.
_DIGEST_CHARS = 32


class InvalidWorkflowIdentity(ValidationFailed, ValueError):
    """A workflow/thread identifier violated its contract.

    Subclasses both `ValidationFailed` (so it reports through the shared error
    taxonomy) and `ValueError`, matching `JobSearchContractError` and
    `InvalidIdempotencyKey` elsewhere in `personalos.domain`.
    """


def derive_thread_id(namespace: str, *parts: Any) -> str:
    """Build a stable thread id from the facts that identify the thread.

    Deterministic by construction: the same `namespace` and `parts` always
    yield the same id, in this process and in the one that restarts after a
    crash. That is the whole point -- a caller that can name *what* the thread
    is about does not need to have persisted the id to resume it.

    The readable namespace is kept as a prefix so a thread id is greppable in
    logs (`job_search:...`), and the varying parts are hashed rather than
    concatenated so neither an over-long part nor a part containing a colon can
    change the shape of the id.
    """
    if not namespace.strip():
        raise InvalidWorkflowIdentity("a thread id namespace must not be blank")
    if ":" in namespace:
        raise InvalidWorkflowIdentity(
            f"thread id namespace '{namespace}' must not contain ':'; it is the "
            f"separator between the namespace and the derived digest"
        )
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()[
        :_DIGEST_CHARS
    ]
    return f"{namespace.strip()}:{digest}"


#: Thread-id namespaces, one per kind of thread. Kept here with the derivation
#: helpers below rather than spelled at each call site: a namespace typo is a
#: thread nothing ever resumes, and it would not fail anywhere -- the run would
#: simply start fresh every time.
JOB_SEARCH_THREAD_NAMESPACE = "job_search"
SUPERVISOR_THREAD_NAMESPACE = "supervisor"
RECRUITER_INBOX_THREAD_NAMESPACE = "recruiter_inbox"


def thread_namespace(thread_id: str) -> str | None:
    """The kind of thread a derived id names -- `job_search`, `supervisor`, ...

    The only per-thread record of which graph a thread runs on. The workflow's
    name cannot answer it: a Supervisor conversation and the Job Search run it
    delegated to share one workflow, and so one name. `None` for an id that was
    not derived.
    """
    namespace, separator, _ = thread_id.partition(":")
    return namespace if separator and namespace else None


def job_search_thread_id(user_id: Any, search_key: str | None = None) -> str:
    """The thread id for one candidate's job search.

    The single definition of that recipe, because two places need to agree on it
    exactly: whatever registers the thread, and whatever runs on it. Two callers
    each deriving "the obvious way" -- one passing the search key, one omitting
    it -- produce different ids for the same search, and the one that runs finds
    no state to resume. Both go through here instead.

    `search_key` distinguishes concurrent searches for one candidate; omitted, a
    candidate has one long-running job search, which is the shape a single-user
    build actually has.
    """
    return derive_thread_id(JOB_SEARCH_THREAD_NAMESPACE, user_id, search_key or "")


def recruiter_inbox_thread_id(user_id: Any) -> str:
    """The thread id inbound recruiter mail for one candidate is processed on.

    Its own thread rather than the job search's: a batch of inbound messages is
    about whichever applications it correlates to, and running it on a search
    thread would overwrite the pending actions of a run parked at an approval.
    """
    return derive_thread_id(RECRUITER_INBOX_THREAD_NAMESPACE, user_id)


def supervisor_thread_id(conversation_key: Any) -> str:
    """The thread id for one Supervisor conversation."""
    return derive_thread_id(SUPERVISOR_THREAD_NAMESPACE, conversation_key)


class WorkflowThread(BaseModel):
    """One resumable thread, bound to the business process it belongs to.

    Immutable and closed, like the other values passed between layers: a
    resume path that could rewrite the `workflow_id` it was handed would be a
    resume path that can lease one process and run another.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: UUID
    thread_id: str
    #: The `workflow_runs` row this thread's checkpoints hang off, once one
    #: exists. `None` before the thread has been registered, which is why
    #: `personalos.persistence.checkpointer` refuses to checkpoint a thread it
    #: cannot resolve to a run.
    workflow_run_id: UUID | None = None
    #: The workflow *definition* name (`workflows.name`), carried for logging
    #: and for the registry's get-or-create; not part of the identity.
    workflow_name: str | None = None

    @field_validator("thread_id")
    @classmethod
    def _check_thread_id(cls, value: str) -> str:
        if not value.strip():
            raise InvalidWorkflowIdentity("thread_id must not be blank")
        if len(value) > MAX_THREAD_ID_LENGTH:
            raise InvalidWorkflowIdentity(
                f"thread_id is {len(value)} characters; the maximum is " f"{MAX_THREAD_ID_LENGTH}"
            )
        return value

    def config(self, **configurable: Any) -> dict[str, Any]:
        """The LangGraph invocation config that binds a run to this thread.

        Returned as a plain dict rather than a `RunnableConfig`: `domain`
        imports nothing, and a `RunnableConfig` *is* a `TypedDict` over
        exactly this shape, so callers can pass the result straight to
        `ainvoke(..., config=...)`.

        `workflow_id` is carried in `configurable` alongside `thread_id` so a
        node or a checkpointer that has only the config still knows which
        business process it is part of.
        """
        return {
            "configurable": {
                "thread_id": self.thread_id,
                "workflow_id": str(self.workflow_id),
                **configurable,
            }
        }


class WorkflowLease(BaseModel):
    """An exclusive, expiring claim on one workflow, held while it runs.

    What makes concurrent resumes safe: a resume acquires the lease for its
    `workflow_id` before it touches the checkpointer, so a second worker that
    wants the same process is refused rather than replaying the same steps
    against the same state.

    `token` is a fencing token, not a formality. A worker that stalled past
    `expires_at`, had its lease taken over, and then woke up still holds a
    lease object -- one whose token no longer matches the row, so its release
    and renewal are rejected instead of cancelling the lease its successor is
    relying on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: UUID
    owner: str
    token: UUID
    acquired_at: datetime
    expires_at: datetime
    thread_id: str | None = None

    @field_validator("owner")
    @classmethod
    def _owner_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise InvalidWorkflowIdentity(
                "a lease must name its owner; it is what identifies the holder in a "
                "stale-lease investigation"
            )
        return value

    def is_expired(self, now: datetime | None = None) -> bool:
        """True once this lease's hold may be taken over by another worker."""
        return (now or datetime.utcnow()) >= self.expires_at


class WorkflowResumeState(BaseModel):
    """What a stored thread says about where its run stopped.

    The answer to "resume from the correct step, not from scratch": `next` is
    the node(s) the checkpoint says run next, so a caller (or a test) can
    assert where a restarted worker picks up without having to infer it from
    side effects.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread: WorkflowThread
    checkpoint_id: str | None = None
    next: tuple[str, ...] = Field(default_factory=tuple)
    values: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_resumable(self) -> bool:
        """True when there is stored state *and* a step left to run."""
        return self.checkpoint_id is not None and bool(self.next)


# --- Workflow status ------------------------------------------------------------
#
# What `GET /v1/workflows/{id}` reports, and what the resume endpoint decides
# against. Everything below is assembled from stored rows by
# `personalos.persistence.workflow_status` -- no process holds it -- so any API
# instance can answer for any workflow, including one started before it booted.


class WorkflowStatus(str, Enum):
    """Where one workflow stands, across all of its threads.

    Wire-visible: a client matches on these strings.
    """

    #: Registered, with nothing queued and nothing run yet.
    PENDING = "pending"
    #: Accepted and handed to the worker; no worker has finished picking it up.
    QUEUED = "queued"
    #: A worker is executing one of its threads.
    RUNNING = "running"
    #: Parked on an interrupt -- an approval, an awaited event -- with no answer
    #: queued. The only status the resume endpoint accepts.
    WAITING = "waiting"
    #: A thread raised and was left failed. Its last checkpoint is intact.
    FAILED = "failed"
    #: Every thread ran to the end.
    COMPLETED = "completed"


class WorkflowCommandKind(str, Enum):
    """What the API asks the worker to do with a thread."""

    START = "start"
    RESUME = "resume"


class WorkflowCommand(BaseModel):
    """One unit of work the API hands to the worker instead of running it inline.

    Carries the caller's identity rather than relying on the worker to infer
    it: `actor_id` and `correlation_id` came off the request that queued this,
    and the worker passes both into the run's config so every checkpoint, log
    line and tool call it produces traces back to that request.

    `graph_input` is the initial state for `START`, and the value an interrupt
    is resumed with for `RESUME`. `interrupt_ids` names the interrupt(s) a
    resume answers, so a second answer to the same question is refused rather
    than queued behind the first.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    command_id: UUID
    kind: WorkflowCommandKind
    workflow_id: UUID
    thread_id: str
    actor_id: str
    correlation_id: UUID
    graph_input: Any = None
    interrupt_ids: tuple[str, ...] = Field(default_factory=tuple)
    created_at: datetime | None = None


class PendingInterrupt(BaseModel):
    """One question a parked thread is waiting to have answered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
    #: The node that raised it, when the stored task path names one.
    step: str | None = None
    interrupt_id: str | None = None
    #: The interrupt's payload as the graph raised it -- for an approval, the
    #: `ApprovalRequest`s the reviewer is being asked about.
    value: Any = None


class RecoverableFailure(BaseModel):
    """A failure the stored state can be resumed past.

    Only failures with a checkpoint behind them are reported here: a node that
    raised leaves the checkpoint before it intact, so the thread can be resumed
    at that node rather than restarted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
    step: str | None = None
    message: str


class WorkflowStep(BaseModel):
    """One graph node, on the thread it ran (or will run) on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
    step: str


class ThreadSnapshot(BaseModel):
    """What the stored rows say about one thread of a workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
    #: Which graph the thread runs on: its id's namespace (`thread_namespace`).
    kind: str | None = None
    #: `workflow_runs.status`, verbatim. A thread parked at an interrupt reads
    #: `running` here -- the runner does not mark it finished -- which is why
    #: `interrupts` is consulted before this.
    run_status: str
    checkpoint_id: str | None = None
    #: Nodes that have run, in the order each first finished.
    completed_steps: tuple[str, ...] = Field(default_factory=tuple)
    #: Nodes the last checkpoint says run next.
    next_steps: tuple[str, ...] = Field(default_factory=tuple)
    interrupts: tuple[PendingInterrupt, ...] = Field(default_factory=tuple)
    failures: tuple[RecoverableFailure, ...] = Field(default_factory=tuple)
    actor_id: str | None = None
    correlation_id: UUID | None = None
    updated_at: datetime | None = None


class WorkflowSnapshot(BaseModel):
    """A workflow's threads and queued commands, and the status they add up to.

    The status is derived here rather than stored, because nothing could keep
    a stored copy honest: it depends on the run rows, the checkpoints and the
    command queue, which are written by different processes at different
    times.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: UUID
    name: str | None = None
    threads: tuple[ThreadSnapshot, ...] = Field(default_factory=tuple)
    #: Commands handed to the worker that it has not finished dispatching.
    queued: tuple[WorkflowCommand, ...] = Field(default_factory=tuple)
    created_at: datetime | None = None

    @property
    def pending_interrupts(self) -> tuple[PendingInterrupt, ...]:
        """Interrupts still waiting on an answer nobody has queued.

        An interrupt whose answer is already queued is not pending any more --
        reporting it would invite a second answer to a question already
        answered.
        """
        answered = {
            interrupt_id
            for command in self.queued
            if command.kind == WorkflowCommandKind.RESUME
            for interrupt_id in command.interrupt_ids
        }
        resumed_threads = {
            command.thread_id
            for command in self.queued
            if command.kind == WorkflowCommandKind.RESUME and not command.interrupt_ids
        }
        return tuple(
            interrupt
            for thread in self.threads
            if thread.thread_id not in resumed_threads
            for interrupt in thread.interrupts
            if interrupt.interrupt_id not in answered
        )

    @property
    def status(self) -> WorkflowStatus:
        """The workflow's status. Precedence is the whole rule.

        An unanswered interrupt beats everything, because it is the one state a
        human has to act on. A thread actually executing beats queued work.
        Queued work beats a failure, because what is queued may be the retry.
        """
        if self.pending_interrupts:
            return WorkflowStatus.WAITING
        if any(thread.run_status == "running" and not thread.interrupts for thread in self.threads):
            return WorkflowStatus.RUNNING
        if self.queued:
            return WorkflowStatus.QUEUED
        if any(thread.run_status == "failed" for thread in self.threads):
            return WorkflowStatus.FAILED
        if self.threads and all(thread.run_status == "completed" for thread in self.threads):
            return WorkflowStatus.COMPLETED
        return WorkflowStatus.PENDING

    @property
    def active_thread(self) -> ThreadSnapshot | None:
        """The thread whose position is the workflow's position.

        The one waiting on input if any, else the most recently updated thread
        that has not finished, else the most recently updated thread at all.
        """
        waiting = {interrupt.thread_id for interrupt in self.pending_interrupts}
        for thread in self.threads:
            if thread.thread_id in waiting:
                return thread
        unfinished = [thread for thread in self.threads if thread.run_status != "completed"]
        candidates = unfinished or list(self.threads)
        if not candidates:
            return None
        return max(candidates, key=lambda thread: thread.updated_at or datetime.min)

    @property
    def current_step(self) -> WorkflowStep | None:
        """The step the workflow is at: interrupted, failed in, or about to run."""
        thread = self.active_thread
        if thread is None:
            return None
        for interrupt in thread.interrupts:
            if interrupt.step:
                return WorkflowStep(thread_id=thread.thread_id, step=interrupt.step)
        for failure in thread.failures:
            if failure.step:
                return WorkflowStep(thread_id=thread.thread_id, step=failure.step)
        if thread.next_steps:
            return WorkflowStep(thread_id=thread.thread_id, step=thread.next_steps[0])
        return None

    @property
    def completed_steps(self) -> tuple[WorkflowStep, ...]:
        """Every finished node across the workflow's threads, thread by thread."""
        return tuple(
            WorkflowStep(thread_id=thread.thread_id, step=step)
            for thread in self.threads
            for step in thread.completed_steps
        )

    @property
    def recoverable_failures(self) -> tuple[RecoverableFailure, ...]:
        """Every failure a resume could get past, across the workflow's threads."""
        return tuple(failure for thread in self.threads for failure in thread.failures)


__all__ = [
    "MAX_THREAD_ID_LENGTH",
    "JOB_SEARCH_THREAD_NAMESPACE",
    "SUPERVISOR_THREAD_NAMESPACE",
    "RECRUITER_INBOX_THREAD_NAMESPACE",
    "InvalidWorkflowIdentity",
    "derive_thread_id",
    "thread_namespace",
    "job_search_thread_id",
    "recruiter_inbox_thread_id",
    "supervisor_thread_id",
    "WorkflowThread",
    "WorkflowLease",
    "WorkflowResumeState",
    "WorkflowStatus",
    "WorkflowCommandKind",
    "WorkflowCommand",
    "PendingInterrupt",
    "RecoverableFailure",
    "WorkflowStep",
    "ThreadSnapshot",
    "WorkflowSnapshot",
]

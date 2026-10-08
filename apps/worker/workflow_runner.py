"""Runs and resumes durable workflows: the process that survives its own death.

A durably checkpointed graph is only half of resumability. Something has to
decide *which* workflow to pick up, prove no one else is already on it, invoke
the graph in a way that continues from the stored checkpoint rather than
restarting, and record how it went. That is this module, and it is part of the
composition root: it owns a session factory, wires the durable checkpointer, and
is therefore allowed to know about every layer.

Resuming is `ainvoke(None, config)`, not `ainvoke(initial_state, config)`. With
no input, LangGraph loads the thread's last checkpoint and runs the tasks that
checkpoint says are still pending -- which is why a worker killed inside the
approval step comes back at the approval step, with the shortlist it had already
computed intact, rather than searching the job boards again. Passing the initial
state instead would re-apply the run's inputs on top of the restored state,
which is what starting looks like, not resuming.

Every start and every resume is wrapped in a lease on the workflow
(`personalos.persistence.leases`). Without it, two workers noticing the same
stalled workflow would both load the same checkpoint and both run the next step,
and that step is allowed to have side effects -- so the lease is not an
optimization, it is the thing that stops a resume from becoming a duplicate.
"""

import logging
from collections.abc import Mapping
from typing import Any
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph

from personalos.domain.errors import ErrorCode, NotFound, PersonalOSError
from personalos.domain.workflow import WorkflowResumeState, WorkflowThread
from personalos.persistence.checkpointer import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    WorkflowThreadRegistry,
)
from personalos.persistence.leases import WorkflowLeaseStore, default_owner

logger = logging.getLogger(__name__)


class NothingToResume(PersonalOSError):
    """The workflow has no stored state to resume from.

    Distinguished from "finished": a thread that was never checkpointed has to
    be *started*, and silently starting it in response to a resume request would
    re-run a workflow an operator believed was already part way through.
    """

    code = ErrorCode.VALIDATION
    http_status = 409
    default_message = "workflow has no checkpoint to resume from"


class AmbiguousWorkflowResume(PersonalOSError):
    """A workflow has several threads and none was named.

    Refused rather than guessed at: picking one of a conversation thread and a
    domain-subgraph thread on the caller's behalf is picking which half of the
    process to advance.
    """

    code = ErrorCode.VALIDATION
    http_status = 409
    default_message = "workflow has several threads; name the one to resume"


class DurableWorkflowRunner:
    """Starts and resumes one compiled graph's threads, under a workflow lease.

    Holds the compiled graph rather than building it: which ports a graph is
    wired to is a deployment decision made where the graph is constructed, and a
    runner that built its own would have to know all of them. What it does own is
    the three things every durable run needs regardless of which graph it is --
    the thread registry, the lease store, and the identity of this worker.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        registry: WorkflowThreadRegistry,
        leases: WorkflowLeaseStore,
        owner: str | None = None,
    ):
        """Wire the graph to the registry and lease store that make it resumable."""
        if graph is None:
            raise ValueError("DurableWorkflowRunner requires a compiled graph")
        if registry is None:
            raise ValueError("DurableWorkflowRunner requires a WorkflowThreadRegistry")
        if leases is None:
            raise ValueError("DurableWorkflowRunner requires a WorkflowLeaseStore")
        self.graph = graph
        self.registry = registry
        self.leases = leases
        self.owner = owner or default_owner()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    async def inspect(self, thread: WorkflowThread) -> WorkflowResumeState:
        """Report where a thread stopped, without running anything.

        `next` is the answer to "would a resume pick up at the right step?", read
        from the checkpoint itself rather than inferred, so an operator (or a
        test) can check it before committing to a resume.
        """
        snapshot = await self.graph.aget_state(thread.config())
        configurable = (snapshot.config or {}).get("configurable") or {}
        return WorkflowResumeState(
            thread=thread,
            checkpoint_id=configurable.get("checkpoint_id"),
            next=tuple(snapshot.next or ()),
            values=dict(snapshot.values or {}),
        )

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    async def start(
        self,
        thread: WorkflowThread,
        initial_state: dict[str, Any],
        *,
        configurable: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a registered thread from its beginning, holding the workflow's lease.

        `configurable` is merged into the run's config next to the thread's own
        ids -- how the actor and correlation id of the request that asked for
        this run reach the nodes that read them.
        """
        with self.leases.hold(thread.workflow_id, owner=self.owner, thread_id=thread.thread_id):
            return await self._invoke(thread, initial_state, configurable)

    async def resume(
        self,
        *,
        workflow_id: UUID | None = None,
        thread_id: str | None = None,
        resume_input: Any = None,
        configurable: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Continue a stored thread from its last checkpoint.

        Identified by `workflow_id` (what an operator knows: which business
        process is stuck) or by `thread_id` (what a caller resuming one specific
        thread of a multi-thread workflow knows). Raises
        `WorkflowLeaseUnavailable` if another worker is already on this workflow,
        `NothingToResume` if the thread has no checkpoint, and returns the stored
        values unchanged if the thread has already finished.

        `resume_input` is for the human-in-the-loop case -- the value an
        interrupted graph is waiting on. Left as `None`, the graph simply carries
        on from where it stopped. `configurable` is as for `start`.
        """
        thread = self._thread_to_resume(workflow_id=workflow_id, thread_id=thread_id)

        with self.leases.hold(thread.workflow_id, owner=self.owner, thread_id=thread.thread_id):
            state = await self.inspect(thread)
            if state.checkpoint_id is None:
                raise NothingToResume(
                    f"thread '{thread.thread_id}' of workflow {thread.workflow_id} has no "
                    f"checkpoint; it has to be started, not resumed"
                )
            if not state.next:
                logger.info(
                    "thread '%s' of workflow %s is already complete; nothing to resume",
                    thread.thread_id,
                    thread.workflow_id,
                )
                self.registry.mark_status(thread.thread_id, RUN_STATUS_COMPLETED, finished=True)
                return state.values

            logger.info(
                "resuming thread '%s' of workflow %s at step(s) %s (checkpoint %s)",
                thread.thread_id,
                thread.workflow_id,
                ", ".join(state.next),
                state.checkpoint_id,
            )
            # `resume_input` is `None` on an ordinary resume, which is what tells
            # LangGraph to continue the stored run instead of applying new input.
            return await self._invoke(thread, resume_input, configurable)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _invoke(
        self,
        thread: WorkflowThread,
        graph_input: Any,
        configurable: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke the graph on a thread, keeping the run's status honest.

        The status is the only part of a run's progress that is visible without
        deserializing a checkpoint, so it is written around every invocation:
        `running` before, and `completed` or `failed` after -- including when the
        graph raised, because a run left marked `running` by a failure is
        indistinguishable from one that is still going.
        """
        self.registry.mark_status(thread.thread_id, RUN_STATUS_RUNNING, started=True)
        try:
            final = await self.graph.ainvoke(
                graph_input, config=thread.config(**dict(configurable or {}))
            )
        except Exception:
            self.registry.mark_status(thread.thread_id, RUN_STATUS_FAILED, finished=True)
            raise

        # A graph that stopped at an interrupt (an unanswered approval, say) has
        # not finished: it is waiting. Marking it completed would hide work that
        # is still owed a human answer.
        state = await self.inspect(thread)
        if state.next:
            logger.info("thread '%s' paused at step(s) %s", thread.thread_id, ", ".join(state.next))
        else:
            self.registry.mark_status(thread.thread_id, RUN_STATUS_COMPLETED, finished=True)
        return final

    def _thread_to_resume(
        self, *, workflow_id: UUID | None, thread_id: str | None
    ) -> WorkflowThread:
        """Resolve which thread a resume request means."""
        if thread_id is not None:
            return self.registry.require(thread_id)
        if workflow_id is None:
            raise ValueError("resume requires either a workflow_id or a thread_id")

        threads = self.registry.threads_for_workflow(workflow_id)
        if not threads:
            raise NotFound(f"workflow {workflow_id} has no registered threads")
        if len(threads) > 1:
            raise AmbiguousWorkflowResume(
                f"workflow {workflow_id} has {len(threads)} threads "
                f"({', '.join(thread.thread_id for thread in threads)}); pass thread_id "
                f"to say which one to resume"
            )
        return threads[0]


__all__ = [
    "NothingToResume",
    "AmbiguousWorkflowResume",
    "DurableWorkflowRunner",
]

"""The durable, database-backed LangGraph checkpointer.

`langgraph.checkpoint.memory.InMemorySaver` is what the graphs compile with by
default, and it is a test fixture: a process restart loses every thread it
held. This module is the production implementation, backed by the `checkpoints`
and `checkpoint_writes` tables, so a run interrupted by a crash, a deploy or a
`kill -9` is still there afterwards and resumes from where it stopped.

Two things are deliberately different from a drop-in `BaseCheckpointSaver`:

**A thread must be registered before it can be checkpointed.**
`checkpoints.workflow_id` is `NOT NULL` -- a checkpoint that does not say which
business process it belongs to cannot be found by an operator resuming that
process, which is the only reason to store it durably. So the saver resolves
`thread_id` to a `workflow_runs` row through a `WorkflowThreadRegistry` and
raises `UnregisteredWorkflowThread` if there is none, instead of inventing a
workflow. `WorkflowThreadRegistry.register` is the one call that mints the
binding, and the composition root makes it before the first `ainvoke`.

**Channel values are stored inside the checkpoint blob.** LangGraph's Postgres
saver splits them into a per-channel blob table to avoid rewriting unchanged
values; this stores the serialized checkpoint whole, in the single `checkpoint`
column the Phase B schema already has. That costs a rewrite of the values on
each super-step and buys one row per checkpoint, which is the right trade at
this scale -- a job-search run is tens of super-steps over a state of a few
hundred kilobytes, not a conversation of millions of tokens.

Sessions are per call, from an injected factory, for the same reason
`SqlOperationStore` takes one: the checkpointer is called from inside a graph
run, and a session borrowed from the caller would tie every checkpoint write to
whatever transaction that caller happened to have open.
"""

import base64
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from personalos.domain.errors import NotFound, ValidationFailed
from personalos.domain.workflow import WorkflowThread
from personalos.persistence.models import (
    CheckpointModel,
    CheckpointWriteModel,
    WorkflowModel,
    WorkflowRunModel,
)

logger = logging.getLogger(__name__)

#: Statuses `WorkflowRunModel.status` may take, repeated here as the values the
#: registry sets rather than re-deriving them from the column's Enum.
RUN_STATUS_PENDING = "pending"
RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"


class UnregisteredWorkflowThread(ValidationFailed, ValueError):
    """A thread was checkpointed without first being bound to a workflow.

    Raised rather than papered over: the alternative is a checkpoint whose
    `workflow_id` was guessed, and an operator resuming that workflow would
    never find it.
    """


class WorkflowThreadRegistry:
    """Binds `thread_id`s to the workflow whose process they are part of.

    The registry owns the `workflows` / `workflow_runs` side of durable
    identity: `register` is get-or-create on both (so a restarted worker
    re-registering the same stable thread id rejoins its existing run rather
    than forking a second one), and `resolve` is the lookup the checkpointer
    and the resume path do.

    Resolutions are memoized per instance. A binding is immutable once made --
    a thread belongs to exactly one run for its whole life -- so a cache cannot
    go stale, and without one the checkpointer would re-query on every
    super-step.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        """Initialize with a callable returning a SQLAlchemy `Session`."""
        self.session_factory = session_factory
        self._cache: dict[str, WorkflowThread] = {}

    def register(
        self,
        *,
        thread_id: str,
        workflow_name: str,
        workflow_id: UUID | None = None,
        user_id: UUID | None = None,
        actor_id: str = "system",
        correlation_id: UUID | None = None,
    ) -> WorkflowThread:
        """Bind `thread_id` to a workflow run, creating either if needed.

        Idempotent in the way a restart needs it to be: called again with the
        same `thread_id` it returns the existing binding untouched, so the
        second process to run a stable thread id continues the first one's run
        instead of starting a fresh one whose checkpoints nothing links.

        `workflow_id` is explicit when the caller is resuming a known process
        and omitted when starting a new one, in which case the workflow is
        looked up by `workflow_name` and created on first use.
        """
        session = self.session_factory()
        try:
            existing = (
                session.query(WorkflowRunModel)
                .filter(WorkflowRunModel.thread_id == thread_id)
                .first()
            )
            if existing is not None:
                thread = self._thread_for(existing, workflow_name)
                self._cache[thread_id] = thread
                return thread

            workflow = self._workflow(session, workflow_id=workflow_id, workflow_name=workflow_name)
            run = WorkflowRunModel(
                workflow_id=workflow.id,
                user_id=user_id,
                thread_id=thread_id,
                actor_id=actor_id,
                correlation_id=correlation_id,
                status=RUN_STATUS_PENDING,
            )
            session.add(run)
            try:
                session.commit()
            except IntegrityError:
                # Lost the insert race on `workflow_runs.thread_id`: another
                # worker registered the same stable thread first, and its row is
                # the binding both of them must use.
                session.rollback()
                winner = (
                    session.query(WorkflowRunModel)
                    .filter(WorkflowRunModel.thread_id == thread_id)
                    .first()
                )
                if winner is None:  # pragma: no cover - a unique violation implies a row
                    raise
                thread = self._thread_for(winner, workflow_name)
                self._cache[thread_id] = thread
                return thread

            thread = self._thread_for(run, workflow.name)
            self._cache[thread_id] = thread
            return thread
        finally:
            session.close()

    def resolve(self, thread_id: str) -> WorkflowThread | None:
        """Return the binding for a thread, or `None` if it was never registered."""
        cached = self._cache.get(thread_id)
        if cached is not None:
            return cached

        session = self.session_factory()
        try:
            run = (
                session.query(WorkflowRunModel)
                .filter(WorkflowRunModel.thread_id == thread_id)
                .first()
            )
            if run is None:
                return None
            workflow = (
                session.query(WorkflowModel).filter(WorkflowModel.id == run.workflow_id).first()
            )
            thread = self._thread_for(run, workflow.name if workflow else None)
            self._cache[thread_id] = thread
            return thread
        finally:
            session.close()

    def require(self, thread_id: str) -> WorkflowThread:
        """Resolve a thread, raising `UnregisteredWorkflowThread` if unbound."""
        thread = self.resolve(thread_id)
        if thread is None:
            raise UnregisteredWorkflowThread(
                f"thread '{thread_id}' is not bound to a workflow run; call "
                f"WorkflowThreadRegistry.register before running or resuming it"
            )
        return thread

    def threads_for_workflow(self, workflow_id: UUID) -> list[WorkflowThread]:
        """Every thread belonging to a workflow, oldest first.

        How a resume that was given only a `workflow_id` finds something to
        resume: an operator knows which business process is stuck, not which
        LangGraph thread it stopped on.
        """
        session = self.session_factory()
        try:
            runs = (
                session.query(WorkflowRunModel)
                .filter(WorkflowRunModel.workflow_id == workflow_id)
                # thread_id breaks ties: two runs registered in the same
                # millisecond would otherwise come back in an arbitrary order,
                # and "the workflow's threads" should not depend on the query
                # plan.
                .order_by(WorkflowRunModel.created_at.asc(), WorkflowRunModel.thread_id.asc())
                .all()
            )
            workflow = session.query(WorkflowModel).filter(WorkflowModel.id == workflow_id).first()
            name = workflow.name if workflow else None
            threads = [self._thread_for(run, name) for run in runs]
            for thread in threads:
                self._cache[thread.thread_id] = thread
            return threads
        finally:
            session.close()

    def mark_status(
        self,
        thread_id: str,
        status: str,
        *,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        """Record a run's lifecycle status, so a stuck run is visible in SQL.

        The checkpoint says where a run stopped; this says whether anyone
        believes it is still going. An operator looking for work to resume
        wants the second question answered without deserializing state.
        """
        session = self.session_factory()
        try:
            run = (
                session.query(WorkflowRunModel)
                .filter(WorkflowRunModel.thread_id == thread_id)
                .first()
            )
            if run is None:
                raise UnregisteredWorkflowThread(
                    f"cannot set status of unregistered thread '{thread_id}'"
                )
            now = datetime.utcnow()
            run.status = status
            run.updated_at = now
            if started and run.started_at is None:
                run.started_at = now
            if finished:
                run.completed_at = now
            session.commit()
        finally:
            session.close()

    @staticmethod
    def _workflow(
        session: Session, *, workflow_id: UUID | None, workflow_name: str
    ) -> WorkflowModel:
        """Get-or-create the workflow definition a new run hangs off."""
        if workflow_id is not None:
            workflow = session.query(WorkflowModel).filter(WorkflowModel.id == workflow_id).first()
            if workflow is None:
                raise NotFound(f"workflow {workflow_id} does not exist")
            return workflow

        workflow = session.query(WorkflowModel).filter(WorkflowModel.name == workflow_name).first()
        if workflow is not None:
            return workflow
        workflow = WorkflowModel(name=workflow_name)
        session.add(workflow)
        session.commit()
        return workflow

    @staticmethod
    def _thread_for(run: WorkflowRunModel, workflow_name: str | None) -> WorkflowThread:
        return WorkflowThread(
            workflow_id=run.workflow_id,
            thread_id=run.thread_id,
            workflow_run_id=run.id,
            workflow_name=workflow_name,
        )


def _encode(serde: Any, value: Any) -> dict[str, Any]:
    """Serialize a value into the JSON envelope a `JSON` column can hold.

    LangGraph's serializer returns `(type_tag, bytes)`; base64 is what carries
    the bytes half through a JSON column unchanged on both Postgres and SQLite.
    """
    type_, blob = serde.dumps_typed(value)
    return {"type": type_, "data": base64.b64encode(blob).decode("ascii")}


def _decode(serde: Any, envelope: dict[str, Any]) -> Any:
    """Rebuild a value from the envelope `_encode` wrote."""
    return serde.loads_typed((envelope["type"], base64.b64decode(envelope["data"])))


def _jsonable_metadata(metadata: CheckpointMetadata) -> dict[str, Any]:
    """Coerce checkpoint metadata to plain, filterable JSON.

    Metadata is stored for inspection and for `list(filter=...)`, so it is kept
    queryable rather than opaque; anything JSON cannot represent degrades to its
    string form (`default=str`) instead of failing the checkpoint write. That is
    a real trade -- a `UUID` in metadata comes back as a string -- and it is the
    same one LangGraph's own Postgres saver makes for its `jsonb` metadata
    column. Checkpoint *state* is never treated this way; see `_encode`.
    """
    return json.loads(json.dumps(dict(metadata), default=str))


class SqlAlchemyCheckpointSaver(BaseCheckpointSaver[str]):
    """A `BaseCheckpointSaver` over the `checkpoints` / `checkpoint_writes` tables.

    Drop-in for `InMemorySaver` in everything except the two properties that
    matter: state outlives the process, and a thread has to be registered
    (`WorkflowThreadRegistry.register`) before it may be written.

    The async methods run their synchronous bodies inline rather than handing
    them to a worker thread, which is a deliberate and slightly surprising
    choice. LangGraph submits each checkpoint write as a background task chained
    onto the previous one; an `await asyncio.to_thread(...)` inside that task
    yields before the row is written, so the chain falls behind the graph it is
    supposed to be recording, and a process killed mid-run loses every write
    still queued behind the hop. Measured on the Job Search pipeline, that lag
    was most of the run. Writing inline costs a few milliseconds of event-loop
    time per super-step and buys a checkpoint that is on disk before the next
    step starts, which is the entire purpose of this class.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        registry: WorkflowThreadRegistry | None = None,
        *,
        serde: Any = None,
    ):
        """Initialize with a session factory and the registry that resolves threads."""
        super().__init__(serde=serde)
        self.session_factory = session_factory
        self.registry = registry or WorkflowThreadRegistry(session_factory)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Load one checkpoint: the one named by the config, else the latest."""
        configurable = config.get("configurable") or {}
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        session = self.session_factory()
        try:
            row = self._row(session, thread_id, checkpoint_ns, checkpoint_id)
            if row is None:
                return None
            return self._tuple(session, row)
        finally:
            session.close()

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002 - the base class names it `filter`
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints newest first, narrowed by thread, namespace and metadata.

        Materialized before yielding so the session is closed by the time the
        caller starts consuming: a generator that held a session open across
        the consumer's own work would keep a connection checked out for as long
        as the caller took to iterate.
        """
        configurable = (config or {}).get("configurable") or {}
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns")
        checkpoint_id = get_checkpoint_id(config) if config else None
        before_id = get_checkpoint_id(before) if before else None

        session = self.session_factory()
        try:
            query = session.query(CheckpointModel)
            if thread_id is not None:
                query = query.filter(CheckpointModel.thread_id == thread_id)
            if checkpoint_ns is not None:
                query = query.filter(CheckpointModel.checkpoint_ns == checkpoint_ns)
            if checkpoint_id is not None:
                query = query.filter(CheckpointModel.checkpoint_id == checkpoint_id)
            if before_id is not None:
                query = query.filter(CheckpointModel.checkpoint_id < before_id)

            results: list[CheckpointTuple] = []
            for row in query.order_by(CheckpointModel.checkpoint_id.desc()).all():
                metadata = row.checkpoint_metadata or {}
                if filter and not all(metadata.get(key) == value for key, value in filter.items()):
                    continue
                results.append(self._tuple(session, row))
                if limit is not None and len(results) >= limit:
                    break
            return iter(results)
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist one checkpoint, and return the config that names it.

        Committed before returning, which is the property the whole design
        rests on: LangGraph writes the checkpoint for a super-step before
        running that step's tasks, so by the time a node performs a side effect
        the state that led to it is already durable, and a worker killed inside
        that node resumes at the node rather than at the start of the run.
        """
        configurable = config.get("configurable") or {}
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        parent_checkpoint_id = configurable.get("checkpoint_id")
        thread = self.registry.require(thread_id)

        payload = _encode(self.serde, checkpoint)
        stored_metadata = _jsonable_metadata(get_checkpoint_metadata(config, metadata))

        session = self.session_factory()
        try:
            row = self._row(session, thread_id, checkpoint_ns, checkpoint["id"])
            if row is None:
                row = CheckpointModel(
                    workflow_id=thread.workflow_id,
                    workflow_run_id=thread.workflow_run_id,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint["id"],
                    parent_checkpoint_id=parent_checkpoint_id,
                    checkpoint=payload,
                    checkpoint_metadata=stored_metadata,
                )
                session.add(row)
                try:
                    session.commit()
                except IntegrityError:
                    # A concurrent writer inserted the same checkpoint id; fall
                    # through to the update path rather than losing this write.
                    session.rollback()
                    row = self._row(session, thread_id, checkpoint_ns, checkpoint["id"])
                    if row is None:  # pragma: no cover - a unique violation implies a row
                        raise
                    self._overwrite(session, row, payload, stored_metadata, parent_checkpoint_id)
            else:
                # Re-put of an existing id: a forked or manually updated state.
                self._overwrite(session, row, payload, stored_metadata, parent_checkpoint_id)
        finally:
            session.close()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Record what one task wrote against a checkpoint.

        Two kinds of write, distinguished by `idx`, following
        `langgraph.checkpoint.base.WRITES_IDX_MAP`: a reserved channel
        (`__error__`, `__interrupt__`, ...) has a fixed negative index and is
        *overwritten* on re-write, because the newest error or interrupt is the
        true one; an ordinary channel write is positional and is *kept*, because
        a replayed task re-emitting the same write must not duplicate it.
        """
        configurable = config.get("configurable") or {}
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable["checkpoint_id"]

        session = self.session_factory()
        try:
            for position, (channel, value) in enumerate(writes):
                idx = WRITES_IDX_MAP.get(channel, position)
                existing = (
                    session.query(CheckpointWriteModel)
                    .filter(
                        CheckpointWriteModel.thread_id == thread_id,
                        CheckpointWriteModel.checkpoint_ns == checkpoint_ns,
                        CheckpointWriteModel.checkpoint_id == checkpoint_id,
                        CheckpointWriteModel.task_id == task_id,
                        CheckpointWriteModel.idx == idx,
                    )
                    .first()
                )
                if existing is not None:
                    if idx >= 0:
                        continue
                    existing.channel = channel
                    existing.value = _encode(self.serde, value)
                    existing.task_path = task_path
                    continue

                session.add(
                    CheckpointWriteModel(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        task_id=task_id,
                        idx=idx,
                        channel=channel,
                        value=_encode(self.serde, value),
                        task_path=task_path,
                    )
                )
            try:
                session.commit()
            except IntegrityError:
                # Another attempt at the same task wrote these rows first. Its
                # writes are equivalent to ours by construction (same task, same
                # indices), so losing this race is not a failure.
                session.rollback()
                logger.debug(
                    "checkpoint writes for task %s on %s already recorded", task_id, checkpoint_id
                )
        finally:
            session.close()

    def delete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint and write belonging to a thread."""
        session = self.session_factory()
        try:
            session.query(CheckpointWriteModel).filter(
                CheckpointWriteModel.thread_id == thread_id
            ).delete()
            session.query(CheckpointModel).filter(CheckpointModel.thread_id == thread_id).delete()
            session.commit()
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Async surface
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async `get_tuple`; see the class docstring on why it runs inline."""
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002 - the base class names it `filter`
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async `list`; see the class docstring on why it runs inline."""
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async `put`; see the class docstring on why it runs inline."""
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async `put_writes`; see the class docstring on why it runs inline."""
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        """Async `delete_thread`; see the class docstring on why it runs inline."""
        self.delete_thread(thread_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _row(
        session: Session, thread_id: str, checkpoint_ns: str, checkpoint_id: str | None
    ) -> CheckpointModel | None:
        """The named checkpoint row, or the newest one for the thread/namespace.

        "Newest" is `max(checkpoint_id)`: LangGraph checkpoint ids are uuid6,
        which sort lexicographically in creation order, so the ordering the
        column gives for free is the right one.
        """
        query = session.query(CheckpointModel).filter(
            CheckpointModel.thread_id == thread_id,
            CheckpointModel.checkpoint_ns == checkpoint_ns,
        )
        if checkpoint_id is not None:
            return query.filter(CheckpointModel.checkpoint_id == checkpoint_id).first()
        return query.order_by(CheckpointModel.checkpoint_id.desc()).first()

    @staticmethod
    def _overwrite(
        session: Session,
        row: CheckpointModel,
        payload: dict[str, Any],
        metadata: dict[str, Any],
        parent_checkpoint_id: str | None,
    ) -> None:
        row.checkpoint = payload
        row.checkpoint_metadata = metadata
        row.parent_checkpoint_id = parent_checkpoint_id
        row.updated_at = datetime.utcnow()
        session.commit()

    def _tuple(self, session: Session, row: CheckpointModel) -> CheckpointTuple:
        """Rebuild the `CheckpointTuple` LangGraph resumes from."""
        pending_writes = [
            (write.task_id, write.channel, _decode(self.serde, write.value))
            for write in session.query(CheckpointWriteModel)
            .filter(
                CheckpointWriteModel.thread_id == row.thread_id,
                CheckpointWriteModel.checkpoint_ns == row.checkpoint_ns,
                CheckpointWriteModel.checkpoint_id == row.checkpoint_id,
            )
            .order_by(CheckpointWriteModel.task_id.asc(), CheckpointWriteModel.idx.asc())
            .all()
        ]
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": row.thread_id,
                    "checkpoint_ns": row.checkpoint_ns,
                    "checkpoint_id": row.checkpoint_id,
                }
            },
            checkpoint=_decode(self.serde, row.checkpoint),
            metadata=row.checkpoint_metadata or {},
            parent_config=(
                {
                    "configurable": {
                        "thread_id": row.thread_id,
                        "checkpoint_ns": row.checkpoint_ns,
                        "checkpoint_id": row.parent_checkpoint_id,
                    }
                }
                if row.parent_checkpoint_id
                else None
            ),
            pending_writes=pending_writes,
        )


__all__ = [
    "RUN_STATUS_PENDING",
    "RUN_STATUS_RUNNING",
    "RUN_STATUS_COMPLETED",
    "RUN_STATUS_FAILED",
    "UnregisteredWorkflowThread",
    "WorkflowThreadRegistry",
    "SqlAlchemyCheckpointSaver",
]

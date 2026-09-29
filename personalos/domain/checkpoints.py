"""Pending checkpoints: conditional waits that outlive the process that made them.

A job search is full of sentences of the form *"seven days after applying, if no
recruiter response exists, draft a follow-up"*. Two things in that sentence are
easy to implement wrongly and expensive to get wrong:

**"seven days after"** is longer than any process, connection or graph run.
Something has to survive a deploy, a crash and a weekend, and the only thing in
this system that does is a row. So a pending checkpoint carries everything
needed to act on it later -- its condition, the moment it becomes actionable
(`trigger_at`), and the moment after which it must never act (`expires_at`) --
and nothing that presumes anybody is still holding it. Nothing here refers to a
thread that is running, a task that is sleeping, or a future that is awaited.
The `thread_id` it names is what a resume *looks up*, not something it holds
open.

**"if no recruiter response exists"** is a statement about the world at the
moment the follow-up would be sent, not at the moment the wait was set up. The
whole point of waiting is that the world may change while you wait, so the
condition is stored *declaratively* -- as a `CheckpointCondition` value, not a
closure, not a captured boolean -- and evaluated by whatever picks the
checkpoint up at trigger time. A condition evaluated at creation time and stored
as a flag would make the wait pointless: it would fire exactly when the
recruiter had already replied.

`decide_checkpoint` is the whole policy, as one pure function of a checkpoint,
a freshly evaluated condition and a `now`. Pure and explicitly-timed so every
branch is testable without sleeping and so the caller records the same answer it
acted on. Its three interesting rules, in the order they apply:

1. **A met condition closes the checkpoint, silently.** The reason for the wait
   went away, so there is nothing to do and nothing to tell anyone about; a
   "your follow-up was cancelled" notification for a follow-up nobody ever saw
   is noise.
2. **Expiry beats firing.** A checkpoint past `expires_at` is marked expired
   even when `trigger_at` also passed -- which is exactly the case where a
   monitor was down for a week. Sending a week-late "just checking in" is worse
   than sending nothing, and the alternative to expiring it is leaving it
   pending forever.
3. **Only then does it fire.**

This module is `domain`: it owns the shape of a durable wait and the rules for
resolving one, and knows nothing about the table it is stored in
(`personalos.persistence.pending_checkpoints`) or the process that sweeps it
(`apps.worker.checkpoint_monitor`).
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from personalos.domain.errors import ValidationFailed
from personalos.domain.job_search import FollowUpKind

#: How long after `trigger_at` a checkpoint stays actionable before it is
#: written off. Bounded, and bounded *explicitly* rather than as "trigger_at
#: plus whatever the sweeper's backlog is": the gap between the two is the only
#: window in which a late follow-up is still worth sending, and how wide it is
#: is a product decision, not a scheduling accident. Three days is roughly "a
#: long weekend of outage still gets the follow-up out".
DEFAULT_CHECKPOINT_GRACE = timedelta(days=3)

#: Upper bound on `thread_id`, matching `personalos.domain.workflow` and the
#: `String(255)` columns both ids live in.
MAX_THREAD_ID_LENGTH = 255


class CheckpointContractError(ValidationFailed, ValueError):
    """A pending checkpoint violated its contract.

    Subclasses both `ValidationFailed` (so it reports through the shared error
    taxonomy) and `ValueError`, matching `JobSearchContractError` and
    `InvalidWorkflowIdentity` elsewhere in `personalos.domain`.
    """


class ConditionKind(str, Enum):
    """What would make a pending checkpoint unnecessary.

    Deliberately phrased as the *resolution* -- the thing whose existence means
    "never mind" -- rather than as the reason the checkpoint was created. That
    is the direction the re-evaluation asks in: at trigger time the question is
    "has the thing we were waiting for happened yet?", and a `True` answer
    closes the checkpoint rather than firing it.

    A closed set, for the same reason `ActionKind` is closed: an evaluator has
    to know how to answer every kind, and a condition nothing knows how to
    evaluate is a checkpoint that either never fires or always does.
    """

    #: Any inbound recruiter message landed on this application after `since`.
    RECRUITER_RESPONSE_RECEIVED = "recruiter_response_received"
    #: The candidate's owed reply actually went out.
    CANDIDATE_REPLY_SENT = "candidate_reply_sent"
    #: The application reached a terminal state (rejected, withdrawn, skipped),
    #: so every pending nudge attached to it is moot.
    APPLICATION_CLOSED = "application_closed"


class CheckpointCondition(BaseModel):
    """The condition a pending checkpoint is re-evaluated against, stored as data.

    A value rather than a callable, and that is the load-bearing choice. A
    closure cannot be written to a row, cannot survive the process that built
    it, and -- worse -- would capture the world as it looked when the wait
    started, which is precisely the world the wait exists to let change. Storing
    the *question* instead means whoever picks the checkpoint up days later asks
    it fresh.

    `subject_id` is what the condition is about (an application, here);
    `since` bounds it in time, so "a recruiter response exists" means one that
    arrived after the checkpoint was created rather than the reply from six
    weeks ago that prompted it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ConditionKind
    subject_id: UUID
    since: datetime | None = None
    #: Anything else an evaluator needs, kept JSON-shaped so the condition
    #: round-trips through a `JSON` column and a checkpointed graph state
    #: without a custom serializer.
    params: dict[str, Any] = Field(default_factory=dict)

    def describe(self) -> str:
        """A one-line, human-readable form, for logs and closure reasons."""
        window = f" since {self.since.isoformat()}" if self.since else ""
        return f"{self.kind.value} for {self.subject_id}{window}"


class PendingCheckpointStatus(str, Enum):
    """Where a pending checkpoint is in its life.

    Four terminal states rather than one, because "why did this never fire?" is
    the question an operator actually asks, and collapsing them into `closed`
    would throw away the answer.
    """

    #: Waiting. The only state from which anything may happen.
    PENDING = "pending"
    #: The condition came true, so the checkpoint was closed with no follow-up.
    RESOLVED = "resolved"
    #: The trigger fired and the graph path was started.
    FIRED = "fired"
    #: The expiry passed before the checkpoint fired; deliberately not sent.
    EXPIRED = "expired"
    #: Closed by something other than its own rules -- the thread it names is
    #: gone, or an operator called it off.
    CANCELLED = "cancelled"


class CheckpointOutcome(str, Enum):
    """What `decide_checkpoint` says to do with a checkpoint right now."""

    #: Not yet actionable; leave it exactly as it is.
    WAIT = "wait"
    #: Condition met: close it silently, emit nothing, start nothing.
    RESOLVE = "resolve"
    #: Due and still unresolved: start the graph path it names.
    FIRE = "fire"
    #: Past its expiry: mark it so, and never act on it.
    EXPIRE = "expire"


class PendingCheckpoint(BaseModel):
    """One durable, conditional wait, independent of any open process.

    Immutable, like every other value crossing a layer here: a sweeper that
    could mutate the checkpoint it is deciding about could decide against a
    checkpoint different from the one it then records. Closing one produces a
    new value (`closed_as`), and the store's guarded `UPDATE` is what makes that
    close exclusive between two sweepers.

    The three fields that make this survivable are `condition`, `trigger_at`
    and `expires_at`, and they are independent on purpose -- see the module
    docstring. `thread_id` (with `workflow_id`) is how the checkpoint names the
    graph path to start when it fires: a *lookup key*, resolved by
    `personalos.persistence.checkpointer.WorkflowThreadRegistry` at trigger
    time, never a live handle.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_id: UUID = Field(default_factory=uuid4)
    #: The aggregate this wait is about -- an application, for every kind this
    #: build creates. Also `condition.subject_id`, which is what the evaluator
    #: reads; kept here too so a checkpoint is findable by application without
    #: unpacking its condition in SQL.
    application_id: UUID
    kind: FollowUpKind
    condition: CheckpointCondition
    #: Why this wait exists, in the words an operator would want to read.
    reason: str
    #: The thread whose graph path this checkpoint starts when it fires.
    thread_id: str
    workflow_id: UUID | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    trigger_at: datetime
    expires_at: datetime
    status: PendingCheckpointStatus = PendingCheckpointStatus.PENDING
    #: Stable identity for "this wait, on this application, of this kind", so
    #: re-running the branch that creates it does not pile up duplicate
    #: reminders. Mirrors `outbox_events.dedupe_key`.
    dedupe_key: str
    closed_at: datetime | None = None
    closed_reason: str | None = None

    @field_validator("reason", "dedupe_key")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise CheckpointContractError(
                "a pending checkpoint must carry a reason and a dedupe key; the first "
                "is what an operator reads, the second is what stops duplicates"
            )
        return value

    @field_validator("thread_id")
    @classmethod
    def _check_thread_id(cls, value: str) -> str:
        if not value.strip():
            raise CheckpointContractError(
                "a pending checkpoint must name the thread it will start; a checkpoint "
                "with nowhere to resume can only ever expire"
            )
        if len(value) > MAX_THREAD_ID_LENGTH:
            raise CheckpointContractError(
                f"thread_id is {len(value)} characters; the maximum is {MAX_THREAD_ID_LENGTH}"
            )
        return value

    @field_validator("trigger_at")
    @classmethod
    def _trigger_after_creation(cls, value: datetime, info) -> datetime:
        created_at = info.data.get("created_at")
        if created_at is not None and value < created_at:
            raise CheckpointContractError(
                f"a checkpoint cannot trigger before it was created "
                f"(trigger_at={value}, created_at={created_at})"
            )
        return value

    @field_validator("expires_at")
    @classmethod
    def _expiry_after_trigger(cls, value: datetime, info) -> datetime:
        trigger_at = info.data.get("trigger_at")
        if trigger_at is not None and value <= trigger_at:
            raise CheckpointContractError(
                f"a checkpoint must expire after it triggers (expires_at={value}, "
                f"trigger_at={trigger_at}); an expiry at or before the trigger is a "
                f"wait that can only ever be written off"
            )
        return value

    @classmethod
    def for_follow_up(
        cls,
        *,
        application_id: UUID,
        kind: FollowUpKind,
        due_at: datetime,
        reason: str,
        thread_id: str,
        workflow_id: UUID | None = None,
        created_at: datetime | None = None,
        grace: timedelta | None = None,
        condition: CheckpointCondition | None = None,
    ) -> "PendingCheckpoint":
        """Build the durable wait behind one `FollowUpCheckpoint`.

        The two are deliberately not the same object.
        `personalos.domain.job_search.FollowUpCheckpoint` is what the graph
        computed -- "this application wants a nudge on the 7th" -- and is part
        of the run's state. This is the row that makes that survive the run, and
        it carries the two things the state form has no business knowing: the
        condition that might make it unnecessary, and where to resume.

        `condition` defaults to the resolution that matches the kind, so the
        common case cannot get the two out of step; a caller with a narrower
        question passes its own.
        """
        created_at = created_at or datetime.utcnow()
        return cls(
            application_id=application_id,
            kind=kind,
            condition=condition
            or condition_for_kind(kind, application_id, since=created_at),
            reason=reason,
            thread_id=thread_id,
            workflow_id=workflow_id,
            created_at=created_at,
            trigger_at=due_at,
            expires_at=due_at + (grace or DEFAULT_CHECKPOINT_GRACE),
            dedupe_key=follow_up_dedupe_key(application_id, kind),
        )

    @property
    def is_open(self) -> bool:
        """True while this checkpoint may still do something."""
        return self.status == PendingCheckpointStatus.PENDING

    def is_due(self, now: datetime) -> bool:
        """True once the trigger moment has arrived."""
        return now >= self.trigger_at

    def has_expired(self, now: datetime) -> bool:
        """True once this checkpoint may no longer act, fired or not."""
        return now >= self.expires_at

    def closed_as(
        self, status: PendingCheckpointStatus, *, at: datetime, reason: str
    ) -> "PendingCheckpoint":
        """This checkpoint, closed in `status`, with the reason recorded.

        Refuses to close it as `PENDING`: "closed, still waiting" is not a
        state, and allowing it would let a caller silently reopen a decided
        checkpoint.
        """
        if status == PendingCheckpointStatus.PENDING:
            raise CheckpointContractError(
                "closing a checkpoint requires a terminal status; PENDING is where it "
                "started"
            )
        return self.model_copy(
            update={"status": status, "closed_at": at, "closed_reason": reason}
        )


#: Which resolution makes each kind of follow-up unnecessary. A table rather
#: than a judgement made at each creation site, for the same reason
#: `ACTION_RISK_PROFILES` is one: two places deciding "the obvious condition"
#: for the same kind is two checkpoints that behave differently for no reason
#: anyone can see in the code.
_CONDITION_FOR_KIND: dict[FollowUpKind, ConditionKind] = {
    FollowUpKind.NO_RESPONSE: ConditionKind.RECRUITER_RESPONSE_RECEIVED,
    FollowUpKind.AWAITING_CANDIDATE_REPLY: ConditionKind.CANDIDATE_REPLY_SENT,
    # Nothing the candidate does makes interview prep unnecessary -- only the
    # application ending does.
    FollowUpKind.INTERVIEW_PREP: ConditionKind.APPLICATION_CLOSED,
}


def condition_for_kind(
    kind: FollowUpKind, application_id: UUID, *, since: datetime | None = None
) -> CheckpointCondition:
    """The resolution condition for a kind of follow-up.

    Raises rather than defaulting, matching
    `personalos.domain.job_search.risk_profile_for`: a new `FollowUpKind` with
    no entry here is a wait nobody has decided how to cancel, and defaulting it
    to something permissive would silently suppress every follow-up of that kind
    (or send every one of them).
    """
    try:
        condition_kind = _CONDITION_FOR_KIND[kind]
    except KeyError as exc:
        raise CheckpointContractError(
            f"no resolution condition registered for follow-up kind "
            f"'{getattr(kind, 'value', kind)}'; every durable wait must say what "
            f"would make it unnecessary"
        ) from exc
    return CheckpointCondition(
        kind=condition_kind, subject_id=application_id, since=since
    )


def follow_up_dedupe_key(application_id: UUID, kind: FollowUpKind) -> str:
    """Stable identity for one application's wait of one kind.

    One definition, because the graph writes it and the store deduplicates on
    it: two spellings would let the same reminder be scheduled twice.
    """
    return f"follow_up:{application_id}:{kind.value}"


def decide_checkpoint(
    *,
    checkpoint: PendingCheckpoint,
    condition_met: bool,
    now: datetime,
) -> CheckpointOutcome:
    """Decide what to do with one pending checkpoint, given a freshly asked condition.

    Pure, and takes both `condition_met` and `now` rather than evaluating or
    reading either: the caller has to record the same decision it acted on, and
    a function that consulted a clock could be asked twice and answer
    differently. It also keeps the interesting cases -- expired-and-due,
    met-and-expired -- testable without waiting for any of them.

    The ordering is the policy:

    1. **Met beats everything.** The reason for the wait is gone, so the
       checkpoint closes silently whether or not it was due and whether or not
       it has expired. Recording it as `EXPIRED` when the recruiter had in fact
       replied would misreport a success as a miss.
    2. **Expired beats due.** A checkpoint that is past `expires_at` is written
       off even though `trigger_at` also passed -- that is the monitor-was-down
       case, and a week-late nudge is worse than none. This is the rule that
       makes `expires_at` worth storing separately from `trigger_at`.
    3. **Due fires.**
    4. Otherwise it waits.

    Refuses a checkpoint that is already closed: deciding about one again is a
    caller bug (a sweep that re-selected a fired row), and answering it would
    fire a follow-up twice.
    """
    if not checkpoint.is_open:
        raise CheckpointContractError(
            f"checkpoint {checkpoint.checkpoint_id} is already {checkpoint.status.value}; "
            f"only a pending checkpoint has an outcome to decide"
        )
    if condition_met:
        return CheckpointOutcome.RESOLVE
    if checkpoint.has_expired(now):
        return CheckpointOutcome.EXPIRE
    if checkpoint.is_due(now):
        return CheckpointOutcome.FIRE
    return CheckpointOutcome.WAIT


__all__ = [
    "DEFAULT_CHECKPOINT_GRACE",
    "MAX_THREAD_ID_LENGTH",
    "CheckpointContractError",
    "ConditionKind",
    "CheckpointCondition",
    "PendingCheckpointStatus",
    "CheckpointOutcome",
    "PendingCheckpoint",
    "condition_for_kind",
    "follow_up_dedupe_key",
    "decide_checkpoint",
]

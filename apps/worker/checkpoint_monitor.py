"""Sweeps pending checkpoints: the process that makes a week-long wait happen.

A `pending_checkpoints` row is a wait nobody is holding. That is what makes it
survive a deploy, and it is also why something has to come and look: a row does
not wake up. This module is that something. It runs on a schedule, asks the
store what is actionable, re-evaluates each condition *now*, and does one of
four things per checkpoint -- nothing, close it silently, start the graph path
it names, or write it off as expired.

The decision itself is not here. It is
`personalos.domain.checkpoints.decide_checkpoint`, a pure function of the
checkpoint, the freshly-asked condition and the clock, so the policy ("a met
condition closes silently; expiry beats firing; only then does it fire") is
testable without a database, a graph or a sweep. What is here is everything the
policy cannot be: the I/O, the ordering, and the exclusion between two monitors
that both noticed the same due checkpoint.

Three orderings in `sweep` are load-bearing:

**The condition is asked at sweep time, never at schedule time.** The whole
value of "wait seven days, then check" is that the world may change during the
seven days. `CheckpointCondition` is stored as a question precisely so this
process can ask it fresh, and a monitor that trusted a flag written a week ago
would send exactly the follow-ups it should not.

**The checkpoint is claimed before the graph is started.** `store.close(...,
FIRED)` is a guarded `UPDATE` that exactly one of two racing monitors wins, and
only the winner invokes. Claim-then-act is the same trade
`personalos.persistence.action_journal` makes and is made here for a weaker
reason -- the run that follows parks at an approval interrupt before anything
leaves the system, so a duplicate is two drafts rather than two messages -- but
a claim taken afterwards would not exclude anybody at all. A crash in the gap
loses that one follow-up, which is why the failure is logged at `error` with
the checkpoint id: reopening it is a deliberate act, not a silent retry.

**A thread already parked is left alone.** A checkpoint whose thread is
currently waiting on an unanswered approval is deferred, not fired: starting a
run on a thread that is mid-interrupt would stack a second request on top of
one a human has not answered. It stays pending, is retried next sweep, and if
nobody ever answers, its own expiry ends it -- which is the behaviour that
expiry exists for.

This module is composition, not a layer: it holds a store, a registry, a runner
and an evaluator and does nothing but join them, which is why it may import
from all of them.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.domain.checkpoints import (
    CheckpointCondition,
    CheckpointOutcome,
    PendingCheckpoint,
    PendingCheckpointStatus,
    decide_checkpoint,
)
from personalos.persistence.checkpointer import WorkflowThreadRegistry
from personalos.persistence.leases import WorkflowLeaseUnavailable
from personalos.persistence.pending_checkpoints import (
    DEFAULT_SWEEP_LIMIT,
    PendingCheckpointStore,
)

logger = logging.getLogger(__name__)


class CheckpointConditionEvaluator(Protocol):
    """Answers, right now, whether a stored condition has come true.

    The port that keeps `decide_checkpoint` pure. A condition is stored as data
    (`personalos.domain.checkpoints.CheckpointCondition`) precisely so that
    answering it is somebody else's job, done at trigger time against whatever
    the world looks like then -- a communications store, an inbox, the
    application's current status.

    Implementations must answer honestly about *not knowing*. Returning `True`
    when the answer is unavailable silently cancels a follow-up nobody decided
    to cancel; raising is the correct response to "I cannot tell", and the
    sweep leaves that checkpoint pending for the next pass.
    """

    async def is_met(self, condition: CheckpointCondition) -> bool:
        """Return whether this condition holds as of now."""
        ...


@dataclass(frozen=True)
class SweepReport:
    """What one sweep did, by outcome.

    Every checkpoint the sweep looked at lands in exactly one bucket, which is
    what makes "the follow-up was not sent" answerable: `resolved` and
    `expired` are both silences, and they mean opposite things.
    """

    #: Condition came true; closed with no follow-up and no event.
    resolved: tuple[UUID, ...] = ()
    #: Trigger reached with the condition still unmet; the graph path was started.
    fired: tuple[UUID, ...] = ()
    #: Expiry passed without firing; deliberately written off.
    expired: tuple[UUID, ...] = ()
    #: This sweep did not decide it -- the thread was parked, or another
    #: monitor won the guarded close first. Not an error, and not a silence
    #: anyone has to explain: either it is still pending and the next sweep
    #: picks it up, or the monitor that beat us to it has already acted.
    deferred: tuple[UUID, ...] = ()
    #: The thread this checkpoint names no longer has anything to resume.
    cancelled: tuple[UUID, ...] = ()
    #: Claimed, then the invocation raised. The follow-up did not happen and
    #: will not be retried without an operator.
    failed: tuple[UUID, ...] = ()
    #: Checkpoints re-read and found not actionable after all.
    waiting: tuple[UUID, ...] = ()

    @property
    def considered(self) -> int:
        """How many checkpoints this sweep decided about."""
        return sum(
            len(bucket)
            for bucket in (
                self.resolved,
                self.fired,
                self.expired,
                self.deferred,
                self.cancelled,
                self.failed,
                self.waiting,
            )
        )


@dataclass
class _Buckets:
    """Mutable accumulator behind the frozen `SweepReport`."""

    resolved: list[UUID] = field(default_factory=list)
    fired: list[UUID] = field(default_factory=list)
    expired: list[UUID] = field(default_factory=list)
    deferred: list[UUID] = field(default_factory=list)
    cancelled: list[UUID] = field(default_factory=list)
    failed: list[UUID] = field(default_factory=list)
    waiting: list[UUID] = field(default_factory=list)

    def report(self) -> SweepReport:
        return SweepReport(
            resolved=tuple(self.resolved),
            fired=tuple(self.fired),
            expired=tuple(self.expired),
            deferred=tuple(self.deferred),
            cancelled=tuple(self.cancelled),
            failed=tuple(self.failed),
            waiting=tuple(self.waiting),
        )


class PendingCheckpointMonitor:
    """Re-evaluates due checkpoints and acts on the ones that still want acting on.

    Holds a compiled graph only indirectly, through the `DurableWorkflowRunner`
    it was given, for the same reason the runner holds one rather than building
    it: which ports a graph is wired to is a deployment decision, and a monitor
    that constructed its own would have to know all of them.

    `clock` is injected so the trigger and expiry behaviour can be tested by
    moving time rather than waiting for it -- a sweep of a seven-day wait is
    otherwise an eight-day test.
    """

    def __init__(
        self,
        *,
        store: PendingCheckpointStore,
        evaluator: CheckpointConditionEvaluator,
        runner: DurableWorkflowRunner,
        registry: WorkflowThreadRegistry,
        clock=datetime.utcnow,
    ):
        """Wire the store, the condition evaluator and the runner that resumes threads."""
        if store is None:
            raise ValueError("PendingCheckpointMonitor requires a PendingCheckpointStore")
        if evaluator is None:
            raise ValueError("PendingCheckpointMonitor requires a condition evaluator")
        if runner is None:
            raise ValueError("PendingCheckpointMonitor requires a DurableWorkflowRunner")
        if registry is None:
            raise ValueError("PendingCheckpointMonitor requires a WorkflowThreadRegistry")
        self.store = store
        self.evaluator = evaluator
        self.runner = runner
        self.registry = registry
        self.clock = clock

    async def sweep(
        self, *, now: datetime | None = None, limit: int = DEFAULT_SWEEP_LIMIT
    ) -> SweepReport:
        """Decide about every actionable checkpoint, and act on each decision.

        `now` is taken once, at the top, and threaded through every decision in
        the sweep. A sweep that re-read the clock per checkpoint could decide
        that one checkpoint had expired and the next, scheduled a millisecond
        earlier, had not -- and the difference would be how long the previous
        checkpoint's graph invocation took, which is not a fact about either
        wait.
        """
        now = now or self.clock()
        buckets = _Buckets()

        for checkpoint in self.store.due(now=now, limit=limit):
            await self._handle(checkpoint, now=now, buckets=buckets)

        report = buckets.report()
        if report.considered:
            logger.info(
                "checkpoint sweep at %s: %s fired, %s resolved, %s expired, %s deferred, "
                "%s cancelled, %s failed",
                now.isoformat(),
                len(report.fired),
                len(report.resolved),
                len(report.expired),
                len(report.deferred),
                len(report.cancelled),
                len(report.failed),
            )
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _handle(
        self, checkpoint: PendingCheckpoint, *, now: datetime, buckets: _Buckets
    ) -> None:
        """Re-evaluate one checkpoint's condition and apply the outcome."""
        condition_met = await self.evaluator.is_met(checkpoint.condition)
        outcome = decide_checkpoint(
            checkpoint=checkpoint, condition_met=condition_met, now=now
        )

        if outcome is CheckpointOutcome.WAIT:  # pragma: no cover - `due` filters these out
            buckets.waiting.append(checkpoint.checkpoint_id)
            return

        if outcome is CheckpointOutcome.RESOLVE:
            # Silently: no event, no draft, no notification. The wait existed
            # because something had not happened; it has happened, so there is
            # nothing to tell anyone and nothing to do.
            if self.store.close(
                checkpoint,
                PendingCheckpointStatus.RESOLVED,
                reason=f"condition met before the trigger fired: {checkpoint.condition.describe()}",
                now=now,
            ):
                logger.info(
                    "checkpoint %s resolved without a follow-up (%s)",
                    checkpoint.checkpoint_id,
                    checkpoint.condition.describe(),
                )
                buckets.resolved.append(checkpoint.checkpoint_id)
            else:
                buckets.deferred.append(checkpoint.checkpoint_id)
            return

        if outcome is CheckpointOutcome.EXPIRE:
            if self.store.close(
                checkpoint,
                PendingCheckpointStatus.EXPIRED,
                reason=(
                    f"expiry {checkpoint.expires_at.isoformat()} passed without firing; "
                    f"a follow-up this late is not worth sending"
                ),
                now=now,
            ):
                logger.warning(
                    "checkpoint %s expired unfired (trigger %s, expiry %s)",
                    checkpoint.checkpoint_id,
                    checkpoint.trigger_at.isoformat(),
                    checkpoint.expires_at.isoformat(),
                )
                buckets.expired.append(checkpoint.checkpoint_id)
            else:
                buckets.deferred.append(checkpoint.checkpoint_id)
            return

        await self._fire(checkpoint, now=now, buckets=buckets)

    async def _fire(
        self, checkpoint: PendingCheckpoint, *, now: datetime, buckets: _Buckets
    ) -> None:
        """Start the graph path this checkpoint names, having first claimed it."""
        thread = self.registry.resolve(checkpoint.thread_id)
        if thread is None:
            self.store.close(
                checkpoint,
                PendingCheckpointStatus.CANCELLED,
                reason=(
                    f"thread '{checkpoint.thread_id}' is not registered to any workflow; "
                    f"there is nothing to resume"
                ),
                now=now,
            )
            logger.error(
                "cancelling checkpoint %s: thread '%s' is unregistered",
                checkpoint.checkpoint_id,
                checkpoint.thread_id,
            )
            buckets.cancelled.append(checkpoint.checkpoint_id)
            return

        state = await self.runner.inspect(thread)
        if state.checkpoint_id is None:
            # The wait names a thread that never ran. Cancelled rather than
            # fired: the follow-up path reads the application out of the
            # thread's stored state, and there is none.
            self.store.close(
                checkpoint,
                PendingCheckpointStatus.CANCELLED,
                reason=(
                    f"thread '{checkpoint.thread_id}' has no stored state to follow up on"
                ),
                now=now,
            )
            logger.error(
                "cancelling checkpoint %s: thread '%s' has never been checkpointed",
                checkpoint.checkpoint_id,
                checkpoint.thread_id,
            )
            buckets.cancelled.append(checkpoint.checkpoint_id)
            return

        if state.next:
            # Parked mid-run, almost certainly on an unanswered approval.
            # Deferred, not fired: a second request stacked on top of one
            # nobody has answered helps nobody, and this checkpoint's own
            # expiry is what eventually settles it.
            logger.info(
                "deferring checkpoint %s: thread '%s' is parked at %s",
                checkpoint.checkpoint_id,
                checkpoint.thread_id,
                ", ".join(state.next),
            )
            buckets.deferred.append(checkpoint.checkpoint_id)
            return

        # Claim before acting. The guarded UPDATE is what excludes a second
        # monitor that selected the same checkpoint in the same second.
        if not self.store.close(
            checkpoint,
            PendingCheckpointStatus.FIRED,
            reason=(
                f"trigger {checkpoint.trigger_at.isoformat()} reached with "
                f"{checkpoint.condition.describe()} still unmet"
            ),
            now=now,
        ):
            buckets.deferred.append(checkpoint.checkpoint_id)
            return

        try:
            await self.runner.start(thread, self.fire_input(checkpoint))
        except WorkflowLeaseUnavailable:
            # Another worker is inside this workflow. The claim is already
            # spent, so this follow-up is lost rather than retried -- logged
            # loudly for the same reason as any other post-claim failure.
            logger.error(
                "checkpoint %s was claimed but its workflow %s is leased elsewhere; "
                "the follow-up was not drafted",
                checkpoint.checkpoint_id,
                thread.workflow_id,
            )
            buckets.failed.append(checkpoint.checkpoint_id)
            return
        except Exception:
            logger.exception(
                "checkpoint %s was claimed but starting thread '%s' failed; the "
                "follow-up was not drafted",
                checkpoint.checkpoint_id,
                checkpoint.thread_id,
            )
            buckets.failed.append(checkpoint.checkpoint_id)
            return

        logger.info(
            "checkpoint %s fired; drafted a follow-up on thread '%s'",
            checkpoint.checkpoint_id,
            checkpoint.thread_id,
        )
        buckets.fired.append(checkpoint.checkpoint_id)

    @staticmethod
    def fire_input(checkpoint: PendingCheckpoint) -> dict[str, Any]:
        """The graph input that routes a run into the follow-up path.

        A `@staticmethod` rather than an inline dict because it is the contract
        between this module and `JobSearchGraph.route_from_start`: the key is
        what decides which way a run enters the graph, and it is spelled once.
        """
        return {"fired_checkpoints": [checkpoint.model_dump(mode="json")]}


class NeverMetConditionEvaluator:
    """An evaluator that answers `False` to everything, so every wait fires.

    The honest default for a deployment with nothing to ask. It is a named
    implementation rather than an `| None` on the constructor for the same
    reason `InterruptOnlyApprovalGate` is: "we cannot tell whether the
    recruiter replied, so we will follow up anyway" is a policy worth choosing
    on purpose, and its failure mode -- a redundant nudge -- is the recoverable
    one. An evaluator that answered `True` would silently cancel every
    follow-up in the system.
    """

    async def is_met(self, condition: CheckpointCondition) -> bool:
        """Report that nothing is known to have resolved this wait."""
        return False


__all__ = [
    "CheckpointConditionEvaluator",
    "NeverMetConditionEvaluator",
    "PendingCheckpointMonitor",
    "SweepReport",
]

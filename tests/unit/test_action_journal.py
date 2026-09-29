"""Contract tests for the journal that brackets an outward-facing action.

The property under test is the one the requirement states: state is persisted
before the side effect and again after its receipt, so that a crash between
those two points cannot produce a duplicate write on resume.

Crashing between them is simulated with a `BaseException` raised immediately
after the side effect. That is not a stand-in for convenience -- it is the
accurate one. `JournaledActionExecutor` catches `Exception` to record a
definite failure, so an ordinary error leaves a `failed` row, which is a
*known* outcome. A hard crash leaves neither the receipt nor the failure: the
claim, and nothing else. `BaseException` slips past the `except Exception` and
reproduces exactly that row, in-process and deterministically, where a real
`SIGKILL` is used in `tests/graph_scenarios/test_durable_resume.py` to prove the
same thing end to end.
"""

import pytest

from personalos.domain.job_search import (
    ActionIntent,
    ActionKind,
    ActionReceipt,
    ApprovalDecision,
    ApprovalVerdict,
)
from personalos.domain.models import ToolExecutionStatus
from personalos.persistence.action_journal import (
    TOOL_NAME_PREFIX,
    JournaledActionExecutor,
)
from personalos.persistence.repositories import ToolExecutionRepository
from tests.fixtures.durable_workflow import session_factory


class _Crash(BaseException):
    """A failure that unwinds past `except Exception`, like a process death.

    `BaseException` on purpose: the journal records ordinary exceptions as
    definite failures, and a test that raised one would be testing the
    failure path, not the crash path.
    """


class SpyExecutor:
    """Counts the side effects it performed, and can die right after one."""

    def __init__(self, *, ok: bool = True, crash_after: bool = False):
        self.ok = ok
        self.crash_after = crash_after
        self.calls: list[ActionIntent] = []

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        self.calls.append(intent)
        receipt = ActionReceipt(
            action_id=intent.action_id,
            ok=self.ok,
            external_reference=f"ext-{len(self.calls)}" if self.ok else None,
            detail=None if self.ok else "provider rejected it",
        )
        if self.crash_after:
            raise _Crash("worker died after submitting, before recording the receipt")
        return receipt


class RaisingExecutor:
    """An executor whose side effect fails outright, before doing anything."""

    def __init__(self):
        self.calls = 0

    async def execute(self, intent: ActionIntent, decision: ApprovalDecision) -> ActionReceipt:
        self.calls += 1
        raise RuntimeError("provider unreachable")


def _intent(key: str = "submit-acme-backend") -> ActionIntent:
    return ActionIntent(
        kind=ActionKind.SUBMIT_APPLICATION,
        target="https://example.test/acme/backend",
        summary="Submit an application to Acme for Backend Engineer",
        payload={"dedupe_key": "acme:backend"},
        idempotency_key=key,
    )


def _decision(intent: ActionIntent) -> ApprovalDecision:
    return ApprovalDecision(
        action_id=intent.action_id,
        action_fingerprint=intent.fingerprint(),
        verdict=ApprovalVerdict.APPROVED,
        decided_by="reviewer@example.test",
    )


# --- The happy path ----------------------------------------------------------


async def test_a_first_attempt_executes_and_records_its_receipt(tmp_path):
    factory = session_factory(tmp_path / "journal.db")
    inner = SpyExecutor()
    intent = _intent()

    receipt = await JournaledActionExecutor(inner, factory).execute(intent, _decision(intent))

    assert receipt.ok is True
    assert len(inner.calls) == 1

    session = factory()
    try:
        record = ToolExecutionRepository(session).get_by_key(intent.idempotency_key)
        assert record.status == ToolExecutionStatus.COMPLETED.value
        assert record.receipt_json["external_reference"] == "ext-1"
        assert record.tool_name == f"{TOOL_NAME_PREFIX}.submit_application"
    finally:
        session.close()


async def test_a_replay_returns_the_stored_receipt_without_acting_again(tmp_path):
    """The same action, attempted twice, happens once.

    The ordinary resume case: the first attempt completed and recorded its
    receipt, and the resumed run re-reaches the action. It must get the first
    attempt's outcome, not a second submission.
    """
    factory = session_factory(tmp_path / "journal.db")
    inner = SpyExecutor()
    intent = _intent()
    first = await JournaledActionExecutor(inner, factory).execute(intent, _decision(intent))

    # A new executor over a new session: the restarted process.
    replayed = await JournaledActionExecutor(
        inner, session_factory(tmp_path / "journal.db")
    ).execute(intent, _decision(intent))

    assert len(inner.calls) == 1, "the side effect was performed twice"
    assert replayed.ok is True
    assert replayed.external_reference == first.external_reference


async def test_a_replayed_receipt_is_rebound_to_the_intent_asking_for_it(tmp_path):
    """The receipt identifies the action it is settling, not the attempt that ran.

    `action_id` is minted per `ActionIntent`, so a fresh run proposing the same
    submission has a different one. A stored receipt handed back with the old id
    would not tie to the intent the rest of the run is carrying -- and
    `_receipt_for_kind` in the graph matches on exactly that.
    """
    factory = session_factory(tmp_path / "journal.db")
    inner = SpyExecutor()
    first = _intent()
    await JournaledActionExecutor(inner, factory).execute(first, _decision(first))

    # A second run proposes the same submission: same idempotency key, new id.
    second = _intent()
    assert second.action_id != first.action_id
    receipt = await JournaledActionExecutor(inner, factory).execute(second, _decision(second))

    assert receipt.action_id == second.action_id
    assert len(inner.calls) == 1


# --- The crash window --------------------------------------------------------


async def test_a_crash_between_the_side_effect_and_the_receipt_does_not_repeat_it(tmp_path):
    """The requirement, stated as a test.

    The first attempt submits and dies before recording anything. The resumed
    attempt finds a claim with no outcome and must not submit again: a duplicate
    application cannot be withdrawn, while a missed one can be resubmitted
    deliberately.
    """
    factory = session_factory(tmp_path / "journal.db")
    inner = SpyExecutor(crash_after=True)
    intent = _intent()

    with pytest.raises(_Crash):
        await JournaledActionExecutor(inner, factory).execute(intent, _decision(intent))
    assert len(inner.calls) == 1

    # The claim outlived the crash, with no outcome recorded against it.
    session = factory()
    try:
        record = ToolExecutionRepository(session).get_by_key(intent.idempotency_key)
        assert record.status == ToolExecutionStatus.IN_PROGRESS.value
        assert record.receipt_json is None
    finally:
        session.close()

    # The restarted worker re-reaches the same action.
    survivor = SpyExecutor()
    receipt = await JournaledActionExecutor(
        survivor, session_factory(tmp_path / "journal.db")
    ).execute(intent, _decision(intent))

    assert survivor.calls == [], "the resumed run submitted a second time"
    assert receipt.ok is False
    assert "did not record an outcome" in receipt.detail
    assert receipt.action_id == intent.action_id


async def test_the_claim_is_committed_before_the_side_effect_runs(tmp_path):
    """The 'before' half of the bracket, observed from outside the transaction.

    Read through a *separate* session while the side effect is still in progress:
    if the claim were sitting in an uncommitted transaction, another worker --
    or this one after a crash -- would not see it, and the guarantee would be
    imaginary.
    """
    factory = session_factory(tmp_path / "journal.db")
    observed: list[str | None] = []
    intent = _intent()

    class ObservingExecutor:
        async def execute(self, action_intent, decision):
            session = session_factory(tmp_path / "journal.db")()
            try:
                record = ToolExecutionRepository(session).get_by_key(action_intent.idempotency_key)
                observed.append(record.status if record else None)
            finally:
                session.close()
            return ActionReceipt(action_id=action_intent.action_id, ok=True)

    await JournaledActionExecutor(ObservingExecutor(), factory).execute(intent, _decision(intent))

    assert observed == [ToolExecutionStatus.IN_PROGRESS.value]


async def test_a_definite_failure_is_recorded_and_re_raised(tmp_path):
    """A side effect that failed outright is recorded as failed, and the error propagates.

    Distinct from the crash case: the executor raised, so the outcome is known,
    and the caller has to see it rather than getting a synthesized receipt.
    """
    factory = session_factory(tmp_path / "journal.db")
    inner = RaisingExecutor()
    intent = _intent()

    with pytest.raises(RuntimeError, match="provider unreachable"):
        await JournaledActionExecutor(inner, factory).execute(intent, _decision(intent))

    session = factory()
    try:
        record = ToolExecutionRepository(session).get_by_key(intent.idempotency_key)
        assert record.status == ToolExecutionStatus.FAILED.value
        assert "provider unreachable" in record.error
    finally:
        session.close()


async def test_a_failed_action_is_not_retried_automatically(tmp_path):
    """A recorded failure is not re-attempted by a later run on its own.

    The conservative reading, and the honest one: the executor raised, but
    whether the request reached the provider before it did is not knowable from
    here. Retrying would risk the duplicate this module exists to prevent, so the
    action comes back not-ok and a human decides.
    """
    factory = session_factory(tmp_path / "journal.db")
    intent = _intent()
    with pytest.raises(RuntimeError):
        await JournaledActionExecutor(RaisingExecutor(), factory).execute(intent, _decision(intent))

    retry = SpyExecutor()
    receipt = await JournaledActionExecutor(retry, factory).execute(intent, _decision(intent))

    assert retry.calls == []
    assert receipt.ok is False
    assert ToolExecutionStatus.FAILED.value in receipt.detail


async def test_a_not_ok_receipt_is_still_recorded_and_replayed(tmp_path):
    """A submission the provider rejected is a real, recorded outcome.

    It did happen -- it just did not succeed -- so a resumed run replays the
    rejection instead of trying again and possibly landing a second application.
    """
    factory = session_factory(tmp_path / "journal.db")
    inner = SpyExecutor(ok=False)
    intent = _intent()

    first = await JournaledActionExecutor(inner, factory).execute(intent, _decision(intent))
    second = await JournaledActionExecutor(inner, factory).execute(intent, _decision(intent))

    assert len(inner.calls) == 1
    assert first.ok is False
    assert second.ok is False
    assert second.detail == "provider rejected it"


# --- Independence ------------------------------------------------------------


async def test_different_actions_are_journaled_independently(tmp_path):
    """The journal keys on the intent's idempotency key, not on the action kind."""
    factory = session_factory(tmp_path / "journal.db")
    inner = SpyExecutor()

    for key in ("submit-acme-backend", "submit-globex-backend"):
        intent = _intent(key)
        await JournaledActionExecutor(inner, factory).execute(intent, _decision(intent))

    assert len(inner.calls) == 2


async def test_the_journal_requires_an_inner_executor(tmp_path):
    """A journal with nothing to journal is a silent no-op waiting to happen."""
    with pytest.raises(ValueError, match="requires an inner ActionExecutor"):
        JournaledActionExecutor(None, session_factory(tmp_path / "journal.db"))

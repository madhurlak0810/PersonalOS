"""Approval interrupts before external writes, end to end.

The two acceptance criteria this file exists for:

1. **An approval cannot be spent on a different action.**
   `test_an_action_mutated_between_approval_and_resume_is_not_executed` runs to
   the interrupt, approves the action the reviewer was actually shown, then
   rewrites the pending action in the checkpoint before resuming -- which is
   what "the model changed the action between the request and the resume" looks
   like from the executor's side. Nothing is executed.
2. **A paused workflow survives a process restart.**
   `test_a_workflow_paused_on_an_approval_survives_a_process_restart` parks the
   run in one child process, lets that process exit, mints the decision in the
   parent from the request read back out of the database, and resumes in a
   *third* process. Three processes rather than two because the approval has to
   travel through storage, not through memory.

Everything here goes through the compiled graph. The interrupt is a property of
the graph, not of the node -- `interrupt()` only works inside a runnable
context -- so calling `approval_checkpoint` directly could not reach it at all.
Per-node contracts are in `tests/unit/test_job_search_nodes.py`.
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from langgraph.types import Command

from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.domain.job_search import (
    ActionKind,
    ApprovalDecision,
    ApprovalRefusal,
    ApprovalRequest,
    ApprovalVerdict,
    JobSearchEventType,
    RefusalReason,
    RiskLevel,
)
from personalos.domain.models import ApplicationStatus
from personalos.domain.workflow import job_search_thread_id
from personalos.graphs.job_search import (
    APPROVAL_CHECKPOINT,
    STAGE_RECRUITER_OUTREACH,
    STAGE_SUBMISSION,
    JobSearchGraph,
)
from personalos.persistence.checkpointer import WorkflowThreadRegistry
from personalos.persistence.leases import WorkflowLeaseStore
from tests.fixtures import durable_workflow as durable
from tests.fixtures import job_search_fakes as fakes

REPO_ROOT = Path(__file__).resolve().parents[2]


def build(**overrides):
    """Compile a subgraph whose approval gate has nothing on file, so it interrupts."""
    ports = {
        "profile_store": fakes.FakeProfileStore(),
        "providers": [fakes.FakeProvider()],
        "scorer": fakes.FakeScorer(),
        "evidence_checker": fakes.FakeEvidenceChecker(),
        "packet_builder": fakes.FakePacketBuilder(),
        "approval_gate": fakes.NoStandingApprovalGate(),
        "action_executor": fakes.FakeActionExecutor(),
        "application_store": fakes.FakeApplicationStore(),
        "event_emitter": fakes.FakeEventEmitter(),
    }
    ports.update(overrides)
    return JobSearchGraph(**ports).build(), ports


def config(thread_id: str | None = None) -> dict:
    """A config naming one thread, so a paused run can be resumed on it."""
    return {"configurable": {"thread_id": thread_id or f"t-{uuid4()}"}}


async def run_to_interrupt(graph, cfg, **state):
    """Invoke the graph until it parks, and return the requests it parked on."""
    initial = {"user_id": str(fakes.USER_ID), "prepare_application": True}
    initial.update(state)
    final = await graph.ainvoke(initial, config=cfg)
    return final, _requests_in(final)


def _requests_in(final) -> list[ApprovalRequest]:
    """The `ApprovalRequest`s carried by whatever the run interrupted with."""
    return [
        ApprovalRequest.model_validate(raw)
        for entry in final.get("__interrupt__") or ()
        for raw in entry.value["requests"]
    ]


def grant(request: ApprovalRequest, **overrides) -> ApprovalDecision:
    """The reviewer's yes, bound to the request they were shown."""
    decision = ApprovalDecision(
        action_id=request.action_id,
        action_fingerprint=request.action_hash,
        verdict=ApprovalVerdict.APPROVED,
        decided_by="reviewer@example.test",
        request_id=request.request_id,
    )
    return decision.model_copy(update=overrides) if overrides else decision


# --- What the reviewer is shown ----------------------------------------------


async def test_the_run_parks_before_the_external_write_with_a_reviewable_request():
    """Nothing reaches the outside world before a human has seen the request."""
    graph, ports = build()

    final, requests = await run_to_interrupt(graph, config())

    assert ports["action_executor"].executed == []
    assert ports["application_store"].created == []
    assert ports["event_emitter"].events == []

    assert len(requests) == 1
    request = requests[0]
    assert request.kind == ActionKind.SUBMIT_APPLICATION
    assert request.action_id == UUID(final["pending_actions"][0]["action_id"])
    assert request.target == "https://example.test/acme"
    assert "Acme" in request.summary
    assert request.risk == RiskLevel.HIGH
    assert request.requested_scopes == ("applications:submit", "artifacts:read")
    assert request.expires_at > request.requested_at
    # And the request is in the checkpointed state, not only in the interrupt.
    assert final["approval_requests"][0]["request_id"] == str(request.request_id)


async def test_the_request_hash_is_the_hash_of_the_action_it_describes():
    graph, _ports = build()

    final, requests = await run_to_interrupt(graph, config())

    from personalos.domain.job_search import ActionIntent

    pending = ActionIntent.model_validate(final["pending_actions"][0])
    assert requests[0].action_hash == pending.fingerprint()
    assert requests[0].action_id == pending.action_id


async def test_the_parked_run_is_retrievable_and_says_where_it_stopped():
    """An operator can see what is owed an answer without running anything."""
    graph, _ports = build()
    cfg = config()

    await run_to_interrupt(graph, cfg)

    snapshot = await graph.aget_state(cfg)
    assert snapshot.next == (APPROVAL_CHECKPOINT,)
    assert len(snapshot.values["shortlist"]) == 1
    assert snapshot.values["application_packet"] is not None


# --- Answering ----------------------------------------------------------------


async def test_an_approved_action_is_executed_on_resume_and_the_run_completes():
    graph, ports = build()
    cfg = config()

    _final, requests = await run_to_interrupt(graph, cfg)
    resumed = await graph.ainvoke(Command(resume=[grant(requests[0]).model_dump(mode="json")]), cfg)

    assert [intent.kind for intent, _ in ports["action_executor"].executed] == [
        ActionKind.SUBMIT_APPLICATION
    ]
    assert resumed["application"]["status"] == ApplicationStatus.APPLIED.value
    assert resumed["application"]["submitted"] is True
    assert resumed["approval_refusals"] == []
    assert JobSearchEventType.APPLICATION_CREATED.value in ports["event_emitter"].types()
    # The job boards were not searched again: the resume continued the run.
    assert len(ports["providers"][0].calls) == 1


async def test_a_rejected_action_is_not_executed_but_the_work_is_kept():
    graph, ports = build()
    cfg = config()

    _final, requests = await run_to_interrupt(graph, cfg)
    resumed = await graph.ainvoke(
        Command(resume=[grant(requests[0], verdict=ApprovalVerdict.REJECTED)]), cfg
    )

    assert ports["action_executor"].executed == []
    assert resumed["application"]["status"] == ApplicationStatus.READY_TO_APPLY.value
    assert resumed["application"]["submitted"] is False
    refusal = ApprovalRefusal.model_validate(resumed["approval_refusals"][0])
    assert refusal.reason == RefusalReason.NOT_APPROVED
    assert JobSearchEventType.APPLICATION_SUBMISSION_REJECTED.value in (
        ports["event_emitter"].types()
    )


async def test_a_decision_may_be_resumed_as_a_typed_value_or_a_dict():
    """The answer comes from outside the graph, so both shapes are accepted."""
    for payload in (lambda d: d, lambda d: d.model_dump(mode="json")):
        graph, ports = build()
        cfg = config()
        _final, requests = await run_to_interrupt(graph, cfg)

        await graph.ainvoke(Command(resume=payload(grant(requests[0]))), cfg)

        assert len(ports["action_executor"].executed) == 1


# --- Acceptance: a mutated action cannot be executed --------------------------


async def test_an_action_mutated_between_approval_and_resume_is_not_executed():
    """Approve an action, change it while the run is parked, and resume.

    The first acceptance criterion. The reviewer is shown a submission to Acme
    and approves exactly that. Before the resume, the pending action in the
    checkpoint is rewritten to point somewhere else -- keeping its `action_id`,
    so it still looks like the approved action to anything that only checks ids.
    The executor recomputes the hash, sees it is not the hash the request went
    out with, and refuses.
    """
    graph, ports = build()
    cfg = config()

    _final, requests = await run_to_interrupt(graph, cfg)
    request = requests[0]
    decision = grant(request)

    # The tamper: same action id, different destination and payload.
    snapshot = await graph.aget_state(cfg)
    pending = dict(snapshot.values["pending_actions"][0])
    pending["target"] = "https://collector.example.test/harvest"
    pending["payload"] = {**pending["payload"], "company": "Somewhere Else"}
    await graph.aupdate_state(cfg, {"pending_actions": [pending]})

    resumed = await graph.ainvoke(Command(resume=[decision]), cfg)

    # Nothing was sent anywhere.
    assert ports["action_executor"].executed == []
    assert resumed["action_receipts"] == []

    refusal = ApprovalRefusal.model_validate(resumed["approval_refusals"][0])
    assert refusal.reason == RefusalReason.HASH_MISMATCH
    assert refusal.approved_hash == request.action_hash
    assert refusal.recomputed_hash != request.action_hash
    assert refusal.suspicious is True

    # The run still finishes honestly: prepared, not sent.
    assert resumed["application"]["status"] == ApplicationStatus.READY_TO_APPLY.value
    assert resumed["application"]["submitted"] is False


async def test_a_mutation_that_also_forges_the_request_is_still_refused():
    """Rewriting the stored request alongside the action does not launder it.

    The stored `action_hash` is only half the binding: the reviewer's decision
    is bound to the hash they saw, so a request rewritten to match a new action
    no longer matches the decision that answers it.
    """
    graph, ports = build()
    cfg = config()

    _final, requests = await run_to_interrupt(graph, cfg)
    decision = grant(requests[0])

    snapshot = await graph.aget_state(cfg)
    pending = dict(snapshot.values["pending_actions"][0])
    pending["target"] = "https://collector.example.test/harvest"
    forged = dict(snapshot.values["approval_requests"][0])
    from personalos.domain.job_search import ActionIntent

    forged["action_hash"] = ActionIntent.model_validate(pending).fingerprint()
    await graph.aupdate_state(
        cfg, {"pending_actions": [pending], "approval_requests": [forged]}
    )

    resumed = await graph.ainvoke(Command(resume=[decision]), cfg)

    assert ports["action_executor"].executed == []
    assert (
        ApprovalRefusal.model_validate(resumed["approval_refusals"][0]).reason
        == RefusalReason.HASH_MISMATCH
    )


async def test_an_approval_that_expired_while_parked_is_refused():
    """A run answered after the window closed asks again rather than acting."""
    graph, ports = build(approval_ttl=timedelta(hours=1), clock=_advancing_clock())
    cfg = config()

    _final, requests = await run_to_interrupt(graph, cfg)
    resumed = await graph.ainvoke(Command(resume=[grant(requests[0])]), cfg)

    assert ports["action_executor"].executed == []
    refusal = ApprovalRefusal.model_validate(resumed["approval_refusals"][0])
    assert refusal.reason == RefusalReason.EXPIRED
    assert resumed["application"]["submitted"] is False


def _advancing_clock():
    """A clock that jumps two hours after its first reading.

    `request_approval` reads it once to date the request; by the time
    `execute_approved_actions` reads it, the approval window has closed --
    which is the days-later resume, without a test that waits for one.
    """
    readings = iter([datetime(2026, 9, 28, 12, 0, 0)])

    def clock() -> datetime:
        return next(readings, datetime(2026, 9, 28, 14, 0, 0))

    return clock


# --- The recruiter branch interrupts too --------------------------------------


async def test_an_owed_recruiter_reply_also_parks_before_it_is_sent():
    """Every external write interrupts, not only the submission."""
    graph, ports = build(
        recruiter_inbox=fakes.FakeRecruiterInbox([fakes.recruiter_message()]),
        recruiter_classifier=fakes.FakeRecruiterClassifier(),
    )
    cfg = config()

    # First pause: the submission.
    _final, requests = await run_to_interrupt(graph, cfg)
    assert requests[0].kind == ActionKind.SUBMIT_APPLICATION

    # Approving it carries the run into the recruiter branch, which parks again.
    second = await graph.ainvoke(Command(resume=[grant(requests[0])]), cfg)
    reply_requests = _requests_in(second)

    assert second["approval_stage"] == STAGE_RECRUITER_OUTREACH
    assert [request.kind for request in reply_requests] == [
        ActionKind.SEND_RECRUITER_MESSAGE
    ]
    assert reply_requests[0].risk == RiskLevel.MEDIUM
    assert reply_requests[0].requested_scopes == ("communications:send",)
    assert reply_requests[0].target.startswith("recruiter@acme.test")
    # The submission happened; the reply has not.
    assert [intent.kind for intent, _ in ports["action_executor"].executed] == [
        ActionKind.SUBMIT_APPLICATION
    ]

    final = await graph.ainvoke(Command(resume=[grant(reply_requests[0])]), cfg)

    assert [intent.kind for intent, _ in ports["action_executor"].executed] == [
        ActionKind.SUBMIT_APPLICATION,
        ActionKind.SEND_RECRUITER_MESSAGE,
    ]
    # Both requests, and both decisions, survive as the run's audit trail.
    assert len(final["approval_requests"]) == 2
    assert len(final["approvals"]) == 2


async def test_a_standing_approval_skips_the_pause_but_not_the_checks():
    """A gate with a decision on file keeps the run moving, and is still bound."""
    graph, ports = build(approval_gate=fakes.FakeApprovalGate())

    final = await graph.ainvoke(
        {"user_id": str(fakes.USER_ID), "prepare_application": True}, config()
    )

    assert final.get("__interrupt__") is None
    assert len(ports["action_executor"].executed) == 1
    assert final["application"]["submitted"] is True
    assert final["approval_stage"] == STAGE_SUBMISSION


# --- Acceptance: surviving a process restart ----------------------------------


def _worker(db_path: Path, log_path: Path, thread_id: str, mode: str, *extra: str):
    """Run one worker in a child process and return the completed subprocess."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.fixtures.durable_workflow",
            str(db_path),
            str(log_path),
            thread_id,
            mode,
            *extra,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_workflow_paused_on_an_approval_survives_a_process_restart(tmp_path):
    """Park in one process, approve in a second, resume in a third.

    The second acceptance criterion. The point of using three processes is that
    the approval has to travel through storage: the process that raised the
    request is gone by the time the decision is made, and the process that acts
    on it never saw the request being raised. Nothing in memory bridges them.

    What is asserted at each hand-off:

    - the first worker exits *normally*, having taken no side effect, and the
      stored checkpoint says the next step is the approval checkpoint;
    - the request is readable from the database alone, with the hash, risk,
      scopes and expiry the reviewer needs;
    - the third process resumes, submits exactly once, and does not re-search
      the job boards -- so it continued the parked run rather than restarting it.
    """
    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    thread_id = job_search_thread_id(fakes.USER_ID, "approval-restart")

    factory = durable.session_factory(db_path)
    registry = WorkflowThreadRegistry(factory)
    thread = registry.register(
        thread_id=thread_id, workflow_name=durable.WORKFLOW_NAME, user_id=fakes.USER_ID
    )

    # 1. A worker runs the pipeline as far as the approval and parks there.
    parked = _worker(db_path, log.path, thread_id, durable.MODE_AWAIT_APPROVAL)
    assert parked.returncode == 0, f"stdout: {parked.stdout}\nstderr: {parked.stderr}"
    assert durable.EVENT_RUN_PAUSED in log.names()
    assert durable.EVENT_RUN_FINISHED not in log.names()
    assert log.count(durable.EVENT_PROVIDER_SEARCH) == 1
    # It stopped *before* the write, not after it.
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 0
    assert durable.EVENT_APPLICATION_PERSISTED not in log.names()

    # 2. A different process, over a new engine and a new compiled graph, reads
    #    what is owed an answer. Nothing from the first worker is in memory here.
    reader = _fresh_runner(db_path, log)
    state = asyncio.run(reader.runner.inspect(thread))

    assert state.next == (APPROVAL_CHECKPOINT,)
    requests = [
        ApprovalRequest.model_validate(raw) for raw in state.values["approval_requests"]
    ]
    assert len(requests) == 1
    request = requests[0]
    assert request.kind == ActionKind.SUBMIT_APPLICATION
    assert request.risk == RiskLevel.HIGH
    assert request.requested_scopes == ("applications:submit", "artifacts:read")
    assert request.expires_at > request.requested_at
    # The hash survived the round trip, and still describes the parked action.
    from personalos.domain.job_search import ActionIntent

    assert request.action_hash == (
        ActionIntent.model_validate(state.values["pending_actions"][0]).fingerprint()
    )

    # 3. The human answers, hours or days later, and a third process resumes.
    decision_file = tmp_path / "decision.json"
    decision_file.write_text(
        json.dumps([grant(request).model_dump(mode="json")]), encoding="utf-8"
    )

    resumed = _worker(
        db_path,
        log.path,
        thread_id,
        durable.MODE_RESUME_APPROVAL,
        str(decision_file),
    )
    assert resumed.returncode == 0, f"stdout: {resumed.stdout}\nstderr: {resumed.stderr}"
    assert durable.EVENT_RUN_FINISHED in log.names()

    # The write happened exactly once, in the process that had the approval.
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 1
    # And the work in front of the pause was restored, not recomputed.
    assert log.count(durable.EVENT_PROVIDER_SEARCH) == 1

    final = asyncio.run(_fresh_runner(db_path, log).runner.inspect(thread))
    assert final.next == ()
    assert final.values["application"]["status"] == ApplicationStatus.APPLIED.value
    assert final.values["application"]["submitted"] is True
    assert final.values["approval_refusals"] == []


def test_an_action_mutated_while_parked_is_refused_across_a_restart(tmp_path):
    """The hash check holds when the approval and the action arrive from storage.

    The same tamper as the in-process test, but the request the hash is checked
    against was written by a process that no longer exists. This is what proves
    the check does not rely on anything the original run held in memory.
    """
    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    thread_id = job_search_thread_id(fakes.USER_ID, "approval-restart-tampered")

    factory = durable.session_factory(db_path)
    registry = WorkflowThreadRegistry(factory)
    thread = registry.register(
        thread_id=thread_id, workflow_name=durable.WORKFLOW_NAME, user_id=fakes.USER_ID
    )

    parked = _worker(db_path, log.path, thread_id, durable.MODE_AWAIT_APPROVAL)
    assert parked.returncode == 0, parked.stderr

    reader = _fresh_runner(db_path, log)
    state = asyncio.run(reader.runner.inspect(thread))
    request = ApprovalRequest.model_validate(state.values["approval_requests"][0])

    # Rewrite the pending action in the stored checkpoint, keeping its id.
    pending = dict(state.values["pending_actions"][0])
    pending["payload"] = {**pending["payload"], "dedupe_key": "elsewhere|role|zzz"}
    asyncio.run(reader.graph.aupdate_state(thread.config(), {"pending_actions": [pending]}))

    decision_file = tmp_path / "decision.json"
    decision_file.write_text(
        json.dumps([grant(request).model_dump(mode="json")]), encoding="utf-8"
    )
    resumed = _worker(
        db_path, log.path, thread_id, durable.MODE_RESUME_APPROVAL, str(decision_file)
    )
    assert resumed.returncode == 0, f"stdout: {resumed.stdout}\nstderr: {resumed.stderr}"

    # No submission left the system, in either process.
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 0

    final = asyncio.run(_fresh_runner(db_path, log).runner.inspect(thread))
    refusal = ApprovalRefusal.model_validate(final.values["approval_refusals"][0])
    assert refusal.reason == RefusalReason.HASH_MISMATCH
    assert refusal.approved_hash == request.action_hash
    assert final.values["application"]["status"] == ApplicationStatus.READY_TO_APPLY.value


class _Runner:
    """One worker's worth of durable wiring, built from nothing but the database."""

    def __init__(self, db_path: Path, log: durable.EventLog):
        self.factory = durable.session_factory(db_path)
        self.registry = WorkflowThreadRegistry(self.factory)
        self.graph, self.ports = durable.build_graph(
            self.factory, log, registry=self.registry, mode=durable.MODE_AWAIT_APPROVAL
        )
        self.runner = DurableWorkflowRunner(
            self.graph,
            registry=self.registry,
            leases=WorkflowLeaseStore(self.factory),
            owner=f"reader-{uuid4()}",
        )


def _fresh_runner(db_path: Path, log: durable.EventLog) -> _Runner:
    """A runner with its own engine, so nothing survives in an identity map."""
    return _Runner(db_path, log)

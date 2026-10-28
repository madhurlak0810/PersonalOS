"""Whole workflows through the failures they are built to survive.

The per-feature scenario files each prove one mechanism. This file runs the
five situations those mechanisms exist for, end to end, against real stores
and fake providers, with every model answer scripted:

1. **A duplicate email event arrives twice.** Exactly one application
   transition, one stored communication event, and one follow-up closed --
   not one per delivery.
2. **A calendar create succeeds but its response is lost.** The run fails at
   the write; a restarted worker retries the step, finds the event the first
   attempt left behind, and creates only the ones that never landed.
3. **The process restarts while waiting for approval.** The workflow resumes
   from its checkpoint and re-raises *the same* approval request -- same id,
   same hash, same expiry -- rather than minting a new one.
4. **The follow-up trigger fires after the recruiter replied.** The condition
   the trigger is gated on comes back "the recruiter replied", and nothing is
   drafted.
5. **The local model returns malformed JSON.** The fallback takes over, the
   model failure is on record, and no action is executed.

Each scenario is a `run_*` coroutine that makes its own assertions and records
every outward effect it can observe into a `SideEffectLedger`. The individual
tests run one each; `test_duplicate_side_effect_rate_stays_zero_across_the_
scenario_suite` runs them all into one ledger and checks the acceptance
criterion: `duplicate_side_effect_rate` is exactly zero.
"""

import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from langgraph.types import Command

from apps.worker.workflow_runner import DurableWorkflowRunner
from personalos.bootstrap import (
    build_calendar_action_executor,
    build_checkpoint_condition_evaluator,
)
from personalos.domain.checkpoints import ConditionKind, PendingCheckpointStatus
from personalos.domain.interview_scheduling import (
    PROP_IDEMPOTENCY_KEY,
    interview_event_key,
    prep_block_key,
)
from personalos.domain.job_search import (
    ActionKind,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalVerdict,
    JobSearchEventType,
    RecruiterMessage,
)
from personalos.domain.models import APPLICATION_STATUS_CHANGED_EVENT, ApplicationStatus
from personalos.domain.recruiter_events import CommitmentActor, ExtractedCommitment
from personalos.domain.workflow import job_search_thread_id, recruiter_inbox_thread_id
from personalos.graphs.job_search import APPROVAL_CHECKPOINT, JobSearchGraph
from personalos.graphs.supervisor import SupervisorGraph
from personalos.models.recruiter_events import (
    DEFAULT_MAX_ATTEMPTS,
    StructuredLLMRecruiterEventExtractor,
)
from personalos.models.routing import StructuredLLMIntentClassifier
from personalos.persistence.checkpointer import (
    SqlAlchemyCheckpointSaver,
    WorkflowThreadRegistry,
)
from personalos.persistence.leases import WorkflowLeaseStore
from personalos.persistence.models import CommunicationEventModel, EventLogModel
from tests.fixtures import durable_workflow as durable
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.side_effects import (
    APPLICATION_TRANSITION,
    CALENDAR_CREATE,
    COMMUNICATION_EVENT,
    EXTERNAL_WRITE,
    FOLLOW_UP_CLOSED,
    FOLLOW_UP_DRAFTED,
    SideEffectLedger,
)
from tests.graph_scenarios import test_conditional_follow_ups as follow_ups
from tests.graph_scenarios import test_recruiter_events as inbound

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- Shared observation ---------------------------------------------------------


def _rows(factory, model, **filters) -> list[dict]:
    session = factory()
    try:
        query = session.query(model)
        for column, value in filters.items():
            query = query.filter(getattr(model, column) == value)
        return [row.to_dict() for row in query.all()]
    finally:
        session.close()


def _transitions(factory, application_id) -> list[dict]:
    """Every status change the lifecycle logged for one application, in order."""
    return [
        row["payload_json"]
        for row in _rows(
            factory,
            EventLogModel,
            aggregate_id=application_id,
            event_type=APPLICATION_STATUS_CHANGED_EVENT,
        )
    ]


def _record_database_effects(ledger: SideEffectLedger, factory, application_id) -> None:
    """File the transitions and communication events one database holds."""
    ledger.record_all(
        APPLICATION_TRANSITION,
        (
            (str(application_id), change["from"], change["to"])
            for change in _transitions(factory, application_id)
        ),
    )
    ledger.record_all(
        COMMUNICATION_EVENT,
        (
            row["provider_message_id"]
            for row in _rows(factory, CommunicationEventModel, application_id=application_id)
        ),
    )


def _record_sweep(ledger: SideEffectLedger, report) -> None:
    """File what one monitor sweep closed or started."""
    ledger.record_all(FOLLOW_UP_CLOSED, (*report.resolved, *report.expired, *report.cancelled))
    ledger.record_all(FOLLOW_UP_DRAFTED, report.fired)


def _record_executions(ledger: SideEffectLedger, executor: fakes.FakeActionExecutor) -> None:
    ledger.record_all(EXTERNAL_WRITE, (intent.idempotency_key for intent, _ in executor.executed))


def _grant(request: ApprovalRequest) -> ApprovalDecision:
    """The reviewer's yes, bound to the request they were shown."""
    return ApprovalDecision(
        action_id=request.action_id,
        action_fingerprint=request.action_hash,
        verdict=ApprovalVerdict.APPROVED,
        decided_by="reviewer@example.test",
        request_id=request.request_id,
    )


# --- 1. A duplicate email event arrives twice -----------------------------------

#: A decision email: it moves the application, and it is a recruiter response,
#: so it also answers the no-response follow-up the application is waiting on.
REJECTION = RecruiterMessage(
    provider_message_id="<decision-1@northwind.test>",
    received_at=follow_ups.REPLIED_AT,
    subject="Re: your application for REQ-48213 at Northwind",
    from_address="priya@northwind.test",
    body=(
        "Hi Sam,\n\nThank you for your time. Unfortunately we will not be moving "
        "forward with your application.\n\nPriya"
    ),
    thread_id="thread-northwind-1",
)


async def run_duplicate_email(tmp_path: Path, ledger: SideEffectLedger) -> None:
    """Apply, then receive one rejection three times: twice on the inbox thread
    (a webhook retried) and once on a second worker's thread (the same webhook
    picked up twice). Two sweeps follow, as two monitor ticks would."""
    deployment = follow_ups.Deployment(tmp_path)
    wait = await deployment.apply()
    second_worker = deployment.registry.register(
        thread_id=job_search_thread_id(deployment.user_id, "inbox-redelivery"),
        workflow_name="job_search",
        user_id=deployment.user_id,
    )

    deployment.now = follow_ups.REPLIED_AT
    delivery = {
        "user_id": str(deployment.user_id),
        "inbound_messages": [REJECTION.model_dump(mode="json")],
    }
    runs = [
        await deployment.runner.start(thread, delivery)
        for thread in (deployment.inbox_thread, deployment.inbox_thread, second_worker)
    ]

    sweeps = [
        await deployment.monitor.sweep(now=follow_ups.TRIGGER_AT),
        await deployment.monitor.sweep(now=follow_ups.TRIGGER_AT + timedelta(hours=1)),
    ]

    # One transition, made by the first delivery.
    recruiter_moves = [
        (change["from"], change["to"])
        for change in _transitions(deployment.factory, deployment.application_id)
        if change["actor"] == "recruiter_event"
    ]
    assert recruiter_moves == [("applied", "rejected")]
    assert deployment.replies_on_file() == 1
    assert [r["created"] for r in runs[0]["recruiter_event_records"]] == [True]
    for rerun in runs[1:]:
        assert [r["created"] for r in rerun["recruiter_event_records"]] == [False]
        assert rerun["pending_actions"] == []

    # One follow-up closed, by the first sweep; the second found nothing to do.
    assert sweeps[0].resolved == (wait.checkpoint_id,)
    assert sweeps[0].fired == ()
    assert sweeps[1].considered == 0
    (closed,) = deployment.store.history_for_application(deployment.application_id)
    assert closed.status == PendingCheckpointStatus.RESOLVED

    # And none of it reached anyone outside: the submission is the only write.
    assert JobSearchEventType.FOLLOW_UP_TRIGGERED.value not in deployment.emitter.types()
    assert [i.kind for i, _ in deployment.executor.executed] == [ActionKind.SUBMIT_APPLICATION]

    _record_database_effects(ledger, deployment.factory, deployment.application_id)
    for report in sweeps:
        _record_sweep(ledger, report)
    _record_executions(ledger, deployment.executor)


async def test_a_duplicate_email_makes_one_transition_and_one_follow_up_cancellation(tmp_path):
    ledger = SideEffectLedger()
    await run_duplicate_email(tmp_path, ledger)
    assert ledger.duplicate_side_effect_rate == 0.0, ledger.duplicated()


# --- 2. A calendar create succeeds but the response is lost ---------------------

#: Two days after `fakes.NOW`, mid-afternoon.
INTERVIEW_AT = datetime(2026, 10, 1, 14, 0)
CALENDAR_KEYS = (
    interview_event_key(fakes.APPLICATION_ID),
    prep_block_key(fakes.APPLICATION_ID, "deep_prep"),
    prep_block_key(fakes.APPLICATION_ID, "warm_up"),
)


class CalendarDeployment:
    """One calendar and one database; each `worker()` is a fresh process's wiring."""

    def __init__(self, tmp_path: Path):
        self.db_path = tmp_path / "calendar.db"
        self.calendar = fakes.FakeCalendar()
        registry = WorkflowThreadRegistry(durable.session_factory(self.db_path))
        self.thread = registry.register(
            thread_id=recruiter_inbox_thread_id(fakes.USER_ID),
            workflow_name=durable.WORKFLOW_NAME,
            user_id=fakes.USER_ID,
        )
        self.config = self.thread.config()

    def worker(self):
        """A graph on a new engine, executor and checkpointer -- nothing in memory
        survives from an earlier worker except what the calendar and the
        database actually hold."""
        factory = durable.session_factory(self.db_path)
        invite = fakes.extraction(
            commitments=[
                ExtractedCommitment(
                    actor=CommitmentActor.EXTERNAL_PERSON,
                    action="Technical interview with the team",
                    due_at=INTERVIEW_AT,
                    confidence=0.9,
                )
            ]
        )
        return JobSearchGraph(
            profile_store=fakes.FakeProfileStore(),
            providers=[fakes.FakeProvider()],
            scorer=fakes.FakeScorer(),
            evidence_checker=fakes.FakeEvidenceChecker(),
            packet_builder=fakes.FakePacketBuilder(),
            approval_gate=fakes.NoStandingApprovalGate(),
            action_executor=build_calendar_action_executor(
                self.calendar, factory, workflow_id=self.thread.workflow_id
            ),
            application_store=fakes.FakeApplicationStore(),
            event_emitter=fakes.FakeEventEmitter(),
            recruiter_event_extractor=fakes.FakeRecruiterEventExtractor(
                by_message={"invite": invite}
            ),
            application_directory=fakes.FakeApplicationDirectory(),
            recruiter_event_recorder=fakes.FakeRecruiterEventRecorder(),
            calendar_reader=self.calendar,
            checkpoint_scheduler=fakes.FakePendingCheckpointScheduler(),
            checkpointer=SqlAlchemyCheckpointSaver(factory, WorkflowThreadRegistry(factory)),
            clock=lambda: fakes.NOW,
        ).build()


async def run_lost_calendar_response(tmp_path: Path, ledger: SideEffectLedger) -> None:
    """Approve three calendar writes; the first insert lands and its response is
    lost, killing the run. A restarted worker retries the step."""
    deployment = CalendarDeployment(tmp_path)
    graph = deployment.worker()
    await graph.ainvoke(
        {
            "user_id": str(fakes.USER_ID),
            "inbound_messages": [fakes.inbound_message("invite").model_dump(mode="json")],
        },
        deployment.config,
    )
    parked = await graph.aget_state(deployment.config)
    assert parked.next == (APPROVAL_CHECKPOINT,)
    requests = [ApprovalRequest.model_validate(raw) for raw in parked.values["approval_requests"]]
    assert [r.kind for r in requests] == [ActionKind.CREATE_CALENDAR_EVENT] * 3

    deployment.calendar.lose_next_create_response = True
    with pytest.raises(fakes.LostResponse):
        await graph.ainvoke(
            Command(resume=[_grant(r).model_dump(mode="json") for r in requests]),
            deployment.config,
        )
    # The insert did land; the run never heard, and stopped at the write.
    (landed,) = deployment.calendar.created
    assert deployment.calendar.events[landed].logical_key in CALENDAR_KEYS
    assert (await graph.aget_state(deployment.config)).next != ()

    # A restarted worker continues the stored run: the same step, the same
    # three approved actions, against what the calendar now holds.
    final = await deployment.worker().ainvoke(None, deployment.config)

    assert len(deployment.calendar.events) == 3
    assert len(deployment.calendar.created) == 3
    assert deployment.calendar.created[0] == landed
    for key in CALENDAR_KEYS:
        deployment.calendar.by_key(key)  # exactly one event per logical key
    receipts = final["action_receipts"]
    assert [r["ok"] for r in receipts] == [True, True, True]
    assert landed in {r["external_reference"] for r in receipts}
    assert final["approval_refusals"] == []
    assert (await deployment.worker().aget_state(deployment.config)).next == ()

    ledger.record_all(
        CALENDAR_CREATE,
        (
            deployment.calendar.events[event_id].properties[PROP_IDEMPOTENCY_KEY]
            for event_id in deployment.calendar.created
        ),
    )


async def test_a_lost_calendar_response_is_reconciled_and_not_duplicated(tmp_path):
    ledger = SideEffectLedger()
    await run_lost_calendar_response(tmp_path, ledger)
    assert ledger.duplicate_side_effect_rate == 0.0, ledger.duplicated()


# --- 3. A restart while waiting for approval ------------------------------------


class Worker:
    """A worker's durable wiring, built from nothing but the database file."""

    def __init__(self, db_path: Path, log: durable.EventLog):
        factory = durable.session_factory(db_path)
        registry = WorkflowThreadRegistry(factory)
        self.graph, _ports = durable.build_graph(
            factory, log, registry=registry, mode=durable.MODE_AWAIT_APPROVAL
        )
        self.runner = DurableWorkflowRunner(
            self.graph,
            registry=registry,
            leases=WorkflowLeaseStore(factory),
            owner=f"worker-{uuid4()}",
        )


def _park_in_a_child_process(db_path: Path, log: durable.EventLog, thread_id: str):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.fixtures.durable_workflow",
            str(db_path),
            str(log.path),
            thread_id,
            durable.MODE_AWAIT_APPROVAL,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


async def run_restart_while_awaiting_approval(tmp_path: Path, ledger: SideEffectLedger) -> None:
    """Park in a child process that then exits; restart and continue the run
    with no answer; answer it from a third worker."""
    db_path = tmp_path / "durable.db"
    log = durable.EventLog(tmp_path / "events.jsonl")
    thread_id = job_search_thread_id(fakes.USER_ID, "restart-awaiting-approval")
    thread = WorkflowThreadRegistry(durable.session_factory(db_path)).register(
        thread_id=thread_id, workflow_name=durable.WORKFLOW_NAME, user_id=fakes.USER_ID
    )

    parked = _park_in_a_child_process(db_path, log, thread_id)
    assert parked.returncode == 0, f"stdout: {parked.stdout}\nstderr: {parked.stderr}"
    assert log.names()[-1] == durable.EVENT_RUN_PAUSED
    stored = await Worker(db_path, log).runner.inspect(thread)
    (request,) = [ApprovalRequest.model_validate(r) for r in stored.values["approval_requests"]]

    # The process restarts and picks the workflow back up with nothing new to
    # say. It must land on the same approval, not on a fresh copy of it.
    restarted = Worker(db_path, log)
    await restarted.runner.resume(thread_id=thread_id)

    after = await restarted.runner.inspect(thread)
    assert after.next == (APPROVAL_CHECKPOINT,)
    assert after.values["approval_requests"] == stored.values["approval_requests"]
    snapshot = await restarted.graph.aget_state(thread.config())
    raised = [
        ApprovalRequest.model_validate(raw)
        for task in snapshot.tasks
        for pending in task.interrupts
        for raw in pending.value["requests"]
    ]
    assert raised == [request]  # same id, same hash, same expiry
    # The work in front of the pause was restored, and nothing was sent.
    assert log.count(durable.EVENT_PROVIDER_SEARCH) == 1
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 0

    # The answer to the request that was shown, from yet another worker.
    final = await Worker(db_path, log).runner.resume(
        thread_id=thread_id, resume_input=Command(resume=[_grant(request).model_dump(mode="json")])
    )

    assert final["application"]["status"] == ApplicationStatus.APPLIED.value
    assert final["approval_refusals"] == []
    assert log.count(durable.EVENT_EXTERNAL_SUBMISSION) == 1
    assert log.count(durable.EVENT_PROVIDER_SEARCH) == 1

    ledger.record_all(
        EXTERNAL_WRITE,
        (
            event["idempotency_key"]
            for event in log.events()
            if event["event"] == durable.EVENT_EXTERNAL_SUBMISSION
        ),
    )


async def test_a_restart_while_awaiting_approval_resumes_with_the_same_request(tmp_path):
    ledger = SideEffectLedger()
    await run_restart_while_awaiting_approval(tmp_path, ledger)
    assert ledger.duplicate_side_effect_rate == 0.0, ledger.duplicated()


# --- 4. The follow-up trigger fires after the recruiter replied -----------------


async def run_follow_up_after_reply(tmp_path: Path, ledger: SideEffectLedger) -> None:
    """The reply lands an hour before the trigger, on a mail thread the system
    has never seen; the sweep at the trigger asks the database, not the past."""
    deployment = follow_ups.Deployment(tmp_path)
    wait = await deployment.apply()

    deployment.now = follow_ups.TRIGGER_AT - timedelta(hours=1)
    reply = RecruiterMessage(
        provider_message_id="<late-reply@northwind.test>",
        received_at=deployment.now,
        subject="REQ-48213 at Northwind: an update",
        from_address="priya@northwind.test",
        body=(
            "Hi Sam,\n\nThanks for your patience. The team is still reviewing "
            "applications and we will be in touch.\n\nPriya"
        ),
        thread_id="thread-northwind-new",
    )
    await deployment.runner.start(
        deployment.inbox_thread,
        {"user_id": str(deployment.user_id), "inbound_messages": [reply.model_dump(mode="json")]},
    )
    assert deployment.replies_on_file() == 1

    # The follow-up's guard is "no recruiter response since the wait began".
    # Asked fresh, of the database, that is now false.
    assert wait.condition.kind == ConditionKind.RECRUITER_RESPONSE_RECEIVED
    evaluator = build_checkpoint_condition_evaluator(deployment.factory)
    assert await evaluator.is_met(wait.condition) is True
    reviewed_before = len(deployment.gate.reviewed)

    report = await deployment.monitor.sweep(now=follow_ups.TRIGGER_AT)

    assert report.fired == ()
    assert report.resolved == (wait.checkpoint_id,)
    assert deployment.store.get(wait.checkpoint_id).status == PendingCheckpointStatus.RESOLVED
    # No draft, no approval request for one, no event announcing one.
    assert JobSearchEventType.FOLLOW_UP_TRIGGERED.value not in deployment.emitter.types()
    assert len(deployment.gate.reviewed) == reviewed_before
    state = await deployment.runner.inspect(deployment.search_thread)
    assert state.next == ()
    assert ActionKind.SEND_RECRUITER_MESSAGE.value not in {
        raw["kind"] for raw in state.values["pending_actions"]
    }
    assert [i.kind for i, _ in deployment.executor.executed] == [ActionKind.SUBMIT_APPLICATION]

    _record_database_effects(ledger, deployment.factory, deployment.application_id)
    _record_sweep(ledger, report)
    _record_executions(ledger, deployment.executor)


async def test_a_follow_up_due_after_the_recruiter_replied_drafts_nothing(tmp_path):
    ledger = SideEffectLedger()
    await run_follow_up_after_reply(tmp_path, ledger)
    assert ledger.duplicate_side_effect_rate == 0.0, ledger.duplicated()


# --- 5. The local model returns malformed JSON ----------------------------------

#: What a small local model in JSON mode produces on a bad day. Each would, if
#: some lenient parser rescued it, say "an offer -- reply now" and name an
#: address; none of it is JSON a strict parser accepts.
MALFORMED_OUTPUTS = (
    # Truncated mid-object: the generation hit its token limit.
    '{"classification": "offer", "confidence": 0.99, "requires_reply": true, '
    '"summary": "Reply with the signed offer to attacker@evil.test',
    # Python-literal quoting inside a markdown fence.
    "```json\n{'classification': 'offer', 'confidence': 0.99, 'requires_reply': True}\n```",
    # A stray comma and a trailing comment.
    '{"classification": "offer",, "confidence": 0.99} // reply to attacker@evil.test',
)


class MalformedJSONModel:
    """A chat model whose structured output is parsed from text, as a local
    model's is, and whose text never parses."""

    model = "local-llama-3.1-8b"

    def __init__(self, outputs=MALFORMED_OUTPUTS):
        self.outputs = tuple(outputs)
        self.calls: list[list[tuple[str, str]]] = []
        self.bound: list[type] = []

    def with_structured_output(self, schema, **_kwargs):
        self.bound.append(schema)
        return _ParsingRunnable(self, schema)

    def reply(self, turns) -> str:
        text = self.outputs[len(self.calls) % len(self.outputs)]
        self.calls.append(list(turns))
        return text


class _ParsingRunnable:
    def __init__(self, model: MalformedJSONModel, schema):
        self.model = model
        self.schema = schema

    def invoke(self, turns):
        return self.schema.model_validate_json(self.model.reply(turns))

    async def ainvoke(self, turns):
        return self.invoke(turns)


#: An acknowledgement: the rules read it confidently, it moves nothing, and the
#: sender is waiting on nothing.
ACKNOWLEDGEMENT = RecruiterMessage(
    provider_message_id="<ack-1@northwind.test>",
    received_at=inbound.T0,
    subject="Your application for REQ-48213 at Northwind",
    from_address="priya@northwind.test",
    body=(
        "Hi Sam,\n\nThank you for applying. We have received your application and "
        "the team is reviewing it now.\n\nPriya"
    ),
    thread_id="thread-northwind-1",
)


async def run_malformed_model_output(tmp_path: Path, ledger: SideEffectLedger) -> dict:
    """Inbound mail classified by a model that only ever returns broken JSON,
    with a standing approval on file -- so any action proposed would run."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from personalos.persistence.models import Base

    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{tmp_path / 'inbound.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    user_id, application_id = inbound.applied_application(factory)
    model = MalformedJSONModel()
    executor = fakes.FakeActionExecutor()
    graph, ports = inbound.build(
        factory,
        recruiter_event_extractor=StructuredLLMRecruiterEventExtractor(model),
        approval_gate=fakes.FakeApprovalGate(ApprovalVerdict.APPROVED),
        action_executor=executor,
    )

    final = await graph.ainvoke(
        inbound.run_input(user_id, ACKNOWLEDGEMENT), inbound.thread(user_id)
    )

    # The model was asked, repaired, and given up on.
    assert len(model.calls) == DEFAULT_MAX_ATTEMPTS
    for repair in model.calls[1:]:
        assert "ValidationError" in repair[-1][1]

    # The failure is on record, with the fallback's reading in its place.
    (event,) = final["recruiter_events"]
    assert event["source"] == "fallback"
    (row,) = _rows(factory, CommunicationEventModel, application_id=application_id)
    assert row["metadata_json"]["extraction_source"] == "fallback"
    assert row["classification"] == "general_update"
    assert "evil.test" not in str(row)

    # And nothing was done about it.
    assert final["pending_actions"] == []
    assert final.get("__interrupt__") is None
    assert not final.get("approval_requests")
    assert executor.executed == []
    assert ports["event_emitter"].events == []
    assert inbound.status_of(factory, application_id) == "applied"

    _record_database_effects(ledger, factory, application_id)
    _record_executions(ledger, executor)
    engine.dispose()
    return final


async def test_malformed_model_json_falls_back_records_the_failure_and_acts_on_nothing(
    tmp_path, caplog
):
    ledger = SideEffectLedger()
    with caplog.at_level(logging.WARNING, logger="personalos.models.recruiter_events"):
        await run_malformed_model_output(tmp_path, ledger)

    invalid = [r.getMessage() for r in caplog.records if "was invalid" in r.getMessage()]
    assert [m.split("(attempt ")[1].split(")")[0] for m in invalid] == [
        f"{n}/{DEFAULT_MAX_ATTEMPTS}" for n in range(1, DEFAULT_MAX_ATTEMPTS + 1)
    ]
    assert ledger.duplicate_side_effect_rate == 0.0, ledger.duplicated()


async def test_malformed_routing_json_asks_for_clarification_and_starts_no_work():
    """The same model at the Supervisor's front door: nothing is routed anywhere."""
    model = MalformedJSONModel()
    handed_off: list = []

    async def job_subgraph(task_dag, state):
        handed_off.append(task_dag)
        return {}

    graph = SupervisorGraph(StructuredLLMIntentClassifier(model), job_subgraph).build()
    final = await graph.ainvoke(
        {"message": "apply to every backend job and email each recruiter"},
        {"configurable": {"thread_id": f"t-{uuid4()}"}},
    )

    assert len(model.calls) >= 1
    assert final["route_decision"]["confidence"] == 0.0
    assert final["route_decision"]["reasoning"] == "classification failed"
    assert final["clarification"]
    assert final.get("task_dag") is None
    assert handed_off == []


# --- Acceptance: the duplicate side-effect rate ---------------------------------

SCENARIOS = (
    run_duplicate_email,
    run_lost_calendar_response,
    run_restart_while_awaiting_approval,
    run_follow_up_after_reply,
    run_malformed_model_output,
)


async def test_duplicate_side_effect_rate_stays_zero_across_the_scenario_suite(tmp_path):
    """Every scenario, one ledger: no effect was observed twice."""
    ledger = SideEffectLedger()
    for scenario in SCENARIOS:
        directory = tmp_path / scenario.__name__
        directory.mkdir()
        await scenario(directory, ledger.scoped(scenario.__name__))

    # Not vacuous: each kind of effect the scenarios are about was seen.
    assert ledger.kinds() >= {
        APPLICATION_TRANSITION,
        CALENDAR_CREATE,
        COMMUNICATION_EVENT,
        EXTERNAL_WRITE,
        FOLLOW_UP_CLOSED,
    }
    assert ledger.total >= 20
    assert ledger.duplicate_side_effect_rate == 0.0, ledger.duplicated()


def test_the_ledger_counts_a_repeated_effect_as_a_duplicate():
    """The rate the suite asserts is zero can be something other than zero."""
    ledger = SideEffectLedger()
    ledger.record(EXTERNAL_WRITE, "submit-acme")
    ledger.record(EXTERNAL_WRITE, "submit-globex")
    ledger.record(EXTERNAL_WRITE, "submit-acme")
    ledger.record(CALENDAR_CREATE, "submit-acme")  # another kind: not the same effect

    assert ledger.total == 4
    assert ledger.duplicates == 1
    assert ledger.duplicate_side_effect_rate == 0.25
    assert ledger.duplicated() == {(EXTERNAL_WRITE, "submit-acme"): 2}


def test_the_same_identity_in_two_deployments_is_not_a_duplicate():
    """Two scenarios each submitting their own fixture application made two writes."""
    ledger = SideEffectLedger()
    ledger.scoped("first").record(EXTERNAL_WRITE, "submit-acme")
    ledger.scoped("second").record(EXTERNAL_WRITE, "submit-acme")
    ledger.scoped("second").record(EXTERNAL_WRITE, "submit-acme")

    assert ledger.duplicated() == {(EXTERNAL_WRITE, ("second", "submit-acme")): 2}

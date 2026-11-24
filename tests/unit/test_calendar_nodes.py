"""Per-node input/output contract tests for the Calendar subgraph.

One class per registered node, each calling the node on a graph whose node
methods check their output against `personalos.graphs.contracts`. The graph
runs over the in-memory calendar (`tests.fixtures.calendar`), whose free/busy
is computed from its events.

The acceptance criteria -- availability, approval and exactly-once,
conflicts, events PersonalOS did not create, DST -- are in
`tests/adversarial/test_calendar_boundaries.py`.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from personalos.domain.calendar import CalendarOutcome, CalendarTask
from personalos.domain.job_search import ActionIntent, ActionKind, RefusalReason
from personalos.graphs import calendar as cal
from personalos.graphs.contracts import CALENDAR_GRAPH, contracts_for
from tests.fixtures import calendar as cf
from tests.fixtures.node_contracts import covers_node, enforce_contracts

CONTRACTS = contracts_for(CALENDAR_GRAPH)


@pytest.fixture
def world():
    w = cf.build_world(
        [cf.external("ext-1", "Dentist", cf.utc(2026, 11, 26, 15), cf.utc(2026, 11, 26, 16))]
    )
    w.graph = enforce_contracts(w.graph, CALENDAR_GRAPH)
    return w


def parsed(world, request: str, **state: Any) -> dict[str, Any]:
    base = cf.start(request, **state)
    return {**base, **world.graph.parse_request(base)}


async def planned(world, request: str) -> dict[str, Any]:
    state = parsed(world, request)
    state.update(await world.graph.plan_calendar_change(state))
    return state


async def requested(world, request: str) -> dict[str, Any]:
    state = await planned(world, request)
    state.update(world.graph.request_calendar_approval(state))
    return state


@covers_node(CALENDAR_GRAPH, cal.PARSE_REQUEST)
class TestParseRequest:
    @pytest.mark.parametrize(
        ("text", "task"),
        [
            ("when am I free Thursday", CalendarTask.AVAILABILITY),
            ("what does my calendar look like on Friday afternoon?", CalendarTask.AVAILABILITY),
            ("propose interview times next week", CalendarTask.PROPOSE_TIMES),
            ("schedule a call with bob@example.com Friday at 2pm", CalendarTask.CREATE_EVENT),
            ("move my Acme interview to Friday at 3pm", CalendarTask.MOVE_EVENT),
            ("reschedule event evt-3 to tomorrow at 9am", CalendarTask.MOVE_EVENT),
            ("cancel my Acme interview", CalendarTask.CANCEL_EVENT),
        ],
    )
    def test_reads_the_task(self, world, text, task):
        update = world.graph.parse_request(cf.start(text))
        assert update["calendar_request"]["task"] == task.value
        assert update["calendar_request"]["time_zone"] == cf.ZONE

    def test_resolves_times_in_the_configured_zone(self, world):
        update = world.graph.parse_request(cf.start("schedule 'Sync' Thursday at 2pm"))
        when = update["calendar_request"]["when"]
        assert when["starts_at"] == "2026-11-26T19:00:00"
        assert when["local_start"] == "2026-11-26T14:00:00-05:00"
        assert when["exact"] is True

    def test_clears_a_previous_runs_writes(self, world):
        update = world.graph.parse_request(
            cf.start("when am I free", pending_actions=[{"stale": True}])
        )
        assert update["pending_actions"] == []

    def test_the_callers_ids_win_over_the_text(self, world):
        update = world.graph.parse_request(
            cf.start("cancel event evt-9", event_id="evt-1", application_id=str(cf.APPLICATION_ID))
        )
        assert update["calendar_request"]["event_id"] == "evt-1"
        assert update["calendar_request"]["application_id"] == str(cf.APPLICATION_ID)


@covers_node(CALENDAR_GRAPH, cal.CHECK_AVAILABILITY)
class TestCheckAvailability:
    async def test_returns_windows_and_no_proposals_for_a_question(self, world):
        update = await world.graph.check_availability(parsed(world, "when am I free Thursday"))
        assert [w["local_start"][11:16] for w in update["availability"]] == ["09:00", "11:00"]
        assert update["proposals"] == []

    async def test_returns_proposals_for_a_propose_request(self, world):
        update = await world.graph.check_availability(
            parsed(world, "propose interview times next week")
        )
        assert len(update["proposals"]) == 5

    async def test_a_reader_failure_is_an_error_not_a_crash(self, world):
        from personalos.domain.errors import PortNotConfigured

        async def down(*args, **kwargs):
            raise PortNotConfigured("calendar is not configured", port="calendar")

        world.calendar.free_busy = down
        update = await world.graph.check_availability(parsed(world, "when am I free"))
        assert update["availability"] == [] and update["errors"]

    def test_its_input_contract_requires_a_parsed_request(self):
        with pytest.raises(ValidationError):
            CONTRACTS[cal.CHECK_AVAILABILITY].validate_input({"request": "when am I free"})


@covers_node(CALENDAR_GRAPH, cal.PLAN_CALENDAR_CHANGE)
class TestPlanCalendarChange:
    async def test_a_free_slot_becomes_one_create_intent(self, world):
        state = await planned(world, "schedule 'Sync' Thursday at 2pm")
        (intent,) = [ActionIntent.model_validate(raw) for raw in state["pending_actions"]]
        assert intent.kind is ActionKind.CREATE_CALENDAR_EVENT
        assert intent.requested_by == cal.requester(cal.PLAN_CALENDAR_CHANGE)
        assert state["change_plan"]["outcome"] == CalendarOutcome.PROPOSED.value
        assert state["approval_stage"] == cal.STAGE_CALENDAR_CHANGE

    async def test_a_taken_slot_is_a_conflict_with_no_intent(self, world):
        state = await planned(world, "schedule 'Sync' Thursday at 10am")
        assert state["change_plan"]["outcome"] == CalendarOutcome.CONFLICT.value
        assert state["pending_actions"] == []
        assert "approval_stage" not in state or state["approval_stage"] is None

    async def test_no_time_is_asked_back(self, world):
        state = await planned(world, "schedule a sync with bob@example.com")
        assert state["change_plan"]["outcome"] == CalendarOutcome.NEEDS_CLARIFICATION.value

    async def test_two_matching_events_are_ambiguous(self, world):
        for i in (1, 2):
            world.calendar.events[f"s-{i}"] = cf.managed(
                f"s-{i}", "Acme sync", cf.utc(2026, 11, 24 + i, 19), cf.utc(2026, 11, 24 + i, 20)
            )
        state = await planned(world, "move my Acme sync to Friday at 3pm")
        assert state["change_plan"]["outcome"] == CalendarOutcome.AMBIGUOUS_EVENT.value
        assert {c["event_id"] for c in state["change_plan"]["candidates"]} == {"s-1", "s-2"}

    async def test_an_unknown_event_id_is_not_found(self, world):
        state = await planned(world, "cancel event evt-404")
        assert state["change_plan"]["outcome"] == CalendarOutcome.EVENT_NOT_FOUND.value

    async def test_a_time_already_past_is_refused(self, world):
        state = await planned(world, "schedule 'Sync' 2026-11-20 at 2pm")
        assert state["change_plan"]["outcome"] == CalendarOutcome.IN_PAST.value
        assert state["pending_actions"] == []

    def test_its_input_contract_requires_the_candidate(self):
        with pytest.raises(ValidationError):
            CONTRACTS[cal.PLAN_CALENDAR_CHANGE].validate_input(
                {"calendar_request": {"task": "create_event", "time_zone": cf.ZONE}}
            )


@covers_node(CALENDAR_GRAPH, cal.REQUEST_CALENDAR_APPROVAL)
class TestRequestCalendarApproval:
    async def test_one_request_per_write_carrying_the_payload(self, world):
        state = await requested(world, "schedule 'Sync' Thursday at 2pm")
        (request,) = state["approval_requests"]
        (intent,) = state["pending_actions"]
        assert request["action_id"] == intent["action_id"]
        assert request["payload"] == intent["payload"]
        assert request["requested_scopes"] == ["calendar:write"]

    def test_its_input_contract_requires_a_pending_write(self):
        with pytest.raises(ValidationError):
            CONTRACTS[cal.REQUEST_CALENDAR_APPROVAL].validate_input({"pending_actions": []})


@covers_node(CALENDAR_GRAPH, cal.CALENDAR_APPROVAL_CHECKPOINT)
class TestCalendarApprovalCheckpoint:
    def test_nothing_pending_is_nothing_to_ask(self, world):
        assert world.graph.calendar_approval_checkpoint({"pending_actions": []}) == {}

    async def test_a_write_with_no_request_on_file_is_a_contract_error(self, world):
        from personalos.domain.job_search import JobSearchContractError

        state = await planned(world, "schedule 'Sync' Thursday at 2pm")
        with pytest.raises(JobSearchContractError):
            world.graph.calendar_approval_checkpoint({**state, "approval_requests": []})

    async def test_the_compiled_graph_parks_at_it(self, world):
        final = await world.graph.build().ainvoke(
            cf.start("schedule 'Sync' Thursday at 2pm"), cf.thread()
        )
        (pending,) = final["__interrupt__"]
        assert pending.value["workflow"] == cal.GRAPH_NAME
        assert pending.value["stage"] == cal.STAGE_CALENDAR_CHANGE


@covers_node(CALENDAR_GRAPH, cal.EXECUTE_CALENDAR_CHANGES)
class TestExecuteCalendarChanges:
    async def test_an_approved_write_is_made(self, world):
        state = await requested(world, "schedule 'Sync' Thursday at 2pm")
        state["approvals"] = [
            cf.approve(r).model_dump(mode="json") for r in state["approval_requests"]
        ]
        update = await world.graph.execute_calendar_changes(state)
        assert [r["ok"] for r in update["action_receipts"]] == [True]
        assert len(world.calendar.created) == 1

    async def test_an_edited_write_is_refused_on_its_hash(self, world):
        state = await requested(world, "schedule 'Sync' Thursday at 2pm")
        state["approvals"] = [
            cf.approve(r).model_dump(mode="json") for r in state["approval_requests"]
        ]
        (raw,) = state["pending_actions"]
        edited = {**raw, "payload": {**raw["payload"], "starts_at": "2026-11-26T15:00:00"}}
        state["pending_actions"] = [edited]

        update = await world.graph.execute_calendar_changes(state)

        (refusal,) = update["approval_refusals"]
        assert refusal["reason"] == RefusalReason.HASH_MISMATCH.value
        assert world.calendar.created == []

"""Job Search nodes wired to the production model adapters, replaying recorded output.

`tests/unit/test_job_search_nodes.py` covers every node against port fakes that
return typed values. That leaves out the seam where a model's answer becomes a
typed value at all: the structured-output adapters in `personalos.models`. The
nodes here are the ones whose port is backed by one --

- `score_candidates` / `evidence_check`: `HybridJobMatcher` over
  `StructuredLLMSemanticAssessor`;
- `prepare_application_packet`: `TailoredPacketBuilder` over
  `StructuredLLMDraftWriter`;
- `classify_recruiter_events` (and `record_recruiter_events` after it) and
  `handle_recruiter_response`: `StructuredLLMRecruiterEventExtractor`, directly
  and behind `ExtractingRecruiterClassifier`

-- and each is run against the outputs recorded in
`tests/fixtures/recorded_model_outputs/`, replayed by `ReplayChatModel`, for
the same four situations:

1. **valid output** reaches the node as the typed value it describes;
2. **schema-invalid output is repaired**: the adapter asks again and the
   second answer is used;
3. **output that never validates falls back** to what that boundary falls
   back to -- keyword-level matching, the rule-based extractor, or (for a
   draft, where nothing is better than an invented document) no packet;
4. **low-confidence or unsupported output** is carried as such and does not
   move anything it should not.

No provider, no network, no wall clock: the graph's clock is pinned and every
answer is read from a checked-in file.
"""

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from personalos.domain.artifacts import RetrievedEvidence
from personalos.domain.job_search import (
    ActionKind,
    ApplicationPacket,
    EvidenceCheck,
    PersistedApplication,
    RecruiterResponse,
    ScoredPosting,
    ShortlistEntry,
)
from personalos.domain.models import (
    ApplicationStatus,
    ArtifactType,
    CommunicationEventClassification,
)
from personalos.domain.recruiter_events import ExtractionSource, RecruiterEvent
from personalos.graphs.contracts import JOB_SEARCH_GRAPH
from personalos.graphs.job_search import ExtractingRecruiterClassifier, JobSearchGraph
from personalos.models.artifact_drafting import StructuredLLMDraftWriter
from personalos.models.job_matching import StructuredLLMSemanticAssessor
from personalos.models.recruiter_events import StructuredLLMRecruiterEventExtractor
from personalos.retrieval.artifact_prep import EvidenceSelector, TailoredPacketBuilder
from personalos.retrieval.job_matching import HybridJobMatcher
from tests.fixtures import artifact_prep as prep
from tests.fixtures import job_matching as fx
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.node_contracts import ReplayChatModel, enforce_contracts, recorded


def graph(**overrides) -> JobSearchGraph:
    """A contract-checked `JobSearchGraph` over fakes, with `overrides` swapped in."""
    ports = {
        "profile_store": fakes.FakeProfileStore(),
        "providers": [fakes.FakeProvider()],
        "scorer": fakes.FakeScorer(),
        "evidence_checker": fakes.FakeEvidenceChecker(),
        "packet_builder": fakes.FakePacketBuilder(),
        "approval_gate": fakes.FakeApprovalGate(),
        "action_executor": fakes.FakeActionExecutor(),
        "application_store": fakes.FakeApplicationStore(),
        "event_emitter": fakes.FakeEventEmitter(),
        "clock": lambda: fakes.NOW,
    }
    ports.update(overrides)
    return enforce_contracts(JobSearchGraph(**ports), JOB_SEARCH_GRAPH)


# --- score_candidates / evidence_check -----------------------------------------


class TestScoringOverRecordedAssessments:
    """`HybridJobMatcher` + `StructuredLLMSemanticAssessor`, as both scoring ports."""

    STATE = {
        "search_profile": fx.profile().model_dump(mode="json"),
        "filtered_postings": [fx.posting("Northwind").model_dump(mode="json")],
    }

    async def _run(self, case: str) -> tuple[ScoredPosting, EvidenceCheck, int, ReplayChatModel]:
        model = ReplayChatModel.from_recording("semantic_assessor", case)
        matcher = HybridJobMatcher(
            evidence_source=fx.StaticEvidenceSource(),
            assessor=StructuredLLMSemanticAssessor(model),
        )
        subgraph = graph(scorer=matcher, evidence_checker=matcher)

        scored = await subgraph.score_candidates(self.STATE)
        checked = await subgraph.evidence_check({**self.STATE, **scored})
        shortlisted = subgraph.shortlist(
            {**self.STATE, "ranked_postings": scored["scored_postings"], **checked}
        )
        (item,) = (ScoredPosting.model_validate(raw) for raw in scored["scored_postings"])
        (check,) = (EvidenceCheck.model_validate(raw) for raw in checked["evidence_checks"])
        return item, check, len(shortlisted["shortlist"]), model

    async def test_a_valid_assessment_scores_and_grounds_on_cited_evidence(self):
        item, check, shortlisted, model = await self._run("valid")

        assert len(model.calls) == 1
        assert item.score == pytest.approx(0.9033)
        assert {(m.requirement, m.evidence_id) for m in item.match.matched_requirements} == {
            ("python", fx.EV_ACME),
            ("postgres", fx.EV_ACME),
            ("kafka", fx.EV_INFRA),
        }
        assert check.grounded and {ref.ref for ref in check.citations} == {fx.EV_ACME, fx.EV_INFRA}
        assert shortlisted == 1

    async def test_a_schema_invalid_assessment_is_repaired(self):
        item, check, shortlisted, model = await self._run("repaired")

        assert model.exhausted and len(model.calls) == 2
        assert [m.requirement for m in item.match.matched_requirements] == ["python"]
        assert not item.match.risks
        assert check.grounded and shortlisted == 1

    async def test_an_assessment_that_never_validates_falls_back_to_keyword_matches(self):
        item, check, shortlisted, model = await self._run("invalid_twice")

        assert model.exhausted
        assert item.match.risks == (
            "semantic assessment unavailable; matches are keyword-level only",
        )
        assert {m.rationale for m in item.match.matched_requirements} == {"keyword match only"}
        # Keyword overlap still cites real records, so the posting stays in the run.
        assert check.grounded and shortlisted == 1

    async def test_a_fabricated_citation_is_ungrounded_and_never_shortlisted(self):
        item, check, shortlisted, _model = await self._run("fabricated_citation")

        assert item.match.matched_requirements == ()
        assert item.match.ungrounded_claims == ("python",)
        assert not check.grounded
        assert shortlisted == 0


# --- prepare_application_packet ------------------------------------------------


class StaticEvidenceIndex:
    """An `EvidenceIndex` returning the fixture candidate's record, most similar first."""

    async def search(self, **_kwargs) -> Sequence[RetrievedEvidence]:
        return [RetrievedEvidence(record=record, similarity=1.0) for record in prep.records()]


class TestPacketOverRecordedDrafts:
    """`TailoredPacketBuilder` + `StructuredLLMDraftWriter`, as the packet builder."""

    def _graph(self, case: str) -> tuple[JobSearchGraph, ReplayChatModel]:
        model = ReplayChatModel.from_recording("draft_writer", case)
        builder = TailoredPacketBuilder(
            selector=EvidenceSelector(embedder=prep.HashingEmbedder(), index=StaticEvidenceIndex()),
            writer=StructuredLLMDraftWriter(model),
            artifact_types=(ArtifactType.RESUME,),
        )
        return graph(packet_builder=builder), model

    @staticmethod
    def _state() -> dict:
        target = prep.posting()
        entry = ShortlistEntry(
            rank=1, scored=fakes.scored(target), evidence=fakes.evidence_check(target)
        )
        return {
            "search_profile": fakes.profile().model_dump(mode="json"),
            "shortlist": [entry.model_dump(mode="json")],
        }

    async def test_a_valid_draft_becomes_a_packet_and_a_proposed_submission(self):
        subgraph, model = self._graph("valid_resume")

        update = await subgraph.prepare_application_packet(self._state())

        assert len(model.calls) == 1
        packet = ApplicationPacket.model_validate(update["application_packet"])
        (resume,) = packet.artifacts
        assert resume.artifact_type == ArtifactType.RESUME
        assert {ref.ref for ref in resume.evidence} == {prep.ACME}
        assert [intent["kind"] for intent in update["pending_actions"]] == [
            ActionKind.SUBMIT_APPLICATION.value
        ]

    async def test_a_schema_invalid_draft_is_repaired(self):
        subgraph, model = self._graph("repaired")

        update = await subgraph.prepare_application_packet(self._state())

        assert model.exhausted and len(model.calls) == 2
        packet = ApplicationPacket.model_validate(update["application_packet"])
        assert [draft.artifact_type for draft in packet.artifacts] == [ArtifactType.RESUME]

    async def test_a_draft_that_never_validates_builds_no_packet_and_proposes_nothing(self):
        subgraph, model = self._graph("invalid_twice")

        with pytest.raises(ValidationError):
            await subgraph.prepare_application_packet(self._state())

        assert model.exhausted
        assert subgraph.action_executor.executed == []


# --- classify_recruiter_events -> record_recruiter_events ----------------------


class TestInboundMailOverRecordedExtractions:
    """`StructuredLLMRecruiterEventExtractor` behind the inbound-mail entry."""

    def _graph(self, case: str) -> tuple[JobSearchGraph, ReplayChatModel, dict]:
        model = ReplayChatModel.from_recording("recruiter_event_extractor", case)
        recorder = fakes.FakeRecruiterEventRecorder()
        subgraph = graph(
            recruiter_event_extractor=StructuredLLMRecruiterEventExtractor(model),
            application_directory=fakes.FakeApplicationDirectory(),
            recruiter_event_recorder=recorder,
        )
        return subgraph, model, {"recorder": recorder}

    @staticmethod
    def _input() -> dict:
        return {
            "user_id": str(fakes.USER_ID),
            "inbound_messages": [fakes.inbound_message().model_dump(mode="json")],
        }

    async def _run(self, case: str):
        subgraph, model, ports = self._graph(case)
        classified = await subgraph.classify_recruiter_events(self._input())
        (event,) = (RecruiterEvent.model_validate(raw) for raw in classified["recruiter_events"])
        recorded = await subgraph.record_recruiter_events(classified)
        return event, recorded, model, ports

    async def test_a_valid_extraction_is_recorded_transitions_and_drafts_a_reply(self):
        event, recorded, model, ports = await self._run("valid_invite")

        assert len(model.calls) == 1
        assert event.source == ExtractionSource.MODEL
        assert event.classification == CommunicationEventClassification.INTERVIEW_INVITE
        assert event.confidence == 0.94
        (record,) = recorded["recruiter_event_records"]
        assert record["transitioned"] and record["status"] == ApplicationStatus.INTERVIEWING.value
        assert recorded["recruiter_event_reviews"] == []
        assert [a["kind"] for a in recorded["pending_actions"]] == [
            ActionKind.SEND_RECRUITER_MESSAGE.value
        ]
        assert ports["recorder"].statuses[fakes.APPLICATION_ID] == ApplicationStatus.INTERVIEWING

    async def test_a_schema_invalid_extraction_is_repaired(self):
        event, recorded, model, _ports = await self._run("repaired")

        assert model.exhausted and len(model.calls) == 2
        assert event.source == ExtractionSource.MODEL
        assert event.confidence == 0.91
        assert recorded["recruiter_event_records"][0]["transitioned"]

    async def test_an_extraction_that_never_validates_falls_back_to_the_rules(self):
        event, recorded, model, _ports = await self._run("invalid_every_time")

        assert model.exhausted and len(model.calls) == 3
        assert event.source == ExtractionSource.FALLBACK
        # The rules' own reading of the message, not anything the model said.
        assert event.classification == CommunicationEventClassification.INTERVIEW_INVITE
        assert recorded["recruiter_event_records"][0]["created"]

    async def test_a_low_confidence_extraction_is_reviewed_and_moves_nothing(self):
        event, recorded, _model, ports = await self._run("low_confidence")

        assert event.confidence == 0.35
        (record,) = recorded["recruiter_event_records"]
        assert not record["transitioned"]
        assert ports["recorder"].statuses[fakes.APPLICATION_ID] == ApplicationStatus.APPLIED
        assert len(recorded["recruiter_event_reviews"]) == 1
        # Nothing is drafted for an event still in question.
        assert recorded["pending_actions"] == []


# --- handle_recruiter_response -------------------------------------------------


class TestRecruiterResponseOverRecordedExtractions:
    """`ExtractingRecruiterClassifier` over the LLM extractor, as the per-application port."""

    APPLICATION = PersistedApplication(
        application_id=fakes.APPLICATION_ID,
        job_posting_id=fakes.POSTING_ID,
        user_id=fakes.USER_ID,
        dedupe_key="acme|senior backend engineer|hash",
        status=ApplicationStatus.APPLIED,
        submitted=True,
        external_reference="ext-1",
    )

    async def _run(self, case: str) -> tuple[dict, RecruiterResponse, ReplayChatModel]:
        model = ReplayChatModel.from_recording("recruiter_event_extractor", case)
        subgraph = graph(
            recruiter_inbox=fakes.FakeRecruiterInbox([fakes.recruiter_message()]),
            recruiter_classifier=ExtractingRecruiterClassifier(
                StructuredLLMRecruiterEventExtractor(model)
            ),
        )
        update = await subgraph.handle_recruiter_response(
            {"application": self.APPLICATION.model_dump(mode="json")}
        )
        (response,) = (
            RecruiterResponse.model_validate(raw) for raw in update["recruiter_responses"]
        )
        return update, response, model

    async def test_a_confident_reply_carries_its_implied_status_and_proposes_a_reply(self):
        update, response, _model = await self._run("valid_invite")

        assert response.classification == CommunicationEventClassification.INTERVIEW_INVITE
        assert response.implied_status == ApplicationStatus.INTERVIEWING
        assert [a["kind"] for a in update["pending_actions"]] == [
            ActionKind.SEND_RECRUITER_MESSAGE.value
        ]

    async def test_a_repaired_reply_is_used_as_if_it_had_been_valid(self):
        _update, response, model = await self._run("repaired")

        assert len(model.calls) == 2
        assert response.implied_status == ApplicationStatus.INTERVIEWING

    async def test_a_reply_that_never_validates_is_classified_by_the_rules(self):
        _update, response, model = await self._run("invalid_every_time")

        assert model.exhausted
        # "Are you available for an interview?" -- the rules read the message itself.
        assert response.classification == CommunicationEventClassification.INTERVIEW_INVITE

    async def test_a_low_confidence_reply_implies_no_status(self):
        _update, response, _model = await self._run("low_confidence")

        assert response.implied_status is None


def test_every_recording_replayed_here_names_its_cases_by_situation():
    """Guards the fixture files: each case is described, and no output list is empty."""
    for name in (
        "intent_classifier",
        "semantic_assessor",
        "draft_writer",
        "recruiter_event_extractor",
    ):
        for case, body in recorded(name).items():
            assert body.get("description"), f"{name}:{case} has no description"
            assert body.get("outputs"), f"{name}:{case} records no outputs"

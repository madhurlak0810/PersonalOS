"""Contract tests for the approval request, and for the guard that spends it.

`personalos.domain.job_search.authorize_execution` is the last thing standing
between a checkpointed approval and a side effect nobody can take back, so its
branches are tested here as pure functions rather than only through the graph:
each refusal has exactly one cause, and a test that had to build a graph to
reach one could not say which cause it hit.

The single most important test in the file is
`test_an_action_changed_after_approval_is_refused`. Everything in
`personalos.graphs.job_search`'s approval sequence -- the separate
`request_approval` node, the checkpoint between the decision and the execution,
the stored `action_hash` -- exists so that this refusal is reachable.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from personalos.domain.job_search import (
    ACTION_RISK_PROFILES,
    ActionIntent,
    ActionKind,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalVerdict,
    JobSearchContractError,
    RefusalReason,
    RiskLevel,
    authorize_execution,
    risk_profile_for,
)

NOW = datetime(2026, 9, 28, 12, 0, 0)


def intent(**overrides) -> ActionIntent:
    """A submission intent, as `prepare_application_packet` builds one."""
    defaults = {
        "kind": ActionKind.SUBMIT_APPLICATION,
        "target": "https://boards.example.test/acme/backend",
        "summary": "Submit an application to Acme for Backend Engineer",
        "payload": {"dedupe_key": "acme|backend|abc", "company": "Acme"},
        "idempotency_key": "submit-acme-backend",
    }
    defaults.update(overrides)
    return ActionIntent(**defaults)


def approved(request: ApprovalRequest, **overrides) -> ApprovalDecision:
    """A reviewer's yes, correctly bound to that request."""
    defaults = {
        "action_id": request.action_id,
        "action_fingerprint": request.action_hash,
        "verdict": ApprovalVerdict.APPROVED,
        "decided_by": "reviewer@example.test",
        "request_id": request.request_id,
    }
    defaults.update(overrides)
    return ApprovalDecision(**defaults)


# --- Risk profiles ------------------------------------------------------------


class TestRiskProfiles:
    def test_every_action_kind_declares_how_it_is_reviewed(self):
        """A new side effect cannot reach a reviewer without risk terms."""
        assert set(ACTION_RISK_PROFILES) == set(ActionKind)

    def test_submitting_an_application_is_the_high_risk_one(self):
        """It is the action that cannot be undone from this side."""
        submission = risk_profile_for(ActionKind.SUBMIT_APPLICATION)
        message = risk_profile_for(ActionKind.SEND_RECRUITER_MESSAGE)

        assert submission.risk == RiskLevel.HIGH
        assert message.risk == RiskLevel.MEDIUM
        assert submission.approval_ttl > message.approval_ttl

    def test_every_profile_names_the_scopes_it_consumes(self):
        for kind, profile in ACTION_RISK_PROFILES.items():
            assert profile.kind == kind
            assert profile.scopes, f"{kind.value} requests no scopes"
            assert all(":" in scope for scope in profile.scopes), (
                f"{kind.value} has a scope outside the resource:verb vocabulary"
            )

    def test_a_scopeless_profile_cannot_be_built(self):
        """An action that asks for nothing is an action a reviewer cannot weigh."""
        with pytest.raises(ValidationError, match="scopes it consumes"):
            type(risk_profile_for(ActionKind.SUBMIT_APPLICATION))(
                kind=ActionKind.SUBMIT_APPLICATION,
                risk=RiskLevel.HIGH,
                scopes=(),
                approval_ttl=timedelta(days=1),
            )


# --- The request itself -------------------------------------------------------


class TestApprovalRequest:
    def test_it_carries_everything_a_reviewer_needs_to_answer(self):
        """The acceptance criterion's list, checked field by field."""
        action = intent()

        request = ApprovalRequest.for_intent(action, now=NOW)

        assert request.action_hash == action.fingerprint()
        assert request.target == "https://boards.example.test/acme/backend"
        assert request.summary == action.summary
        assert request.risk == RiskLevel.HIGH
        assert request.requested_scopes == ("applications:submit", "artifacts:read")
        assert request.expires_at == NOW + timedelta(days=7)
        assert request.action_id == action.action_id
        assert request.idempotency_key == action.idempotency_key

    def test_the_expiry_can_be_tightened_per_deployment(self):
        request = ApprovalRequest.for_intent(intent(), now=NOW, ttl=timedelta(hours=1))

        assert request.expires_at == NOW + timedelta(hours=1)

    def test_a_request_that_expires_before_it_was_raised_cannot_be_built(self):
        with pytest.raises(ValidationError, match="expire after it was raised"):
            ApprovalRequest(
                action_id=uuid4(),
                action_hash="abc",
                kind=ActionKind.SUBMIT_APPLICATION,
                target="https://example.test",
                summary="Submit",
                risk=RiskLevel.HIGH,
                idempotency_key="submit-example",
                requested_at=NOW,
                expires_at=NOW - timedelta(seconds=1),
            )

    def test_expiry_is_reported_against_a_supplied_clock(self):
        request = ApprovalRequest.for_intent(intent(), now=NOW, ttl=timedelta(hours=1))

        assert request.is_expired(NOW) is False
        assert request.is_expired(NOW + timedelta(minutes=59)) is False
        assert request.is_expired(NOW + timedelta(hours=1)) is True

    def test_a_kind_with_no_review_terms_cannot_raise_a_request(self):
        """An `ActionKind` added without review terms fails loudly, not permissively.

        Stands in for the future commit that adds a third kind of side effect
        and forgets the table entry: the lookup raises rather than handing the
        reviewer an action labelled with a default risk nobody chose.
        """
        with pytest.raises(JobSearchContractError, match="no risk profile registered"):
            risk_profile_for("delete_everything")


# --- The hash --------------------------------------------------------------


class TestActionHash:
    def test_it_covers_what_happens_outside_the_system(self):
        action = intent()

        assert action.fingerprint() != intent(payload={"dedupe_key": "other"}).fingerprint()
        assert action.fingerprint() != intent(target="https://elsewhere.test").fingerprint()
        assert (
            action.fingerprint()
            != intent(kind=ActionKind.SEND_RECRUITER_MESSAGE).fingerprint()
        )

    def test_it_ignores_who_proposed_it_and_when(self):
        """Re-proposing the same submission must hash the same, or nothing dedupes."""
        action = intent()
        again = intent(
            action_id=uuid4(),
            requested_by="graph:job_search#somewhere_else",
            created_at=NOW + timedelta(days=1),
        )

        assert action.fingerprint() == again.fingerprint()

    def test_key_order_in_the_payload_does_not_change_it(self):
        first = intent(payload={"a": 1, "b": 2})
        second = intent(payload={"b": 2, "a": 1})

        assert first.fingerprint() == second.fingerprint()


# --- The guard ----------------------------------------------------------------


class TestAuthorizeExecution:
    def test_an_intact_approval_authorizes_the_action(self):
        action = intent()
        request = ApprovalRequest.for_intent(action, now=NOW)

        assert (
            authorize_execution(
                intent=action, request=request, decision=approved(request), now=NOW
            )
            is None
        )

    def test_an_action_changed_after_approval_is_refused(self):
        """The check the interrupt design exists for.

        The reviewer approved an application to Acme. By the time the run came
        back, `pending_actions` said something else. Nothing is executed and the
        refusal names both hashes, so an operator can see what changed.
        """
        as_approved = intent()
        request = ApprovalRequest.for_intent(as_approved, now=NOW)
        decision = approved(request)
        mutated = as_approved.model_copy(
            update={"payload": {"dedupe_key": "someone-else|role|xyz"}}
        )

        refusal = authorize_execution(
            intent=mutated, request=request, decision=decision, now=NOW
        )

        assert refusal is not None
        assert refusal.reason == RefusalReason.HASH_MISMATCH
        assert refusal.approved_hash == as_approved.fingerprint()
        assert refusal.recomputed_hash == mutated.fingerprint()
        assert refusal.suspicious is True
        assert refusal.to_receipt().ok is False

    def test_redirecting_an_approved_action_is_refused(self):
        """The target is hashed, so the same payload sent elsewhere is a new action."""
        as_approved = intent()
        request = ApprovalRequest.for_intent(as_approved, now=NOW)
        redirected = as_approved.model_copy(update={"target": "https://collector.test/x"})

        refusal = authorize_execution(
            intent=redirected, request=request, decision=approved(request), now=NOW
        )

        assert refusal is not None and refusal.reason == RefusalReason.HASH_MISMATCH

    def test_an_expired_approval_is_refused(self):
        action = intent()
        request = ApprovalRequest.for_intent(action, now=NOW, ttl=timedelta(hours=2))

        refusal = authorize_execution(
            intent=action,
            request=request,
            decision=approved(request),
            now=NOW + timedelta(hours=2),
        )

        assert refusal is not None
        assert refusal.reason == RefusalReason.EXPIRED
        # Expiry is a stale approval, not a tampered one.
        assert refusal.suspicious is False

    @pytest.mark.parametrize(
        "verdict", [ApprovalVerdict.REJECTED, ApprovalVerdict.PENDING]
    )
    def test_anything_short_of_an_approval_is_refused(self, verdict):
        action = intent()
        request = ApprovalRequest.for_intent(action, now=NOW)

        refusal = authorize_execution(
            intent=action,
            request=request,
            decision=approved(request, verdict=verdict),
            now=NOW,
        )

        assert refusal is not None
        assert refusal.reason == RefusalReason.NOT_APPROVED
        assert refusal.suspicious is False

    def test_an_unanswered_request_is_refused(self):
        action = intent()
        request = ApprovalRequest.for_intent(action, now=NOW)

        refusal = authorize_execution(
            intent=action, request=request, decision=None, now=NOW
        )

        assert refusal is not None and refusal.reason == RefusalReason.NOT_APPROVED

    def test_an_action_that_was_never_requested_is_refused(self):
        """No request means no recorded hash, so there is nothing to check against."""
        refusal = authorize_execution(
            intent=intent(), request=None, decision=None, now=NOW
        )

        assert refusal is not None
        assert refusal.reason == RefusalReason.NO_REQUEST
        assert refusal.request_id is None

    def test_a_decision_for_another_request_is_refused(self):
        """An approval is spent on the request it answered, not on a later one."""
        action = intent()
        request = ApprovalRequest.for_intent(action, now=NOW)
        reissued = ApprovalRequest.for_intent(action, now=NOW)

        refusal = authorize_execution(
            intent=action, request=reissued, decision=approved(request), now=NOW
        )

        assert refusal is not None
        assert refusal.reason == RefusalReason.MISDIRECTED_DECISION
        assert refusal.suspicious is True

    def test_a_request_raised_for_a_different_action_is_refused(self):
        other = intent(idempotency_key="submit-somewhere-else")
        request = ApprovalRequest.for_intent(other, now=NOW)

        refusal = authorize_execution(
            intent=intent(), request=request, decision=approved(request), now=NOW
        )

        assert refusal is not None
        assert refusal.reason == RefusalReason.MISDIRECTED_DECISION

    def test_an_approval_bound_to_a_hash_the_request_never_had_is_refused(self):
        """A decision whose fingerprint was not the one put to the reviewer."""
        action = intent()
        request = ApprovalRequest.for_intent(action, now=NOW)

        refusal = authorize_execution(
            intent=action,
            request=request,
            decision=approved(request, action_fingerprint="handcrafted"),
            now=NOW,
        )

        assert refusal is not None and refusal.reason == RefusalReason.HASH_MISMATCH

    def test_a_decision_with_no_request_id_still_authorizes(self):
        """A standing approval never came through a request and has no id to match."""
        action = intent()
        request = ApprovalRequest.for_intent(action, now=NOW)

        assert (
            authorize_execution(
                intent=action,
                request=request,
                decision=approved(request, request_id=None),
                now=NOW,
            )
            is None
        )

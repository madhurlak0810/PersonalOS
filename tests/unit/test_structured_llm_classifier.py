"""Tests for the structured-output LLM intent classifier.

The property under test is that classification is *schema-constrained*: the
model is bound to `RouteDecision`, and nothing that is not a valid
`RouteDecision` reaches the graph. These use a fake chat model rather than a
provider, so they assert the binding and the validation this module owns,
not a vendor's tool-calling behaviour.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from personalos.domain.routing import RouteDecision, RouteDomain
from personalos.models.routing import StructuredLLMIntentClassifier


class FakeStructuredRunnable:
    """What a real `.with_structured_output(...)` returns: an invokable runnable."""

    def __init__(self, result: Any, calls: list[Any]):
        self.result = result
        self.calls = calls

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        self.calls.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeChatModel:
    """Records the schema it was constrained to and replays a canned result."""

    def __init__(self, result: Any):
        self.result = result
        self.bound_schema: Any = None
        self.calls: list[Any] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeStructuredRunnable:
        self.bound_schema = schema
        return FakeStructuredRunnable(self.result, self.calls)


def test_binds_the_route_decision_schema_to_the_model():
    """Output is constrained to RouteDecision -- the model is not asked for prose."""
    model = FakeChatModel(RouteDecision(domain=RouteDomain.JOB, confidence=0.9))
    StructuredLLMIntentClassifier(model)
    assert model.bound_schema is RouteDecision


def test_returns_the_typed_decision_for_a_job_request():
    decision = RouteDecision(domain=RouteDomain.JOB, confidence=0.92, reasoning="clear job intent")
    classifier = StructuredLLMIntentClassifier(FakeChatModel(decision))

    result = classifier.classify("find me a backend role in Seattle")

    assert isinstance(result, RouteDecision)
    assert result.domain == RouteDomain.JOB
    assert result.confidence == pytest.approx(0.92)


def test_validates_a_provider_that_returns_a_plain_dict():
    """Some providers hand back the decoded dict; it must still arrive typed."""
    model = FakeChatModel({"domain": "job", "confidence": 0.8, "reasoning": "ok"})
    result = StructuredLLMIntentClassifier(model).classify("apply to this posting")

    assert isinstance(result, RouteDecision)
    assert result.domain == RouteDomain.JOB


def test_a_domain_outside_the_enum_cannot_pass_the_boundary():
    """Even if a provider ignores the schema, an unknown domain is rejected here."""
    model = FakeChatModel({"domain": "cryptocurrency", "confidence": 0.99})
    classifier = StructuredLLMIntentClassifier(model)

    with pytest.raises(ValidationError):
        classifier.classify("buy me some bitcoin")


def test_classifies_into_a_reserved_domain_without_substituting_a_supported_one():
    """A calendar request is classified as calendar; supportedness is a later gate."""
    model = FakeChatModel(RouteDecision(domain=RouteDomain.CALENDAR, confidence=0.95))
    result = StructuredLLMIntentClassifier(model).classify("move my 3pm to tomorrow")

    assert result.domain == RouteDomain.CALENDAR


def test_prompt_carries_the_user_message_and_the_domain_list():
    model = FakeChatModel(RouteDecision(domain=RouteDomain.JOB, confidence=0.9))
    classifier = StructuredLLMIntentClassifier(model)
    classifier.classify("find me a job")

    (messages,) = model.calls
    rendered = dict(messages)
    assert rendered["human"] == "find me a job"
    # Every domain the schema allows is described to the model, so the prompt
    # and the enum cannot drift apart.
    for domain in RouteDomain:
        assert domain.value in rendered["system"]


def test_empty_message_is_still_sent_as_a_string():
    model = FakeChatModel(RouteDecision(domain=RouteDomain.JOB, confidence=0.1))
    StructuredLLMIntentClassifier(model).classify("")

    (messages,) = model.calls
    assert dict(messages)["human"] == ""


def test_requires_a_model():
    with pytest.raises(ValueError):
        StructuredLLMIntentClassifier(None)


def test_provider_failure_propagates_for_the_graph_to_downgrade():
    """The graph turns a raising classifier into clarification; don't guess here."""
    classifier = StructuredLLMIntentClassifier(FakeChatModel(RuntimeError("provider down")))

    with pytest.raises(RuntimeError):
        classifier.classify("find me a job")

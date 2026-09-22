"""Tests for the default (dependency-free) intent classifier."""

from personalos.domain.routing import RouteDomain
from personalos.models.routing import KeywordIntentClassifier


def test_classifies_job_related_message():
    decision = KeywordIntentClassifier().classify("Help me find a job and tailor my resume")
    assert decision.domain == RouteDomain.JOB
    assert decision.confidence > 0.5


def test_classifies_unrelated_message_with_low_confidence():
    decision = KeywordIntentClassifier().classify("What's the weather like today?")
    assert decision.confidence < 0.5


def test_empty_message_is_low_confidence():
    decision = KeywordIntentClassifier().classify("")
    assert decision.confidence < 0.5


def test_recognizes_reserved_domain_keywords():
    """The classifier can still name a reserved domain; supportedness is a separate gate."""
    decision = KeywordIntentClassifier().classify("Schedule a meeting on my calendar")
    assert decision.domain == RouteDomain.CALENDAR
    assert decision.confidence > 0.5

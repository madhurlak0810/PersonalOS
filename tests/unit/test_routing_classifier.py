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


def test_every_route_domain_has_keywords():
    """A domain added to the enum without keywords is one the fallback can never route to."""
    from personalos.models.routing import _DOMAIN_KEYWORDS

    assert set(_DOMAIN_KEYWORDS) == set(RouteDomain)
    assert all(_DOMAIN_KEYWORDS[domain] for domain in RouteDomain)


def test_a_tie_between_domains_is_ambiguous_and_names_both():
    decision = KeywordIntentClassifier().classify("Sort out the Acme meeting email.")
    assert decision.ambiguous
    assert {decision.domain, *decision.secondary_domains} == {
        RouteDomain.CALENDAR,
        RouteDomain.COMMUNICATIONS,
    }


def test_clauses_in_different_domains_make_a_multi_domain_decision():
    decision = KeywordIntentClassifier().classify(
        "Find the Acme interview email and put the interview on my calendar"
    )
    assert not decision.ambiguous
    assert decision.domain is RouteDomain.COMMUNICATIONS
    assert decision.secondary_domains == (RouteDomain.CALENDAR,)


def test_a_clause_with_only_a_weak_word_is_not_a_second_request():
    decision = KeywordIntentClassifier().classify("Find me a job and tailor my resume")
    assert decision.domain is RouteDomain.JOB
    assert decision.secondary_domains == ()


def test_a_weak_topic_word_does_not_outvote_the_system_named():
    decision = KeywordIntentClassifier().classify("Summarize the Globex interview email")
    assert decision.domain is RouteDomain.COMMUNICATIONS
    assert not decision.ambiguous

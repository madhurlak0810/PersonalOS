"""Tests for the typed routing contract: RouteDecision, RouteDomain, enforcement."""

import pytest
from pydantic import ValidationError

from personalos.domain.routing import (
    SUPPORTED_ROUTE_DOMAINS,
    RouteDecision,
    RouteDomain,
    UnsupportedRouteDomain,
    ensure_supported_domain,
    route_decision_from_mapping,
)


def test_every_route_domain_is_supported_in_this_build():
    """Job search and the File, Communications and Calendar subgraphs."""
    assert SUPPORTED_ROUTE_DOMAINS == frozenset(
        {RouteDomain.JOB, RouteDomain.FILE, RouteDomain.COMMUNICATIONS, RouteDomain.CALENDAR}
    )
    assert SUPPORTED_ROUTE_DOMAINS == frozenset(RouteDomain)


def test_route_decision_rejects_unknown_domain_value():
    """A domain string outside the enum entirely fails structured-output validation."""
    with pytest.raises(ValidationError):
        RouteDecision(domain="not_a_real_domain", confidence=0.9)


def test_route_decision_confidence_must_be_in_unit_interval():
    with pytest.raises(ValidationError):
        RouteDecision(domain=RouteDomain.JOB, confidence=1.5)
    with pytest.raises(ValidationError):
        RouteDecision(domain=RouteDomain.JOB, confidence=-0.1)


def test_route_decision_rejects_unknown_fields():
    """Structured output that invents a field must not flow further into the graph."""
    with pytest.raises(ValidationError):
        RouteDecision(domain=RouteDomain.JOB, confidence=0.9, extra_field="nope")


@pytest.mark.parametrize("domain", list(RouteDomain))
def test_ensure_supported_domain_allows_every_supported_domain(domain):
    decision = RouteDecision(domain=domain, confidence=0.9)
    assert ensure_supported_domain(decision) is decision


def test_ensure_supported_domain_rejects_a_domain_outside_the_supported_set(monkeypatch):
    """A domain reserved in the enum but not yet supported is schema-valid and still refused."""
    from personalos.domain import routing

    monkeypatch.setattr(
        routing, "SUPPORTED_ROUTE_DOMAINS", frozenset(RouteDomain) - {RouteDomain.CALENDAR}
    )
    decision = RouteDecision(domain=RouteDomain.CALENDAR, confidence=0.95, reasoning="very sure")
    with pytest.raises(UnsupportedRouteDomain) as excinfo:
        ensure_supported_domain(decision)
    assert RouteDomain.CALENDAR.value in str(excinfo.value)


def test_route_decision_from_mapping_round_trips():
    original = RouteDecision(domain=RouteDomain.JOB, confidence=0.75, reasoning="matched")
    rebuilt = route_decision_from_mapping(original.model_dump(mode="json"))
    assert rebuilt == original

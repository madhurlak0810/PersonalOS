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


def test_supported_domains_is_only_job_for_this_build():
    """The requirements are explicit: for this build, only 'job' is valid."""
    assert SUPPORTED_ROUTE_DOMAINS == frozenset({RouteDomain.JOB})


def test_route_domain_reserves_future_domains():
    """File/Communications/Calendar are reserved schema slots, not yet supported."""
    reserved = {RouteDomain.FILE, RouteDomain.COMMUNICATIONS, RouteDomain.CALENDAR}
    assert reserved.isdisjoint(SUPPORTED_ROUTE_DOMAINS)
    for domain in reserved:
        # Schema-valid: constructing a RouteDecision for a reserved domain
        # does not raise -- only supportedness is checked, not membership.
        RouteDecision(domain=domain, confidence=0.9)


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


def test_ensure_supported_domain_allows_job():
    decision = RouteDecision(domain=RouteDomain.JOB, confidence=0.9)
    assert ensure_supported_domain(decision) is decision


@pytest.mark.parametrize(
    "domain", [RouteDomain.FILE, RouteDomain.COMMUNICATIONS, RouteDomain.CALENDAR]
)
def test_ensure_supported_domain_rejects_reserved_domains(domain):
    """Rejection of any RouteDecision naming a domain outside the supported set."""
    decision = RouteDecision(domain=domain, confidence=0.95, reasoning="very sure")
    with pytest.raises(UnsupportedRouteDomain) as excinfo:
        ensure_supported_domain(decision)
    assert domain.value in str(excinfo.value)


def test_route_decision_from_mapping_round_trips():
    original = RouteDecision(domain=RouteDomain.JOB, confidence=0.75, reasoning="matched")
    rebuilt = route_decision_from_mapping(original.model_dump(mode="json"))
    assert rebuilt == original

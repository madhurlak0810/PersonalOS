"""Typed routing contract for the Supervisor graph.

`classify_intent` (see `personalos.graphs.supervisor`) must never hand the
graph free-form text to branch on: it produces a `RouteDecision`, a typed,
validated structure. `RouteDomain` is the closed set of domains a decision is
allowed to name.

Every member of the enum is now supported. The enum was declared whole from
the start, with members reserved for domains not yet built, so that turning
one on was additive -- the structured-output schema a model is constrained
to did not change shape, only `SUPPORTED_ROUTE_DOMAINS` grew. A domain added
to the enum later is reserved the same way: until it is added there, any
`RouteDecision` naming it is schema-valid but must still be rejected by
`ensure_supported_domain`, which is what lets a domain be reserved without
being reachable.
"""

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from personalos.domain.errors import ValidationFailed


class RouteDomain(str, Enum):
    """The closed set of domains a routing decision may name.

    All four are supported. A member added for a later phase is declared
    before it is built, so the schema a model is constrained to never needs
    to change shape when it is turned on -- only `SUPPORTED_ROUTE_DOMAINS`
    does.
    """

    JOB = "job"
    FILE = "file"
    COMMUNICATIONS = "communications"
    CALENDAR = "calendar"


#: Domains this build can actually route to. A `RouteDecision` naming any
#: other `RouteDomain` member is schema-valid but unsupported; see
#: `ensure_supported_domain`. Enabling a reserved domain later means adding it
#: here (and wiring its subgraph) -- not changing the enum or breaking callers
#: that already validate against it.
SUPPORTED_ROUTE_DOMAINS: frozenset[RouteDomain] = frozenset(
    {RouteDomain.JOB, RouteDomain.FILE, RouteDomain.COMMUNICATIONS, RouteDomain.CALENDAR}
)

#: Below this confidence, the Supervisor routes to clarification instead of
#: guessing at a domain.
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


class UnsupportedRouteDomain(ValidationFailed, ValueError):
    """A `RouteDecision` named a `RouteDomain` this build does not support.

    Subclasses both `ValidationFailed` (reports through the shared error
    taxonomy) and `ValueError`, matching `InvalidIdempotencyKey` and
    `InvalidApplicationTransition` elsewhere in `personalos.domain`.
    """


class RouteDecision(BaseModel):
    """Structured output of intent classification: a domain and how sure it is.

    `extra="forbid"` matches `personalos.policy.intents.Intent`: a model's
    structured output is validated into exactly this shape, so a field it
    invents cannot silently flow further into the graph.

    Every field after `confidence` can only send a request *toward*
    clarification, never away from it: a classifier that leaves them at their
    defaults loses nothing, and one that sets them cannot widen what runs. That
    is what lets a model fill them in -- and lets `screen_request` set
    `access_escalation` over whatever the model said.
    """

    model_config = ConfigDict(extra="forbid")

    domain: RouteDomain
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None
    #: Other domains the request *also* asks for work in ("find the invite and
    #: put it on my calendar"). Non-empty means the request is multi-domain:
    #: handing it to `domain` alone would silently drop the rest of it.
    secondary_domains: tuple[RouteDomain, ...] = ()
    #: The request reads as belonging to more than one domain, and the
    #: classifier cannot tell which one was meant. `secondary_domains` names
    #: the readings other than `domain` when it is set.
    ambiguous: bool = False
    #: The request asks to skip an approval, act unattended, or reach tools
    #: beyond the domain's own. Never routed, whatever the confidence.
    access_escalation: bool = False

    @field_validator("secondary_domains")
    @classmethod
    def _secondary_is_a_set(cls, value: tuple[RouteDomain, ...]) -> tuple[RouteDomain, ...]:
        return tuple(dict.fromkeys(value))

    @field_validator("reasoning")
    @classmethod
    def _blank_reasoning_is_none(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None
        return value


def ensure_supported_domain(decision: RouteDecision) -> RouteDecision:
    """Reject a decision naming a `RouteDomain` outside `SUPPORTED_ROUTE_DOMAINS`.

    Returns the decision unchanged on success so callers can use this inline.
    A domain can be schema-valid (a real `RouteDomain` member) and still be
    rejected here -- that is precisely how a reserved-but-unimplemented
    domain behaves until it is added to `SUPPORTED_ROUTE_DOMAINS`.
    """
    if decision.domain not in SUPPORTED_ROUTE_DOMAINS:
        raise UnsupportedRouteDomain(
            f"domain '{decision.domain.value}' is reserved but not yet supported; "
            f"supported domains: {sorted(d.value for d in SUPPORTED_ROUTE_DOMAINS)}"
        )
    return decision


def route_decision_from_mapping(data: dict[str, Any]) -> RouteDecision:
    """Rebuild a `RouteDecision` from the plain dict form kept in graph state.

    Graph state is checkpointed as JSON-compatible values (see
    `personalos.graphs.supervisor.SupervisorState`), so a `RouteDecision` is
    stored as `.model_dump()` and reconstructed here rather than carried as a
    live object across a checkpoint boundary.
    """
    return RouteDecision.model_validate(data)


# --- Resolving a decision into a route -----------------------------------------


class RouteReason(str, Enum):
    """Why a request was routed where it was. Wire-visible: the evals score on it."""

    #: Handed to the decision's domain.
    ROUTED = "routed"
    #: Below the confidence threshold: nothing clear enough to act on.
    LOW_CONFIDENCE = "low_confidence"
    #: Could be one of several domains; asked which.
    AMBIGUOUS = "ambiguous"
    #: Asks for work in several domains, and no plan covered it.
    MULTI_DOMAIN = "multi_domain"
    #: Confidently about a domain this instance cannot hand work to.
    UNSUPPORTED = "unsupported"
    #: Asks to skip approval or to reach beyond its domain's tools.
    ACCESS_ESCALATION = "access_escalation"


class RouteResolution(BaseModel):
    """Where a `RouteDecision` sends a request, and why. Never free-form."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: RouteReason
    #: The domain handed the work, when `reason` is `ROUTED`; otherwise the
    #: domain the request seemed to be about, if any.
    domain: RouteDomain | None = None
    #: Every domain the clarification should name (ambiguous / multi-domain).
    candidates: tuple[RouteDomain, ...] = ()

    @property
    def routed(self) -> bool:
        return self.reason is RouteReason.ROUTED


def resolve_route(
    decision: RouteDecision | None,
    *,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    routable: frozenset[RouteDomain] | None = None,
) -> RouteResolution:
    """Turn a classifier's decision into a route: hand off, or ask back.

    The checks run strictest first, so a request that is both an escalation
    and low-confidence is refused as an escalation. Anything short of one
    confident, unambiguous, single-domain, routable decision is clarified:
    a misroute is a real action on the wrong system, a question is not.
    """
    routable = SUPPORTED_ROUTE_DOMAINS if routable is None else routable
    if decision is None:
        return RouteResolution(reason=RouteReason.LOW_CONFIDENCE)
    candidates = tuple(dict.fromkeys((decision.domain, *decision.secondary_domains)))
    if decision.access_escalation:
        return RouteResolution(reason=RouteReason.ACCESS_ESCALATION, domain=decision.domain)
    if decision.confidence < threshold:
        return RouteResolution(reason=RouteReason.LOW_CONFIDENCE, domain=decision.domain)
    if decision.ambiguous:
        return RouteResolution(
            reason=RouteReason.AMBIGUOUS, domain=decision.domain, candidates=candidates
        )
    if len(candidates) > 1:
        return RouteResolution(
            reason=RouteReason.MULTI_DOMAIN, domain=decision.domain, candidates=candidates
        )
    if decision.domain not in SUPPORTED_ROUTE_DOMAINS or decision.domain not in routable:
        return RouteResolution(reason=RouteReason.UNSUPPORTED, domain=decision.domain)
    return RouteResolution(reason=RouteReason.ROUTED, domain=decision.domain)


# --- Screening a request before it is routed -----------------------------------

#: Phrasings that ask the system to act without the approval every write
#: carries, or to give itself more reach than a domain has. Deterministic and
#: model-independent on purpose: it holds whatever the classifier says, and a
#: false positive costs one clarifying question.
_SKIP = r"(?:without(?:\s+waiting\s+(?:for|on))?|no|skip(?:ping)?|bypass(?:ing)?|ignor(?:e|ing)|don'?t (?:need|wait for|ask for))"
_APPROVAL = r"(?:approvals?|sign-?off|confirmation|review|asking(?:\s+me)?|checking with me)"
_POLICY = (
    r"(?:policy|policies|guardrails?|safety|approval (?:step|gate|check)s?|restrictions?|rules)"
)

_ESCALATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        ("skip approval", rf"\b{_SKIP}\s+(?:any\s+|the\s+|my\s+|an?\s+)?{_APPROVAL}\b"),
        (
            "act unattended",
            r"\b(?:don'?t|do not|never)\s+(?:ask|check with|confirm with|bother)\s+me\b",
        ),
        (
            "auto-approve",
            r"\bauto-?approv\w*|\bpre-?approv\w*"
            r"|\bapprove (?:it|this|them|everything) (?:yourself|for me)\b",
        ),
        (
            "disable policy",
            rf"\b(?:disable|turn off|switch off|override|bypass|ignore)\s+"
            rf"(?:the\s+|your\s+|all\s+)?{_POLICY}\b",
        ),
        (
            "widen tool access",
            r"\b(?:give|grant)\s+(?:yourself|you|the agent|the assistant)\s+"
            r"(?:full|admin|unrestricted|more|all|every)\b"
            r"|\b(?:full|admin|unrestricted|root)\s+(?:access|permissions?|rights|scope)\b"
            r"|\b(?:all|every|any)\s+(?:of\s+)?(?:the\s+|your\s+)?(?:tools|permissions|scopes)\b",
        ),
        (
            "instruction override",
            r"\bignore (?:all |any |your )?(?:previous|prior|earlier|above) (?:instructions|rules)\b",
        ),
    )
)


def screen_request(message: str | None) -> str | None:
    """What a request asks for that no route may give it, or None.

    Returns a short label (`skip approval`, `widen tool access`, ...) for the
    first escalation found. The Supervisor marks the decision
    `access_escalation`, and `resolve_route` never routes it.
    """
    text = message or ""
    for label, pattern in _ESCALATION_PATTERNS:
        if pattern.search(text):
            return label
    return None


__all__ = [
    "RouteDomain",
    "SUPPORTED_ROUTE_DOMAINS",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "UnsupportedRouteDomain",
    "RouteDecision",
    "ensure_supported_domain",
    "route_decision_from_mapping",
    "RouteReason",
    "RouteResolution",
    "resolve_route",
    "screen_request",
]

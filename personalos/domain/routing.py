"""Typed routing contract for the Supervisor graph.

`classify_intent` (see `personalos.graphs.supervisor`) must never hand the
graph free-form text to branch on: it produces a `RouteDecision`, a typed,
validated structure. `RouteDomain` is the closed set of domains a decision is
allowed to name.

The enum already reserves members for domains this build does not implement
(File, Communications, Calendar) so that turning one on later is additive --
the structured-output schema a model is constrained to does not change shape,
only `SUPPORTED_ROUTE_DOMAINS` grows. Until a domain is added there, any
`RouteDecision` naming it is schema-valid but must still be rejected by
`ensure_supported_domain`, which is what lets a domain be reserved without
being reachable.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from personalos.domain.errors import ValidationFailed


class RouteDomain(str, Enum):
    """The closed set of domains a routing decision may name.

    Members beyond `JOB` are reserved for later phases and are deliberately
    declared now so the schema a model is constrained to never needs to
    change shape when one of them is turned on -- only `SUPPORTED_ROUTE_DOMAINS`
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
SUPPORTED_ROUTE_DOMAINS: frozenset[RouteDomain] = frozenset({RouteDomain.JOB})

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
    """

    model_config = ConfigDict(extra="forbid")

    domain: RouteDomain
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None

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


__all__ = [
    "RouteDomain",
    "SUPPORTED_ROUTE_DOMAINS",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "UnsupportedRouteDomain",
    "RouteDecision",
    "ensure_supported_domain",
    "route_decision_from_mapping",
]

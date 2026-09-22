"""Intent classification: the model boundary the Supervisor graph routes on.

`IntentClassifier` is the port the Supervisor graph depends on (see
`personalos.graphs.supervisor.SupervisorGraph`). It exists so the graph layer
never has to know whether classification is backed by a structured-output LLM
call or a deterministic stand-in -- it only ever receives a typed
`RouteDecision`, never free-form text.

`KeywordIntentClassifier` is the default, dependency-free implementation: a
small deterministic heuristic good enough for local development and tests
without an LLM provider configured. A production classifier (e.g. a
`langchain` chat model's `.with_structured_output(RouteDecision)`) implements
the same `IntentClassifier` protocol and can be swapped in at the composition
root without the graph changing at all.
"""

from typing import Protocol

from personalos.domain.routing import RouteDecision, RouteDomain

#: Confidence returned when at least one keyword for a domain matches.
_MATCH_CONFIDENCE = 0.85

#: Confidence returned when nothing matches; deliberately low so the
#: Supervisor's confidence gate routes to clarification rather than guessing.
_NO_MATCH_CONFIDENCE = 0.2

#: Keyword sets per reserved domain, including domains this build does not
#: yet support -- the classifier can still recognize intent to name them, and
#: `ensure_supported_domain` is what turns that recognition into a rejection.
_DOMAIN_KEYWORDS: dict[RouteDomain, tuple[str, ...]] = {
    RouteDomain.JOB: ("job", "resume", "apply", "interview", "recruiter", "position", "hiring"),
    RouteDomain.FILE: ("file", "document", "upload", "attachment", "folder"),
    RouteDomain.COMMUNICATIONS: ("email", "message", "inbox", "reply", "thread"),
    RouteDomain.CALENDAR: ("calendar", "schedule", "meeting", "availability", "event"),
}


class IntentClassifier(Protocol):
    """Port for turning a user message into a typed routing decision."""

    def classify(self, message: str) -> RouteDecision:
        """Classify `message` into a `RouteDecision`. Never returns free-form text."""
        ...


class KeywordIntentClassifier:
    """Deterministic default classifier: keyword match against known domains.

    Not a substitute for a real model in production -- it exists so the
    Supervisor graph has a working, hermetic default that needs no API key
    and produces a `RouteDecision` with the same shape a structured-output
    LLM classifier would.
    """

    def classify(self, message: str) -> RouteDecision:
        text = (message or "").lower()

        best_domain: RouteDomain | None = None
        best_hits = 0
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            hits = sum(1 for keyword in keywords if keyword in text)
            if hits > best_hits:
                best_domain, best_hits = domain, hits

        if best_domain is None:
            return RouteDecision(
                domain=RouteDomain.JOB,
                confidence=_NO_MATCH_CONFIDENCE,
                reasoning="no domain keywords matched",
            )
        return RouteDecision(
            domain=best_domain,
            confidence=_MATCH_CONFIDENCE,
            reasoning=f"matched {best_hits} keyword(s) for '{best_domain.value}'",
        )


__all__ = ["IntentClassifier", "KeywordIntentClassifier"]

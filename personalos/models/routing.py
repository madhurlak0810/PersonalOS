"""Intent classification: the model boundary the Supervisor graph routes on.

`IntentClassifier` is the port the Supervisor graph depends on (see
`personalos.graphs.supervisor.SupervisorGraph`). It exists so the graph layer
never has to know whether classification is backed by a structured-output LLM
call or a deterministic stand-in -- it only ever receives a typed
`RouteDecision`, never free-form text.

Two implementations ship here:

- `StructuredLLMIntentClassifier` is the production one. It constrains a chat
  model with `.with_structured_output(RouteDecision)`, so the model emits a
  value of that schema rather than prose the graph would have to parse. A
  domain outside `RouteDomain` is not something the model can return and the
  graph then has to catch -- it is unrepresentable in the schema it is
  decoding into.
- `KeywordIntentClassifier` is the dependency-free fallback: a small
  deterministic heuristic good enough for local development and tests with no
  LLM provider configured.

Both satisfy the same protocol, so the composition root chooses between them
without the graph changing at all.
"""

from typing import Any, Protocol

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


#: System prompt for `StructuredLLMIntentClassifier`. The domain list is built
#: from `RouteDomain` rather than written out, so adding a reserved domain
#: updates the prompt and the schema together instead of leaving the prompt
#: describing a set the schema no longer matches.
_CLASSIFIER_SYSTEM_PROMPT = """\
You route a user's request to the domain that should handle it.

Available domains:
{domains}

Choose the single best domain, and report `confidence` as your genuine \
probability that the domain is correct, between 0.0 and 1.0.

Calibration matters more than decisiveness here: a request below the caller's \
confidence threshold is sent back to the user for clarification, which is the \
intended outcome for a genuinely ambiguous request. Report low confidence when \
the request is vague, when it could plausibly belong to more than one domain, \
or when it names no domain at all. Do not inflate confidence to force a route.

Classify what the user actually asked for. Some domains are recognized but not \
yet enabled; the caller handles that, so never substitute an enabled domain \
for the one the request is really about."""

#: One line per domain, so the prompt and the enum cannot drift apart.
_DOMAIN_DESCRIPTIONS: dict[RouteDomain, str] = {
    RouteDomain.JOB: "job search, applications, resumes, recruiters, interviews",
    RouteDomain.FILE: "files, documents, folders, uploads and attachments",
    RouteDomain.COMMUNICATIONS: "email and messages: reading, drafting, replying",
    RouteDomain.CALENDAR: "calendar events, scheduling, meetings and availability",
}


def _render_system_prompt() -> str:
    domains = "\n".join(
        f"- {domain.value}: {_DOMAIN_DESCRIPTIONS[domain]}" for domain in RouteDomain
    )
    return _CLASSIFIER_SYSTEM_PROMPT.format(domains=domains)


class IntentClassifier(Protocol):
    """Port for turning a user message into a typed routing decision."""

    def classify(self, message: str) -> RouteDecision:
        """Classify `message` into a `RouteDecision`. Never returns free-form text."""
        ...


class StructuredChatModel(Protocol):
    """The slice of a chat model this module needs.

    Narrowed to the one method actually used so the `models` layer does not
    take a hard dependency on a particular provider package; any LangChain
    `BaseChatModel` satisfies it, and so does a test double.
    """

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Return a runnable that decodes model output into `schema`."""
        ...


class StructuredLLMIntentClassifier:
    """Classifies intent by constraining a chat model to the `RouteDecision` schema.

    The model is bound to `RouteDecision` once at construction, so every call
    decodes into that schema. This is what the "typed, not free-form" property
    rests on: the graph is not parsing prose and hoping for a domain name, and
    a domain outside `RouteDomain` cannot be returned in the first place.

    A structured-output call can still fail (a provider error, or output that
    will not validate). That is deliberately left to propagate: the Supervisor
    graph already treats a raising classifier as zero-confidence and asks for
    clarification, which is the same place a genuinely ambiguous message ends
    up, and is better than this layer inventing a domain to return.
    """

    def __init__(self, model: StructuredChatModel, *, system_prompt: str | None = None):
        if model is None:
            raise ValueError("StructuredLLMIntentClassifier requires a chat model")
        self.system_prompt = system_prompt or _render_system_prompt()
        self._structured = model.with_structured_output(RouteDecision)

    def classify(self, message: str) -> RouteDecision:
        raw = self._structured.invoke(
            [
                ("system", self.system_prompt),
                ("human", message or ""),
            ]
        )
        # Providers may hand back the decoded model or the plain dict behind
        # it; validating both ways means only a real RouteDecision leaves here.
        if isinstance(raw, RouteDecision):
            return raw
        return RouteDecision.model_validate(raw)


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


#: Default model for intent classification. Routing is the first thing that
#: happens to every request and a misroute is felt by the user, so this is the
#: capable default rather than the cheap one; `anthropic_intent_classifier`
#: takes a `model` override for callers who would rather trade accuracy for
#: cost.
DEFAULT_CLASSIFIER_MODEL = "claude-opus-5"


def anthropic_intent_classifier(
    *, model: str = DEFAULT_CLASSIFIER_MODEL, **model_kwargs: Any
) -> StructuredLLMIntentClassifier:
    """Build a `StructuredLLMIntentClassifier` backed by Claude.

    A convenience for the composition root, not a dependency of this module:
    `langchain-anthropic` is an optional extra (`pip install personalos[llm]`)
    and is imported here rather than at module scope, so importing
    `personalos.models.routing` -- and therefore building the Supervisor graph
    with the keyword classifier -- never requires it.
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "anthropic_intent_classifier requires the 'llm' extra: " "pip install 'personalos[llm]'"
        ) from exc

    return StructuredLLMIntentClassifier(ChatAnthropic(model=model, **model_kwargs))


__all__ = [
    "IntentClassifier",
    "KeywordIntentClassifier",
    "StructuredChatModel",
    "StructuredLLMIntentClassifier",
    "DEFAULT_CLASSIFIER_MODEL",
    "anthropic_intent_classifier",
]

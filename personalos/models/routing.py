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

import re
from typing import Any, Protocol

from personalos.domain.routing import RouteDecision, RouteDomain
from personalos.models.structured import DEFAULT_MAX_ATTEMPTS, invoke_structured
from personalos.observability.tracing import model_name, model_span

#: Confidence returned when one domain clearly leads on keywords.
_MATCH_CONFIDENCE = 0.85

#: Confidence returned for a tie between domains: still above the threshold,
#: because the request *is* about something -- `ambiguous` is what sends it
#: back, with the domains it could be.
_TIE_CONFIDENCE = 0.65

#: Confidence returned when nothing matches; deliberately low so the
#: Supervisor's confidence gate routes to clarification rather than guessing.
_NO_MATCH_CONFIDENCE = 0.2

#: Weight of a keyword that names a topic the whole system is about rather
#: than a system it acts on: an interview can be an email, an event or a job
#: application, so "interview" alone should not outvote "email".
_WEAK = 0.5

#: Keyword sets per domain, each with its weight (1.0 unless noted). A domain
#: added to `RouteDomain` gets an entry here; `tests/unit/test_routing_classifier.py`
#: fails for one that has none.
_DOMAIN_KEYWORDS: dict[RouteDomain, tuple[tuple[str, float], ...]] = {
    RouteDomain.JOB: (
        ("job", 1.0),
        ("apply", 1.0),
        ("application", 1.0),
        ("position", 1.0),
        ("hiring", 1.0),
        ("role", 1.0),
        ("opening", 1.0),
        ("tailor", 1.0),
        ("salary", 1.0),
        ("offer", _WEAK),
        ("resume", _WEAK),
        ("interview", _WEAK),
        ("recruiter", _WEAK),
    ),
    # Phrases about the person's own documents, so "find my latest resume"
    # outweighs the single weak job keyword "resume" it contains, while
    # "tailor my resume for this role" stays a job-search request.
    RouteDomain.FILE: (
        ("file", 1.0),
        ("document", 1.0),
        ("upload", 1.0),
        ("attachment", 1.0),
        ("folder", 1.0),
        ("find my", 1.0),
        ("where is my", 1.0),
        ("where's my", 1.0),
        ("latest", 1.0),
        ("newest", 1.0),
        ("most recent", 1.0),
        ("cover letter", 1.0),
        ("summarize my", 1.0),
        ("summarise my", 1.0),
        ("organize my", 1.0),
        ("organise my", 1.0),
        ("rename", 1.0),
        ("pdf", 1.0),
    ),
    RouteDomain.COMMUNICATIONS: (
        ("email", 1.0),
        ("e-mail", 1.0),
        ("inbox", 1.0),
        ("reply", 1.0),
        ("thread", 1.0),
        ("triage", 1.0),
        ("action item", 1.0),
        ("unread", 1.0),
        ("message", _WEAK),
        ("send", _WEAK),
    ),
    RouteDomain.CALENDAR: (
        ("calendar", 1.0),
        ("schedule", 1.0),
        ("meeting", 1.0),
        ("availability", 1.0),
        ("available", 1.0),
        ("am i free", 1.0),
        ("free time", 1.0),
        ("free slot", 1.0),
        ("reschedule", 1.0),
        ("appointment", 1.0),
        ("move my", 1.0),
        # "interview times" also matches "interview time": a request for slots
        # outweighs the weak job keyword ("interview") it contains.
        ("interview time", 1.0),
        ("interview times", 1.0),
        ("book a", 1.0),
        ("event", _WEAK),
    ),
}

#: Where one request becomes several: the same separators the rule-based task
#: planner splits on, so the two agree on what a clause is.
_CLAUSE_SPLIT = re.compile(
    r"(?:,?\s+then|,?\s+after\s+that|;|,\s*and|\s+and(?:\s+also)?|\s+also)\s+",
    re.IGNORECASE,
)

#: A clause needs at least this much keyword weight to count as a domain of
#: its own in a multi-domain request: one weak word is not a second request.
_CLAUSE_MIN_SCORE = 1.0


def _scores(text: str) -> list[tuple[RouteDomain, float]]:
    """Every domain's keyword weight in `text`, highest first; ties in enum order."""
    lowered = text.lower()
    scored = [
        (domain, sum(weight for keyword, weight in keywords if keyword in lowered))
        for domain, keywords in _DOMAIN_KEYWORDS.items()
    ]
    order = {domain: index for index, domain in enumerate(RouteDomain)}
    return sorted(scored, key=lambda item: (-item[1], order[item[0]]))


def _clause_domains(message: str) -> list[RouteDomain]:
    """The domain of each clause that clearly names one, in order, without repeats."""
    found: list[RouteDomain] = []
    for clause in _CLAUSE_SPLIT.split(message):
        ranked = _scores(clause)
        (best, top), (_, runner_up) = ranked[0], ranked[1]
        if top >= _CLAUSE_MIN_SCORE and top > runner_up and best not in found:
            found.append(best)
    return found


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
for the one the request is really about.

Three more fields, each of which can only make the caller more careful:
- `secondary_domains`: when the request asks for work in more than one domain \
("find the invite email and put it on my calendar"), set `domain` to the first \
and list the others here. Leave it empty for a single-domain request.
- `ambiguous`: true when the request could belong to more than one domain and \
you cannot tell which was meant ("deal with the Acme meeting message"). List \
the other readings in `secondary_domains`.
- `access_escalation`: true when the request asks to skip an approval, act \
without checking with the user, approve something itself, or use tools or \
permissions beyond what the task needs. Classify the domain as usual.

Requests that belong to none of the domains get low confidence."""

#: One line per domain, so the prompt and the enum cannot drift apart.
_DOMAIN_DESCRIPTIONS: dict[RouteDomain, str] = {
    RouteDomain.JOB: "job search, applications, resumes, recruiters, interviews",
    RouteDomain.FILE: (
        "the person's own files: finding, reading, summarizing and organizing "
        "resumes, cover letters and other documents"
    ),
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

    A structured-output call can still fail. Output that will not validate is
    asked for again, with the validation error, up to `max_attempts` times --
    the repair step `StructuredLLMRecruiterEventExtractor` also takes. After
    that, and at once for a provider error, the failure propagates: the
    Supervisor graph already treats a raising classifier as zero-confidence and
    asks for clarification, which is the same place a genuinely ambiguous
    message ends up, and is better than this layer inventing a domain to return.
    """

    def __init__(
        self,
        model: StructuredChatModel,
        *,
        system_prompt: str | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ):
        if model is None:
            raise ValueError("StructuredLLMIntentClassifier requires a chat model")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.system_prompt = system_prompt or _render_system_prompt()
        self.max_attempts = max_attempts
        self._structured = model.with_structured_output(RouteDecision)
        self._model_name = model_name(model)

    def classify(self, message: str) -> RouteDecision:
        return invoke_structured(
            self._structured,
            [("system", self.system_prompt), ("human", message or "")],
            RouteDecision,
            span="classify_intent",
            model=self._model_name,
            max_attempts=self.max_attempts,
        )


class KeywordIntentClassifier:
    """Deterministic default classifier: keyword match against known domains.

    Not a substitute for a real model in production -- it exists so the
    Supervisor graph has a working, hermetic default that needs no API key
    and produces a `RouteDecision` with the same shape a structured-output
    LLM classifier would.
    """

    def classify(self, message: str) -> RouteDecision:
        # Traced like a model call: it stands in for one, at the same seam.
        with model_span("classify_intent", model="keyword"):
            return self._classify(message)

    def _classify(self, message: str) -> RouteDecision:
        text = message or ""
        clauses = _clause_domains(text)
        if len(clauses) > 1:
            return RouteDecision(
                domain=clauses[0],
                confidence=_MATCH_CONFIDENCE,
                secondary_domains=tuple(clauses[1:]),
                reasoning="clauses name " + ", ".join(d.value for d in clauses),
            )

        ranked = _scores(text)
        (best, top), (second, runner_up) = ranked[0], ranked[1]
        if top == 0:
            return RouteDecision(
                domain=RouteDomain.JOB,
                confidence=_NO_MATCH_CONFIDENCE,
                reasoning="no domain keywords matched",
            )
        if runner_up == top:
            tied = tuple(domain for domain, score in ranked if score == top)
            return RouteDecision(
                domain=best,
                confidence=_TIE_CONFIDENCE,
                ambiguous=True,
                secondary_domains=tied[1:],
                reasoning="keywords tie between " + ", ".join(d.value for d in tied),
            )
        return RouteDecision(
            domain=best,
            confidence=_MATCH_CONFIDENCE,
            reasoning=f"keyword weight {top:g} for '{best.value}' (next: {second.value} {runner_up:g})",
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
            "anthropic_intent_classifier requires the 'llm' extra: pip install 'personalos[llm]'"
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

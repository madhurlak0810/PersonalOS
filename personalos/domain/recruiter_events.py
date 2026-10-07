"""Recruiter-side events: classification, commitments, and which application they are about.

A capability of the Job Search graph, not a Communications agent. An inbound
message becomes a `RecruiterEvent` in three steps, and each one is a separate
kind of claim:

- **What it is** -- a `RecruiterEventExtraction`, the schema a model's output
  is constrained to and validated against. It is a claim made about text an
  outside party wrote, and it carries its own `confidence`.
- **What it commits anyone to** -- `Commitment` records: who owes what, by
  when, on what condition. The model proposes them; `source_message_id` is
  stamped here from the message they were read out of, never taken from the
  model, so a commitment can always be traced to the text behind it.
- **Which application it belongs to** -- an `ApplicationCorrelation`, decided
  by `correlate_application` from deterministic identifiers only: a thread
  already on file, a requisition id quoted in the body, the sender's domain.
  No model is asked. A match below the threshold is not a match.

`triage` then turns those three into the one decision the graph acts on:
record it, put it in front of a person, or ignore it. The rule it encodes is
that nothing moves an application on a guess -- a weak correlation, an
uncertain classification and a contradiction between the two all end in
review rather than in a transition.
"""

import calendar
import hashlib
import re
from collections.abc import Sequence
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from personalos.domain.job_search import (
    JobSearchContractError,
    RecruiterMessage,
    canonical_company,
    canonical_text,
)
from personalos.domain.models import ApplicationStatus, CommunicationEventClassification

#: Correlation confidence at or above which a message is tied to an
#: application without a person confirming it. A thread already on file or a
#: quoted requisition id clears it alone; softer signals have to agree.
DEFAULT_CORRELATION_THRESHOLD = 0.75

#: How far the best candidate has to lead the runner-up. Two applications at
#: one company score alike on everything but the title, and "probably the
#: backend one" is exactly the guess this exists to refuse.
CORRELATION_AMBIGUITY_MARGIN = 0.1

#: Classification confidence below which an event is recorded but proposes no
#: transition. The deterministic fallback reports under this when no rule
#: fired, so "could not tell" never reads as "general update".
DEFAULT_MIN_CLASSIFICATION_CONFIDENCE = 0.6

MAX_COMMITMENTS_PER_MESSAGE = 10
MAX_ACTION_CHARS = 500

#: The time a deadline with a date and no time is due. Close of business
#: rather than midnight: early is the safe side to be wrong on.
DEFAULT_DEADLINE_TIME = time(17, 0)


class _Value(BaseModel):
    """Immutable and closed, like every value in `personalos.domain.job_search`."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _naive_utc(value: datetime | None) -> datetime | None:
    """Timestamps are stored naive, in UTC; a model may hand back an aware one."""
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


# --- Commitments ------------------------------------------------------------


class CommitmentActor(str, Enum):
    """Who a commitment binds."""

    #: The candidate: "please send your availability by Friday".
    USER = "user"
    #: Someone on the other side: "we will get back to you next week".
    EXTERNAL_PERSON = "external_person"


class ExtractedCommitment(_Value):
    """One commitment as an extractor reports it, before it has a source."""

    actor: CommitmentActor
    action: str = Field(max_length=MAX_ACTION_CHARS)
    due_at: datetime | None = None
    #: What the commitment depends on, if anything: "if you are still interested".
    condition: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("action")
    @classmethod
    def _action_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise JobSearchContractError("a commitment must say what is owed")
        return value.strip()

    @field_validator("condition")
    @classmethod
    def _blank_condition_is_none(cls, value: str | None) -> str | None:
        return value.strip() or None if value else None

    @field_validator("due_at")
    @classmethod
    def _due_at_naive_utc(cls, value: datetime | None) -> datetime | None:
        return _naive_utc(value)


class Commitment(ExtractedCommitment):
    """A commitment tied to the message it was read out of."""

    source_message_id: str

    @classmethod
    def from_extracted(cls, extracted: ExtractedCommitment, source_message_id: str) -> "Commitment":
        return cls(**extracted.model_dump(), source_message_id=source_message_id)


# --- Extraction -------------------------------------------------------------


class RecruiterEventExtraction(_Value):
    """What an extractor says a message is. The schema model output is held to.

    Every field is something read out of the message; nothing here names an
    application, a status or a message id. Those are decided by code.
    """

    classification: CommunicationEventClassification
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str | None = Field(default=None, max_length=500)
    requires_reply: bool = False
    commitments: tuple[ExtractedCommitment, ...] = Field(
        default=(), max_length=MAX_COMMITMENTS_PER_MESSAGE
    )


class ExtractionSource(str, Enum):
    """Which extractor an extraction came from."""

    MODEL = "model"
    #: The deterministic rules, used because the model's output would not validate.
    FALLBACK = "fallback"


class ExtractionOutcome(_Value):
    """An extraction and how it was arrived at."""

    extraction: RecruiterEventExtraction
    source: ExtractionSource
    #: Model outputs rejected before this extraction was produced.
    invalid_attempts: int = Field(default=0, ge=0)


#: The lifecycle move each classification proposes. A proposal: the store
#: still runs it through `validate_application_status_transition` against the
#: stored status, and an offer email on an application that never reached
#: INTERVIEWING is refused there, not forced.
IMPLIED_STATUS: dict[CommunicationEventClassification, ApplicationStatus | None] = {
    CommunicationEventClassification.RECRUITER_RESPONSE: ApplicationStatus.RESPONSE,
    CommunicationEventClassification.INTERVIEW_INVITE: ApplicationStatus.INTERVIEWING,
    CommunicationEventClassification.REJECTION: ApplicationStatus.REJECTED,
    CommunicationEventClassification.OFFER: ApplicationStatus.OFFER,
    CommunicationEventClassification.ACTION_REQUIRED: ApplicationStatus.RESPONSE,
    CommunicationEventClassification.GENERAL_UPDATE: None,
    CommunicationEventClassification.UNRELATED: None,
}


# --- Correlation ------------------------------------------------------------


class ApplicationCandidate(_Value):
    """What is on file about one application that a message could be matched on."""

    application_id: UUID
    company: str
    title: str
    status: ApplicationStatus
    #: The posting's id at its source: a requisition or job number.
    reference: str | None = None
    #: Provider threads earlier messages for this application arrived on.
    thread_ids: tuple[str, ...] = ()
    #: Addresses earlier messages for this application came from.
    contact_addresses: tuple[str, ...] = ()


class CorrelationSignal(str, Enum):
    """The identifiers a correlation can rest on."""

    REVIEWER_CONFIRMED = "reviewer_confirmed"
    THREAD = "thread"
    REFERENCE = "reference"
    KNOWN_SENDER = "known_sender"
    SENDER_DOMAIN = "sender_domain"
    COMPANY_NAME = "company_name"
    TITLE = "title"


#: How much each identifier counts for on its own. Combined as independent
#: evidence (`1 - prod(1 - w)`), so agreement adds up but never past 1.
SIGNAL_WEIGHTS: dict[CorrelationSignal, float] = {
    CorrelationSignal.THREAD: 0.95,
    CorrelationSignal.REFERENCE: 0.9,
    CorrelationSignal.KNOWN_SENDER: 0.6,
    CorrelationSignal.SENDER_DOMAIN: 0.5,
    CorrelationSignal.COMPANY_NAME: 0.35,
    CorrelationSignal.TITLE: 0.3,
}

#: A requisition id shorter than this matches too much text by accident.
_MIN_REFERENCE_CHARS = 5


class CorrelationOutcome(str, Enum):
    MATCHED = "matched"
    #: Something pointed at an application, but not firmly or not at only one.
    NEEDS_REVIEW = "needs_review"
    #: Nothing in the message identifies any application on file.
    UNMATCHED = "unmatched"


class CorrelationCandidate(_Value):
    """One application a message might be about, and why."""

    application_id: UUID
    confidence: float = Field(ge=0.0, le=1.0)
    signals: tuple[CorrelationSignal, ...]


class ApplicationCorrelation(_Value):
    """Which application a message is about, if that could be established.

    `application_id` is set only for `MATCHED`. A `NEEDS_REVIEW` correlation
    keeps its best guesses in `candidates` for the reviewer and nowhere else,
    so no caller can mistake a suggestion for a link.
    """

    outcome: CorrelationOutcome
    application_id: UUID | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    signals: tuple[CorrelationSignal, ...] = ()
    threshold: float = DEFAULT_CORRELATION_THRESHOLD
    candidates: tuple[CorrelationCandidate, ...] = ()

    @model_validator(mode="after")
    def _linked_only_when_matched(self) -> "ApplicationCorrelation":
        if (self.outcome is CorrelationOutcome.MATCHED) != (self.application_id is not None):
            raise JobSearchContractError(
                "a correlation names an application if, and only if, it matched one"
            )
        return self


def _contains_phrase(haystack: str, phrase: str) -> bool:
    """Whole-word containment over `canonical_text` forms."""
    return bool(phrase) and f" {phrase} " in f" {haystack} "


def _sender_domain_labels(address: str | None) -> set[str]:
    if not address or "@" not in address:
        return set()
    return {label for label in address.rsplit("@", 1)[1].lower().split(".") if label}


def _signals_for(
    message: RecruiterMessage, candidate: ApplicationCandidate, text: str
) -> tuple[CorrelationSignal, ...]:
    signals: list[CorrelationSignal] = []
    if message.thread_id and message.thread_id in candidate.thread_ids:
        signals.append(CorrelationSignal.THREAD)
    reference = (candidate.reference or "").strip()
    if len(reference) >= _MIN_REFERENCE_CHARS and re.search(
        rf"(?<![A-Za-z0-9]){re.escape(reference)}(?![A-Za-z0-9])",
        f"{message.subject or ''}\n{message.body}",
        re.IGNORECASE,
    ):
        signals.append(CorrelationSignal.REFERENCE)
    sender = (message.from_address or "").strip().lower()
    if sender and sender in {address.lower() for address in candidate.contact_addresses}:
        signals.append(CorrelationSignal.KNOWN_SENDER)
    company = canonical_company(candidate.company)
    if company and company.replace(" ", "") in _sender_domain_labels(sender):
        signals.append(CorrelationSignal.SENDER_DOMAIN)
    if _contains_phrase(text, company):
        signals.append(CorrelationSignal.COMPANY_NAME)
    if _contains_phrase(text, canonical_text(candidate.title)):
        signals.append(CorrelationSignal.TITLE)
    return tuple(signals)


def _combined(signals: Sequence[CorrelationSignal]) -> float:
    miss = 1.0
    for signal in signals:
        miss *= 1.0 - SIGNAL_WEIGHTS[signal]
    return round(1.0 - miss, 4)


def correlate_application(
    message: RecruiterMessage,
    candidates: Sequence[ApplicationCandidate],
    *,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    confirmed_application_id: UUID | None = None,
) -> ApplicationCorrelation:
    """Decide which application `message` is about, from identifiers alone.

    Deterministic: the same message and candidates give the same answer, and
    no model is consulted. `confirmed_application_id` is a reviewer's answer
    to an earlier `NEEDS_REVIEW`; it must name one of `candidates`, so a
    confirmation cannot attach a message to someone else's application.
    """
    if confirmed_application_id is not None:
        if confirmed_application_id not in {c.application_id for c in candidates}:
            raise JobSearchContractError(
                f"application {confirmed_application_id} was confirmed for message "
                f"{message.provider_message_id} but is not one of this user's applications"
            )
        return ApplicationCorrelation(
            outcome=CorrelationOutcome.MATCHED,
            application_id=confirmed_application_id,
            confidence=1.0,
            signals=(CorrelationSignal.REVIEWER_CONFIRMED,),
            threshold=threshold,
        )

    text = canonical_text(f"{message.subject or ''} {message.body}")
    scored = [
        CorrelationCandidate(
            application_id=candidate.application_id,
            confidence=_combined(signals),
            signals=signals,
        )
        for candidate in candidates
        if (signals := _signals_for(message, candidate, text))
    ]
    if not scored:
        return ApplicationCorrelation(outcome=CorrelationOutcome.UNMATCHED, threshold=threshold)

    # Ties broken by id so the order, and so the answer, is reproducible.
    scored.sort(key=lambda c: (-c.confidence, str(c.application_id)))
    best = scored[0]
    runner_up = scored[1].confidence if len(scored) > 1 else 0.0
    if best.confidence >= threshold and best.confidence - runner_up >= CORRELATION_AMBIGUITY_MARGIN:
        return ApplicationCorrelation(
            outcome=CorrelationOutcome.MATCHED,
            application_id=best.application_id,
            confidence=best.confidence,
            signals=best.signals,
            threshold=threshold,
        )
    return ApplicationCorrelation(
        outcome=CorrelationOutcome.NEEDS_REVIEW,
        confidence=best.confidence,
        signals=best.signals,
        threshold=threshold,
        candidates=tuple(scored[:3]),
    )


# --- The classified, correlated event ----------------------------------------


_MAX_READABLE_MESSAGE_ID = 200


def communication_dedupe_key(provider_message_id: str) -> str:
    """The key one inbound message is recorded under, however often it is delivered.

    Derived from the provider's message id alone -- not from the application --
    so a redelivery collapses onto the first row even if correlation would
    have answered differently the second time.
    """
    message_id = provider_message_id.strip()
    if not message_id:
        raise JobSearchContractError("a message with no provider id cannot be deduplicated")
    if len(message_id) > _MAX_READABLE_MESSAGE_ID:
        # Keys derived from this one are stored in varchar(255) columns.
        return f"message:sha256:{hashlib.sha256(message_id.encode('utf-8')).hexdigest()}"
    return f"message:{message_id}"


class RecruiterEvent(_Value):
    """One inbound message, classified and correlated."""

    message: RecruiterMessage
    classification: CommunicationEventClassification
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str | None = None
    requires_reply: bool = False
    commitments: tuple[Commitment, ...] = ()
    source: ExtractionSource
    correlation: ApplicationCorrelation

    @classmethod
    def build(
        cls,
        message: RecruiterMessage,
        outcome: ExtractionOutcome,
        correlation: ApplicationCorrelation,
    ) -> "RecruiterEvent":
        extraction = outcome.extraction
        return cls(
            message=message,
            classification=extraction.classification,
            confidence=extraction.confidence,
            summary=extraction.summary,
            requires_reply=extraction.requires_reply,
            commitments=tuple(
                Commitment.from_extracted(commitment, message.provider_message_id)
                for commitment in extraction.commitments
            ),
            source=outcome.source,
            correlation=correlation,
        )

    @property
    def dedupe_key(self) -> str:
        return communication_dedupe_key(self.message.provider_message_id)


class TriageAction(str, Enum):
    RECORD = "record"
    REVIEW = "review"
    IGNORE = "ignore"


class RecruiterEventTriage(_Value):
    """What to do with one event.

    `RECORD` with a `review_reason` means the event is linked and stored but a
    person is still told: the application was identified, the meaning was not.
    """

    action: TriageAction
    transition: ApplicationStatus | None = None
    review_reason: str | None = None


def triage(
    event: RecruiterEvent,
    *,
    min_classification_confidence: float = DEFAULT_MIN_CLASSIFICATION_CONFIDENCE,
) -> RecruiterEventTriage:
    """Decide whether an event is recorded, reviewed or dropped.

    A transition is proposed only when the application is identified *and* the
    classification is confident. Everything short of that either reaches a
    person or, for mail that neither names an application nor claims to be a
    lifecycle signal, is dropped.
    """
    unrelated = event.classification is CommunicationEventClassification.UNRELATED
    outcome = event.correlation.outcome

    if outcome is CorrelationOutcome.MATCHED:
        confirmed = CorrelationSignal.REVIEWER_CONFIRMED in event.correlation.signals
        # A reviewer who tied it to the application has already answered this.
        if unrelated and not confirmed:
            return RecruiterEventTriage(
                action=TriageAction.REVIEW,
                review_reason="classified unrelated, but its identifiers match an application",
            )
        if event.confidence < min_classification_confidence:
            return RecruiterEventTriage(
                action=TriageAction.RECORD,
                review_reason=(
                    f"classification confidence {event.confidence:.2f} is below "
                    f"{min_classification_confidence:.2f}"
                ),
            )
        return RecruiterEventTriage(
            action=TriageAction.RECORD, transition=IMPLIED_STATUS[event.classification]
        )

    if unrelated:
        return RecruiterEventTriage(action=TriageAction.IGNORE)
    if outcome is CorrelationOutcome.NEEDS_REVIEW:
        return RecruiterEventTriage(
            action=TriageAction.REVIEW,
            review_reason=(
                f"application match confidence {event.correlation.confidence:.2f} is below "
                f"{event.correlation.threshold:.2f} or not unique"
            ),
        )
    if event.classification is CommunicationEventClassification.GENERAL_UPDATE:
        return RecruiterEventTriage(action=TriageAction.IGNORE)
    return RecruiterEventTriage(
        action=TriageAction.REVIEW,
        review_reason="no application on file matches this message",
    )


class RecruiterEventRecord(_Value):
    """What storing one event did, as the store reported it.

    `created=False` is a redelivery: the row, the transition and the events
    all belong to the first delivery and nothing was written this time.
    """

    dedupe_key: str
    application_id: UUID
    communication_event_id: UUID
    created: bool
    status: ApplicationStatus
    transitioned: bool = False
    #: A proposed status the lifecycle would not allow from where the
    #: application stood. Recorded, not forced.
    refused_transition: ApplicationStatus | None = None


class RecruiterEventReview(_Value):
    """One event waiting on a person."""

    dedupe_key: str
    provider_message_id: str
    classification: CommunicationEventClassification
    reason: str
    #: Set when the event was recorded against an application and only its
    #: meaning or its transition is in question.
    application_id: UUID | None = None
    candidates: tuple[CorrelationCandidate, ...] = ()


class InterviewInviteReceived(_Value):
    """Payload of `application.interview_invite_received`.

    The contract a calendar step consumes: enough to propose a hold and to
    find the thread again, without re-reading or re-classifying the message.
    `commitments` carries whatever times and deadlines were extracted.
    """

    application_id: UUID
    source_message_id: str
    received_at: datetime
    thread_id: str | None = None
    from_address: str | None = None
    subject: str | None = None
    summary: str | None = None
    commitments: tuple[Commitment, ...] = ()

    @classmethod
    def from_event(cls, event: RecruiterEvent, application_id: UUID) -> "InterviewInviteReceived":
        message = event.message
        return cls(
            application_id=application_id,
            source_message_id=message.provider_message_id,
            received_at=message.received_at,
            thread_id=message.thread_id,
            from_address=message.from_address,
            subject=message.subject,
            summary=event.summary,
            commitments=event.commitments,
        )


# --- Reply drafts -----------------------------------------------------------


class ReplyDraft(_Value):
    """A proposed reply. Text only: sending it is a separate, approved action."""

    recipient: str
    subject: str
    body: str


_REPLY_OPENERS: dict[CommunicationEventClassification, str] = {
    CommunicationEventClassification.INTERVIEW_INVITE: (
        "Thank you for the invitation. I would be glad to interview."
    ),
    CommunicationEventClassification.OFFER: (
        "Thank you for the offer. I am reviewing the details and will respond shortly."
    ),
    CommunicationEventClassification.ACTION_REQUIRED: (
        "Thank you for letting me know what you need from me."
    ),
}


def template_reply(event: RecruiterEvent) -> ReplyDraft:
    """A plain acknowledgement, with what the candidate owes spelled back.

    The dependency-free default. It states nothing the message did not: no
    availability, no acceptance, no facts about the candidate.
    """
    lines = [
        "Hello,",
        "",
        _REPLY_OPENERS.get(event.classification, "Thank you for your message."),
    ]
    owed = [c for c in event.commitments if c.actor is CommitmentActor.USER and c.due_at]
    if owed:
        soonest = min(owed, key=lambda c: c.due_at or datetime.max)
        lines.append(f"I will follow up before {soonest.due_at:%B %d}.")
    lines += ["", "Best regards"]
    subject = event.message.subject or "your message"
    return ReplyDraft(
        recipient=event.message.from_address or "",
        subject=subject if subject.lower().startswith("re:") else f"Re: {subject}",
        body="\n".join(lines),
    )


# --- Deadlines in free text --------------------------------------------------

_MONTHS = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): number for number, name in enumerate(calendar.month_abbr) if name})
_MONTHS["sept"] = 9
_WEEKDAYS = {name.lower(): number for number, name in enumerate(calendar.day_name)}
_WEEKDAYS.update({name.lower(): number for number, name in enumerate(calendar.day_abbr)})

_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY_RE = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))

#: A date is only read as a deadline when one of these introduces it.
_LEAD = (
    r"\b(?:by|before|until|till|no later than|due(?:\s+(?:on|by))?|on|for)\s+"
    r"(?:the\s+)?(?:(?:end|close)\s+of\s+(?:the\s+)?(?:day|business)\s+)?(?:eod\s+|cob\s+)?"
    r"(?:on\s+)?(?:this\s+|next\s+|coming\s+)?"
)
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(_LEAD + body, re.IGNORECASE)
    for body in (
        r"(?P<iso>\d{4}-\d{2}-\d{2})",
        rf"(?:(?:{_WEEKDAY_RE})\.?,?\s+)?(?P<month>{_MONTH_RE})\.?\s+(?P<day>\d{{1,2}})"
        r"(?:st|nd|rd|th)?(?:,?\s+(?P<year>\d{4}))?",
        rf"(?:(?:{_WEEKDAY_RE})\.?,?\s+)?(?:the\s+)?(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?"
        rf"(?P<month>{_MONTH_RE})\b\.?(?:,?\s+(?P<year>\d{{4}}))?",
        r"(?P<nmonth>\d{1,2})/(?P<nday>\d{1,2})(?:/(?P<nyear>\d{2,4}))?",
        rf"(?P<weekday>{_WEEKDAY_RE})\b",
        r"(?P<tomorrow>tomorrow)\b",
    )
)
_RELATIVE_RE = re.compile(
    r"\b(?:within|in)\s+(?:the\s+next\s+)?(?P<count>\d{1,3})\s+"
    r"(?P<business>business\s+|working\s+)?(?P<unit>day|week)s?\b",
    re.IGNORECASE,
)
_END_OF_WEEK_RE = re.compile(
    r"\b(?:by|before)\s+(?:the\s+)?end\s+of\s+(?:(?:the|this)\s+)?week\b", re.I
)
_END_OF_DAY_RE = re.compile(
    r"\b(?:by|before)\s+(?:the\s+)?(?:(?:end|close)\s+of\s+(?:the\s+)?(?:day|business)|eod|cob)"
    r"(?:\s+today)?\b|\bby\s+today\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b(?:at\s+)?(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>a\.?m\.?|p\.?m\.?)",
    re.IGNORECASE,
)


_RECENT_PAST = timedelta(days=180)


def _next_weekday(reference: datetime, weekday: int) -> datetime:
    """The next such weekday strictly after `reference`'s date."""
    return reference + timedelta(days=(weekday - reference.weekday() - 1) % 7 + 1)


def _add_business_days(reference: datetime, count: int) -> datetime:
    day = reference
    while count > 0:
        day += timedelta(days=1)
        if day.weekday() < 5:
            count -= 1
    return day


def _time_near(text: str, start: int) -> time | None:
    match = _TIME_RE.search(text, start, start + 40)
    if match is None:
        return None
    hour = int(match["hour"]) % 12
    if match["meridiem"].lower().startswith("p"):
        hour += 12
    minute = int(match["minute"] or 0)
    return time(hour, minute) if minute < 60 else None


def _calendar_date(match: re.Match[str], reference: datetime) -> datetime | None:
    groups = match.groupdict()
    try:
        if groups.get("iso"):
            return datetime.strptime(groups["iso"], "%Y-%m-%d")
        if groups.get("weekday"):
            return _next_weekday(reference, _WEEKDAYS[groups["weekday"].lower()])
        if groups.get("tomorrow"):
            return reference + timedelta(days=1)
        if groups.get("nmonth"):
            month, day, year = int(groups["nmonth"]), int(groups["nday"]), groups.get("nyear")
        else:
            month, day, year = _MONTHS[groups["month"].lower()], int(groups["day"]), groups["year"]
        if year:
            return datetime(int(year) + (2000 if len(year) == 2 else 0), month, day)
        # No year given: the next time that date comes round -- unless it only
        # just passed, in which case it is a date being recalled ("you applied
        # on October 1"), not one being set.
        candidate = datetime(reference.year, month, day)
        if candidate.date() < reference.date():
            if reference - candidate < _RECENT_PAST:
                return None
            candidate = datetime(reference.year + 1, month, day)
        return candidate
    except ValueError:
        return None


def parse_deadline(text: str, *, reference: datetime) -> datetime | None:
    """The first deadline stated in `text`, relative to `reference`, or `None`.

    Deliberately narrow. It reads the forms recruiters actually write -- "by
    Friday", "before October 16", "by 10/16 at 5pm", "within 3 business days",
    "by end of week" -- and returns `None` for anything else rather than
    guessing. Times are taken as written, with no timezone: the message does
    not say whose clock it means.
    """
    found: list[tuple[int, datetime, int]] = []
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            date = _calendar_date(match, reference)
            if date is not None:
                found.append((match.start(), date, match.end()))
    for match in _RELATIVE_RE.finditer(text):
        count = int(match["count"])
        if match["unit"].lower() == "week":
            date = reference + timedelta(weeks=count)
        elif match["business"]:
            date = _add_business_days(reference, count)
        else:
            date = reference + timedelta(days=count)
        found.append((match.start(), date, match.end()))
    for match in _END_OF_WEEK_RE.finditer(text):
        friday = reference + timedelta(days=(4 - reference.weekday()) % 7)
        found.append((match.start(), friday, match.end()))
    for match in _END_OF_DAY_RE.finditer(text):
        found.append((match.start(), reference, match.end()))
    if not found:
        return None

    # Earliest in the text; at the same position the longest match is the
    # most specific reading ("by Friday, October 16" over "by Friday").
    _start, date, end = min(found, key=lambda item: (item[0], -item[2]))
    return datetime.combine(date.date(), _time_near(text, end) or DEFAULT_DEADLINE_TIME)


__all__ = [
    "DEFAULT_CORRELATION_THRESHOLD",
    "CORRELATION_AMBIGUITY_MARGIN",
    "DEFAULT_MIN_CLASSIFICATION_CONFIDENCE",
    "MAX_COMMITMENTS_PER_MESSAGE",
    "DEFAULT_DEADLINE_TIME",
    "CommitmentActor",
    "ExtractedCommitment",
    "Commitment",
    "RecruiterEventExtraction",
    "ExtractionSource",
    "ExtractionOutcome",
    "IMPLIED_STATUS",
    "ApplicationCandidate",
    "CorrelationSignal",
    "SIGNAL_WEIGHTS",
    "CorrelationOutcome",
    "CorrelationCandidate",
    "ApplicationCorrelation",
    "correlate_application",
    "communication_dedupe_key",
    "RecruiterEvent",
    "TriageAction",
    "RecruiterEventTriage",
    "triage",
    "RecruiterEventRecord",
    "RecruiterEventReview",
    "InterviewInviteReceived",
    "ReplyDraft",
    "template_reply",
    "parse_deadline",
]

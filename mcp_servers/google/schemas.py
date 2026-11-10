"""Typed contracts for the Google tools: what a caller may send, and what Google may return.

Two directions, two postures:

**Requests** are proposals, possibly from a model. Each tool has a `*Params`
model (the side effect's parameters) and, for a mutating tool, an `*Intent`
that adds the idempotency key. Every field that ends up in a URL path is held
to a character set that cannot traverse or inject (`GoogleId`,
`CalendarId`), and every field that ends up in a mail header refuses CR and LF
(`HeaderText`), so no argument can add a `Bcc:` or reach another endpoint.
None of them names an account or a credential: that is the gateway-attached
`ToolAuthorization`.

**Responses** are untrusted data. Google returns whatever a stranger put in an
email, and the `Raw*` models below read only the fields this server uses,
ignore the rest, and reject a payload whose *structure* is wrong (an id that
is not an id, a list that is not a list). What a tool hands back is then
built from scratch as one of the output models -- fixed fields, text
truncated to a cap and stripped of control characters, and every result
stamped `content_trust: "untrusted"` -- so nothing in a message body or a
header can add a field to a result, let alone change which tools exist.
"""

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from personalos.domain.models import Intent, MutatingIntent

# --- Field types --------------------------------------------------------------

#: A Gmail message, thread or draft id, or a Calendar event id: an opaque
#: token Google issues. URL-path safe by construction.
GoogleId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,256}$")]

#: `primary`, or a calendar's address-shaped id.
CalendarId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._@+-]{1,256}$")]

#: One mailbox address. No display names, no lists, no header syntax.
EmailAddress = Annotated[
    str,
    StringConstraints(
        pattern=r"^[^\s<>()\[\],;:\"\\@]+@[^\s<>()\[\],;:\"\\@]+\.[^\s<>()\[\],;:\"\\@]+$",
        max_length=320,
    ),
]

#: Text that lands in a mail header or a one-line field. CR and LF are what
#: would let it start a header of its own.
HeaderText = Annotated[str, StringConstraints(pattern=r"^[^\r\n]*$", max_length=998)]

#: An RFC 5322 `Message-ID`, as `In-Reply-To` carries it.
MessageIdHeader = Annotated[str, StringConstraints(pattern=r"^<[^\s<>\r\n]{1,994}>$")]

_PROPERTY_KEY = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

MAX_SEARCH_RESULTS = 25
MAX_EVENT_RESULTS = 100
MAX_BODY_CHARS = 100_000


def _aware(value: datetime | None, field: str) -> datetime | None:
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{field} must carry a timezone offset")
    return value


# --- Requests -----------------------------------------------------------------


class GmailSearchParams(Intent):
    """`gmail_search`: Gmail's own search syntax, e.g. `from:recruiter newer_than:7d`."""

    query: str = Field(min_length=1, max_length=1000)
    max_results: int = Field(10, ge=1, le=MAX_SEARCH_RESULTS)
    page_token: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,512}$")] | None = None


class GmailReadParams(Intent):
    """`gmail_read_message`."""

    message_id: GoogleId


class GmailMessageParams(Intent):
    """The message `gmail_create_draft` saves and `gmail_send_message` sends."""

    to: list[EmailAddress] = Field(min_length=1, max_length=20)
    cc: list[EmailAddress] = Field(default_factory=list, max_length=20)
    subject: HeaderText
    body: str = Field(max_length=MAX_BODY_CHARS)
    thread_id: GoogleId | None = None
    in_reply_to: MessageIdHeader | None = None


class GmailCreateDraftIntent(GmailMessageParams, MutatingIntent):
    """Typed parameters for the mutating `gmail_create_draft`."""


class GmailSendIntent(GmailMessageParams, MutatingIntent):
    """Typed parameters for the mutating `gmail_send_message`."""


class CalendarListParams(Intent):
    """`calendar_list_events`: single events, ordered by start time."""

    calendar_id: CalendarId = "primary"
    time_min: datetime | None = None
    time_max: datetime | None = None
    query: str | None = Field(None, max_length=500)
    max_results: int = Field(25, ge=1, le=MAX_EVENT_RESULTS)

    @model_validator(mode="after")
    def _check_window(self) -> "CalendarListParams":
        _aware(self.time_min, "time_min")
        _aware(self.time_max, "time_max")
        if self.time_min and self.time_max and self.time_max <= self.time_min:
            raise ValueError("time_max must be after time_min")
        return self


class CalendarCreateParams(Intent):
    """`calendar_create_event`."""

    calendar_id: CalendarId = "primary"
    summary: HeaderText = Field(min_length=1)
    start: datetime
    end: datetime
    description: str | None = Field(None, max_length=8000)
    location: HeaderText | None = None
    attendees: list[EmailAddress] = Field(default_factory=list, max_length=50)
    #: Written to the event's private extended properties, e.g. an
    #: idempotency stamp a later reconciliation looks the event up by.
    private_properties: dict[str, str] = Field(default_factory=dict, max_length=20)

    @field_validator("private_properties")
    @classmethod
    def _check_properties(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if not _PROPERTY_KEY.match(key) or len(item) > 1024:
                raise ValueError("private property keys are [A-Za-z0-9_.-]{1,64}, values ≤1024")
        return value

    @model_validator(mode="after")
    def _check_times(self) -> "CalendarCreateParams":
        _aware(self.start, "start")
        _aware(self.end, "end")
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class CalendarCreateIntent(CalendarCreateParams, MutatingIntent):
    """Typed parameters for the mutating `calendar_create_event`."""


class CalendarUpdateParams(Intent):
    """`calendar_update_event`: a patch; fields left out are left alone."""

    calendar_id: CalendarId = "primary"
    event_id: GoogleId
    summary: HeaderText | None = None
    start: datetime | None = None
    end: datetime | None = None
    description: str | None = Field(None, max_length=8000)
    location: HeaderText | None = None

    @model_validator(mode="after")
    def _check_patch(self) -> "CalendarUpdateParams":
        if all(
            getattr(self, name) is None
            for name in ("summary", "start", "end", "description", "location")
        ):
            raise ValueError("an update must change at least one field")
        _aware(self.start, "start")
        _aware(self.end, "end")
        if self.start and self.end and self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class CalendarUpdateIntent(CalendarUpdateParams, MutatingIntent):
    """Typed parameters for the mutating `calendar_update_event`."""


class CalendarDeleteParams(Intent):
    """`calendar_delete_event`."""

    calendar_id: CalendarId = "primary"
    event_id: GoogleId


class CalendarDeleteIntent(CalendarDeleteParams, MutatingIntent):
    """Typed parameters for the mutating `calendar_delete_event`."""


# --- What Google returns (read, never trusted) --------------------------------


class _Raw(BaseModel):
    """A Google API object: unknown fields are dropped, not carried."""

    model_config = ConfigDict(extra="ignore")


class RawMessageRef(_Raw):
    """An entry in a `messages.list` page."""

    id: GoogleId
    threadId: GoogleId | None = None


class RawMessageList(_Raw):
    """`users.messages.list`."""

    messages: list[RawMessageRef] = Field(default_factory=list, max_length=500)
    nextPageToken: Annotated[str, StringConstraints(max_length=512)] | None = None
    resultSizeEstimate: int | None = None


class RawHeader(_Raw):
    """One message header."""

    name: Annotated[str, StringConstraints(max_length=256)]
    value: Annotated[str, StringConstraints(max_length=20_000)]


class RawMessage(_Raw):
    """`users.messages.get`. `payload` is walked by hand, depth-limited."""

    id: GoogleId
    threadId: GoogleId
    labelIds: list[Annotated[str, StringConstraints(max_length=256)]] = Field(
        default_factory=list, max_length=200
    )
    snippet: Annotated[str, StringConstraints(max_length=20_000)] = ""
    internalDate: Annotated[str, StringConstraints(pattern=r"^\d{1,16}$")] | None = None
    payload: dict[str, Any] | None = None


class RawDraft(_Raw):
    """`users.drafts.create`."""

    id: GoogleId
    message: RawMessageRef


class RawSentMessage(_Raw):
    """`users.messages.send`."""

    id: GoogleId
    threadId: GoogleId
    labelIds: list[Annotated[str, StringConstraints(max_length=256)]] = Field(
        default_factory=list, max_length=200
    )


class RawEventTime(_Raw):
    """An event's `start` or `end`: a timed `dateTime` or an all-day `date`."""

    dateTime: datetime | None = None
    date: Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")] | None = None


class RawAttendee(_Raw):
    """One event attendee."""

    email: Annotated[str, StringConstraints(max_length=320)] | None = None
    responseStatus: Annotated[str, StringConstraints(max_length=32)] | None = None


class RawExtendedProperties(_Raw):
    """An event's extended properties; only the private half is read."""

    private: dict[str, Annotated[str, StringConstraints(max_length=1024)]] = Field(
        default_factory=dict, max_length=300
    )


class RawEvent(_Raw):
    """`events.get` / `events.insert` / `events.patch`, and a `list` item."""

    id: GoogleId
    status: Annotated[str, StringConstraints(max_length=32)] | None = None
    summary: Annotated[str, StringConstraints(max_length=20_000)] | None = None
    description: Annotated[str, StringConstraints(max_length=100_000)] | None = None
    location: Annotated[str, StringConstraints(max_length=20_000)] | None = None
    htmlLink: Annotated[str, StringConstraints(max_length=2048)] | None = None
    start: RawEventTime | None = None
    end: RawEventTime | None = None
    attendees: list[RawAttendee] = Field(default_factory=list, max_length=500)
    extendedProperties: RawExtendedProperties | None = None


class RawEventList(_Raw):
    """`events.list`."""

    items: list[RawEvent] = Field(default_factory=list, max_length=2500)
    nextPageToken: Annotated[str, StringConstraints(max_length=1024)] | None = None


# --- What a tool returns ------------------------------------------------------

#: Stamped on every result: the text fields came from outside and are data to
#: be read, never instructions to be followed.
ContentTrust = Literal["untrusted"]


class _Output(BaseModel):
    """A tool result. Fixed fields only, built by this server, never parsed from Google."""

    model_config = ConfigDict(extra="forbid")

    content_trust: ContentTrust = "untrusted"


class GmailMessageSummary(BaseModel):
    """One search hit."""

    model_config = ConfigDict(extra="forbid")

    message_id: str
    thread_id: str | None
    sender: str
    subject: str
    date: str
    snippet: str
    label_ids: list[str]


class GmailSearchResult(_Output):
    """`gmail_search`."""

    messages: list[GmailMessageSummary]
    next_page_token: str | None
    result_size_estimate: int | None


class GmailMessageResult(_Output):
    """`gmail_read_message`."""

    message_id: str
    thread_id: str
    sender: str
    to: str
    cc: str
    subject: str
    date: str
    rfc822_message_id: str
    label_ids: list[str]
    snippet: str
    body_text: str
    body_truncated: bool


class GmailDraftResult(_Output):
    """`gmail_create_draft`."""

    draft_id: str
    message_id: str
    thread_id: str | None


class GmailSendResult(_Output):
    """`gmail_send_message`."""

    message_id: str
    thread_id: str
    label_ids: list[str]


class CalendarAttendee(BaseModel):
    """One attendee, as a result carries it."""

    model_config = ConfigDict(extra="forbid")

    email: str
    response_status: str | None


class CalendarEventResult(BaseModel):
    """One event, as a result carries it. `start`/`end` are ISO strings."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    status: str | None
    summary: str
    description: str
    location: str
    start: str | None
    end: str | None
    all_day: bool
    html_link: str | None
    attendees: list[CalendarAttendee]
    private_properties: dict[str, str]


class CalendarListResult(_Output):
    """`calendar_list_events`."""

    calendar_id: str
    events: list[CalendarEventResult]
    next_page_token: str | None


class CalendarWriteResult(_Output):
    """`calendar_create_event` / `calendar_update_event`."""

    calendar_id: str
    event: CalendarEventResult


class CalendarDeleteResult(_Output):
    """`calendar_delete_event`. `already_deleted` when Google reported it gone."""

    calendar_id: str
    event_id: str
    deleted: bool
    already_deleted: bool


__all__ = [
    "GoogleId",
    "CalendarId",
    "EmailAddress",
    "HeaderText",
    "MAX_SEARCH_RESULTS",
    "MAX_EVENT_RESULTS",
    "MAX_BODY_CHARS",
    "GmailSearchParams",
    "GmailReadParams",
    "GmailMessageParams",
    "GmailCreateDraftIntent",
    "GmailSendIntent",
    "CalendarListParams",
    "CalendarCreateParams",
    "CalendarCreateIntent",
    "CalendarUpdateParams",
    "CalendarUpdateIntent",
    "CalendarDeleteParams",
    "CalendarDeleteIntent",
    "RawMessageRef",
    "RawMessageList",
    "RawHeader",
    "RawMessage",
    "RawDraft",
    "RawSentMessage",
    "RawEventTime",
    "RawEvent",
    "RawEventList",
    "GmailMessageSummary",
    "GmailSearchResult",
    "GmailMessageResult",
    "GmailDraftResult",
    "GmailSendResult",
    "CalendarAttendee",
    "CalendarEventResult",
    "CalendarListResult",
    "CalendarWriteResult",
    "CalendarDeleteResult",
]

"""Connected integrations: which Google capabilities exist, and what consent each needs.

A *capability* is something PersonalOS may do with a Google account -- read
mail, send mail, write calendar events -- and each maps to the narrowest OAuth
scope Google offers for it. Two rules shape the consent flow:

- **Read-only by default.** The first consent asks for `gmail.readonly` and
  `calendar.readonly` and nothing else. No default ever includes a scope that
  can change anything.
- **Write consent is its own step.** A write capability is requested only when
  it is named explicitly, and only on top of an existing read-only connection
  (`personalos.cli auth google --enable gmail.send`). Google is asked to
  include the scopes already granted, so the new grant is a superset.

`openid` and `email` are requested alongside, so the token response names the
account being connected. Both are non-sensitive and grant no access to data.

Pure values, like the rest of `domain`. No route here returns, or has a field
for, a token: `GoogleConnection` is what the API and CLI show.
"""

from collections.abc import Iterable
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

#: Provider name a Google credential is filed under: `cred://google/<account>`.
GOOGLE_PROVIDER = "google"

_SCOPE_ROOT = "https://www.googleapis.com/auth/"

GMAIL_READONLY_SCOPE = _SCOPE_ROOT + "gmail.readonly"
GMAIL_COMPOSE_SCOPE = _SCOPE_ROOT + "gmail.compose"
GMAIL_SEND_SCOPE = _SCOPE_ROOT + "gmail.send"
CALENDAR_READONLY_SCOPE = _SCOPE_ROOT + "calendar.readonly"
CALENDAR_EVENTS_SCOPE = _SCOPE_ROOT + "calendar.events"

#: Identity scopes: enough for the token response to say which account it is.
#: Google echoes `email` back as `.../userinfo.email`.
IDENTITY_SCOPES: tuple[str, ...] = ("openid", "email")
_IDENTITY_SCOPE_FORMS = frozenset({"openid", "email", _SCOPE_ROOT + "userinfo.email"})


class GoogleCapability(str, Enum):
    """Something PersonalOS may do with a connected Google account. Wire-visible."""

    GMAIL_READ = "gmail.read"
    CALENDAR_READ = "calendar.read"
    #: Create drafts. (Google's compose scope also permits sending a draft.)
    GMAIL_COMPOSE = "gmail.compose"
    GMAIL_SEND = "gmail.send"
    CALENDAR_WRITE = "calendar.write"


#: The narrowest scope for each capability.
SCOPE_BY_CAPABILITY: dict[GoogleCapability, str] = {
    GoogleCapability.GMAIL_READ: GMAIL_READONLY_SCOPE,
    GoogleCapability.CALENDAR_READ: CALENDAR_READONLY_SCOPE,
    GoogleCapability.GMAIL_COMPOSE: GMAIL_COMPOSE_SCOPE,
    GoogleCapability.GMAIL_SEND: GMAIL_SEND_SCOPE,
    GoogleCapability.CALENDAR_WRITE: CALENDAR_EVENTS_SCOPE,
}

#: What a plain `auth google` asks for.
DEFAULT_CAPABILITIES: tuple[GoogleCapability, ...] = (
    GoogleCapability.GMAIL_READ,
    GoogleCapability.CALENDAR_READ,
)

#: Capabilities that can change something in the account.
WRITE_CAPABILITIES: frozenset[GoogleCapability] = frozenset(
    {
        GoogleCapability.GMAIL_COMPOSE,
        GoogleCapability.GMAIL_SEND,
        GoogleCapability.CALENDAR_WRITE,
    }
)


def parse_capabilities(values: Iterable[str | GoogleCapability]) -> tuple[GoogleCapability, ...]:
    """Capabilities named by their wire values, deduplicated, in the order given.

    Raises `ValueError` naming the valid values on an unknown one.
    """
    parsed: list[GoogleCapability] = []
    for value in values:
        try:
            capability = GoogleCapability(value)
        except ValueError:
            valid = ", ".join(c.value for c in GoogleCapability)
            raise ValueError(
                f"unknown Google capability {value!r}; expected one of {valid}"
            ) from None
        if capability not in parsed:
            parsed.append(capability)
    return tuple(parsed)


def consent_scopes(enable: Iterable[GoogleCapability] = ()) -> tuple[str, ...]:
    """Every scope one consent request asks for: identity, the defaults, then `enable`."""
    scopes = list(IDENTITY_SCOPES)
    for capability in (*DEFAULT_CAPABILITIES, *enable):
        scope = SCOPE_BY_CAPABILITY[capability]
        if scope not in scopes:
            scopes.append(scope)
    return tuple(scopes)


def capabilities_for(scopes: Iterable[str]) -> list[GoogleCapability]:
    """The capabilities a set of granted scopes covers, in declaration order."""
    granted = set(scopes)
    return [c for c, scope in SCOPE_BY_CAPABILITY.items() if scope in granted]


def data_scopes(scopes: Iterable[str]) -> list[str]:
    """Granted scopes without the identity ones, deduplicated, sorted."""
    return sorted({s for s in scopes if s not in _IDENTITY_SCOPE_FORMS})


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConsentRequest(_Value):
    """A consent flow that has begun and waits for Google's redirect.

    Holds nothing secret: the PKCE verifier stays in the secret store, filed
    under `state`, until the redirect comes back.
    """

    authorization_url: str
    state: str
    redirect_uri: str
    requested_scopes: list[str]
    #: True when this request asks for a write capability.
    includes_write: bool
    expires_at: datetime


class GoogleConnection(_Value):
    """One connected Google account, as the CLI and `GET /v1/integrations` show it."""

    provider: str = GOOGLE_PROVIDER
    account: str
    #: Where the refresh token is filed. A name, not the token.
    credential_ref: str
    #: `active` or `revoked`.
    status: str
    granted_scopes: list[str] = Field(default_factory=list)
    capabilities: list[GoogleCapability] = Field(default_factory=list)
    #: When Google will stop honouring the refresh token without a fresh
    #: consent, if it is known to expire. An app in "Testing" publishing status
    #: gets 7-day refresh tokens; a published app's do not expire on a clock.
    consent_expires_at: datetime | None = None
    #: When the row last changed: the latest consent, or the latest exchange.
    updated_at: datetime | None = None
    last_exchanged_at: datetime | None = None


class IntegrationStatus(_Value):
    """One provider's connection state, for `GET /v1/integrations`."""

    provider: str
    connected: bool
    #: True when the provider was disconnected or refused its credential, and
    #: nothing will work until it is connected again.
    reconnect_required: bool = False
    health: str | None = None
    health_reason: str | None = None
    accounts: list[GoogleConnection] = Field(default_factory=list)


class RevokeResult(_Value):
    """What disconnecting an account did."""

    account: str
    credential_ref: str
    #: Google confirmed the grant is gone (or already was).
    revoked_at_provider: bool
    #: The keychain entry was removed.
    secret_deleted: bool
    #: Why Google could not be told, when it could not.
    provider_error: str | None = None


__all__ = [
    "GOOGLE_PROVIDER",
    "GMAIL_READONLY_SCOPE",
    "GMAIL_COMPOSE_SCOPE",
    "GMAIL_SEND_SCOPE",
    "CALENDAR_READONLY_SCOPE",
    "CALENDAR_EVENTS_SCOPE",
    "IDENTITY_SCOPES",
    "GoogleCapability",
    "SCOPE_BY_CAPABILITY",
    "DEFAULT_CAPABILITIES",
    "WRITE_CAPABILITIES",
    "parse_capabilities",
    "consent_scopes",
    "capabilities_for",
    "data_scopes",
    "ConsentRequest",
    "GoogleConnection",
    "IntegrationStatus",
    "RevokeResult",
]

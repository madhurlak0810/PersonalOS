"""The HTTP port the Google tools call through, and its httpx implementation.

`GoogleHttp` is the seam: the tools build a method, a URL and a body, hand
over the short-lived `AccessToken` they were just brokered, and get parsed
JSON back. A fake Google API (`tests/fixtures/fake_google.py`) implements the
same port, which is what lets the contract tests run offline against real
Google response shapes.

Failures are mapped onto the shared taxonomy by status code alone. A response
body is never put into an exception: an API error body can quote the request,
and the request carried a bearer token.
"""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import httpx

from personalos.domain.credentials import AccessToken
from personalos.domain.errors import (
    Conflict,
    NotFound,
    ProviderPermanentFailure,
    RetryableFailure,
    ToolFailure,
    ValidationFailed,
)

GOOGLE_PROVIDER = "google"

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"

#: Largest response body read from Google. A message is fetched as structured
#: JSON, not raw MIME, so this is generous; it bounds a hostile mailbox.
MAX_RESPONSE_BYTES = 10 * 1024 * 1024

#: 403 reasons that mean "slow down", not "you may not".
_RATE_LIMIT_REASONS = frozenset({"rateLimitExceeded", "userRateLimitExceeded", "quotaExceeded"})


class GoogleResponseInvalid(ToolFailure):
    """Google answered with something that is not the shape the tool expects.

    The message names the fields that failed, never their values: the values
    are untrusted content and the message is read by whoever called the tool.
    """

    retryable = False
    default_message = "Google returned a response this tool could not validate"


class GoogleGone(NotFound):
    """The resource existed and has been deleted (HTTP 410)."""

    default_message = "the Google resource has been deleted"


@runtime_checkable
class GoogleHttp(Protocol):
    """Port: one authenticated call to a Google REST API."""

    async def request(
        self,
        method: str,
        url: str,
        *,
        token: AccessToken,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
    ) -> Any:
        """Send the call with `token`; return the parsed JSON body (`None` if empty)."""
        ...


def raise_for_status(status: int, *, reason: str | None = None, retry_after: Any = None) -> None:
    """Raise the taxonomy error for a non-2xx status. `reason` is Google's error reason."""
    if 200 <= status < 300:
        return
    seconds = _seconds(retry_after)
    if status == 401:
        raise ProviderPermanentFailure(
            "Google rejected the access token (HTTP 401); reconnect the Google account",
            provider=GOOGLE_PROVIDER,
            reason=ProviderPermanentFailure.REVOKED_AUTH,
        )
    if status == 403:
        if reason in _RATE_LIMIT_REASONS:
            raise RetryableFailure("Google rate limited the call (HTTP 403)", retry_after=seconds)
        raise ProviderPermanentFailure(
            "Google refused the call (HTTP 403); the grant may lack the scope it needs",
            provider=GOOGLE_PROVIDER,
            reason=ProviderPermanentFailure.MISSING_SCOPE,
        )
    if status == 404:
        raise NotFound("Google has no such resource (HTTP 404)")
    if status == 410:
        raise GoogleGone("the Google resource has been deleted (HTTP 410)")
    if status in (409, 412):
        raise Conflict(f"Google reported a conflict (HTTP {status})")
    if status == 429 or status >= 500:
        raise RetryableFailure(
            f"Google is temporarily unavailable (HTTP {status})", retry_after=seconds
        )
    raise ValidationFailed(f"Google rejected the request (HTTP {status})")


def _seconds(value: Any) -> float | None:
    try:
        seconds = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return seconds if seconds is not None and seconds >= 0 else None


def _error_reason(body: Any) -> str | None:
    """Google's `error.errors[0].reason` (or `error.status`), if it is a short token."""
    if not isinstance(body, dict) or not isinstance(body.get("error"), dict):
        return None
    error = body["error"]
    errors = error.get("errors")
    reason = None
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        reason = errors[0].get("reason")
    reason = reason or error.get("status")
    return reason if isinstance(reason, str) and len(reason) <= 64 else None


class HttpxGoogleHttp:
    """`GoogleHttp` over httpx. One short-lived client per call, nothing cached."""

    def __init__(
        self,
        *,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ):
        """Optionally take the client factory (tests pass a mock transport)."""
        self._client_factory = client_factory or (lambda: httpx.AsyncClient(timeout=15.0))
        self.max_response_bytes = max_response_bytes

    async def request(
        self,
        method: str,
        url: str,
        *,
        token: AccessToken,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
    ) -> Any:
        """Send the call; map failures by status; return parsed JSON."""
        if token.is_expired():
            raise RetryableFailure("the brokered access token expired before the call was made")
        try:
            async with self._client_factory() as client:
                response = await client.request(
                    method,
                    url,
                    params=dict(params) if params else None,
                    json=json,
                    headers={**token.authorization_header(), "Accept": "application/json"},
                )
        except httpx.HTTPError as error:
            # `from None`: the chained exception holds the request, and the
            # request holds the Authorization header.
            raise RetryableFailure(f"Google API unreachable ({type(error).__name__})") from None

        content = response.content
        if len(content) > self.max_response_bytes:
            raise GoogleResponseInvalid(
                f"Google returned {len(content)} bytes; the limit is {self.max_response_bytes}"
            )
        body: Any = None
        if content:
            try:
                body = response.json()
            except ValueError:
                if 200 <= response.status_code < 300:
                    raise GoogleResponseInvalid("Google returned a body that is not JSON") from None
        raise_for_status(
            response.status_code,
            reason=_error_reason(body),
            retry_after=response.headers.get("Retry-After"),
        )
        return body


__all__ = [
    "GOOGLE_PROVIDER",
    "GMAIL_API_BASE",
    "CALENDAR_API_BASE",
    "MAX_RESPONSE_BYTES",
    "GoogleResponseInvalid",
    "GoogleGone",
    "GoogleHttp",
    "HttpxGoogleHttp",
    "raise_for_status",
]

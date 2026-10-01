"""Token exchangers: trade a long-lived secret for something short-lived.

One per provider. An exchanger is handed the secret by the credential broker,
talks to the provider's token endpoint, and returns an `AccessToken`. It never
logs, stores or returns the secret it was given, and it never puts a response
body into an exception -- a token endpoint's error text routinely repeats the
credential it rejected.
"""

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

import httpx

from personalos.domain.credentials import (
    AccessToken,
    CredentialExchangeFailed,
    CredentialRef,
    CredentialRevoked,
    SecretValue,
)

#: Google's OAuth 2.0 token endpoint.
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

#: Lifetime assumed when a provider does not state one.
DEFAULT_ACCESS_TOKEN_TTL = timedelta(minutes=5)

#: How long a leased API key is treated as valid. The key itself does not
#: expire; the lease does, so nothing holds on to it past one execution.
API_KEY_LEASE_TTL = timedelta(minutes=5)

#: OAuth error codes that mean the stored credential is dead, not that the
#: request should be retried.
_TERMINAL_OAUTH_ERRORS = frozenset({"invalid_grant", "invalid_client", "unauthorized_client"})


@runtime_checkable
class TokenExchanger(Protocol):
    """Port: turn one provider's long-lived secret into an access token."""

    async def exchange(
        self, ref: CredentialRef, secret: SecretValue, scopes: Sequence[str]
    ) -> AccessToken:
        """Exchange `secret` for a short-lived token limited to `scopes`."""
        ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GoogleOAuthTokenExchanger:
    """Exchanges a Google OAuth refresh token for an access token.

    Covers Gmail and Calendar alike: they share one token endpoint and differ
    only in the scopes requested.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: SecretValue,
        *,
        token_url: str = GOOGLE_TOKEN_URL,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ):
        """Take the OAuth client this deployment is registered as."""
        self.client_id = client_id
        self._client_secret = client_secret
        self.token_url = token_url
        self._client_factory = client_factory or (lambda: httpx.AsyncClient(timeout=10.0))
        self._clock = clock

    async def exchange(
        self, ref: CredentialRef, secret: SecretValue, scopes: Sequence[str]
    ) -> AccessToken:
        """POST the refresh grant and wrap the access token that comes back."""
        form = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self._client_secret.reveal(),
            "refresh_token": secret.reveal(),
        }
        if scopes:
            form["scope"] = " ".join(scopes)

        try:
            async with self._client_factory() as client:
                response = await client.post(self.token_url, data=form)
        except httpx.HTTPError as error:
            # `from None`: the chained exception holds the request, and the
            # request holds the form above.
            raise CredentialExchangeFailed(
                f"token endpoint unreachable for {ref} ({type(error).__name__})"
            ) from None

        try:
            body = response.json()
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}

        if response.status_code != 200:
            code = body.get("error")
            code = code if isinstance(code, str) and len(code) <= 64 else "unknown_error"
            message = f"token exchange for {ref} failed: HTTP {response.status_code} ({code})"
            if code in _TERMINAL_OAUTH_ERRORS:
                raise CredentialRevoked(message)
            raise CredentialExchangeFailed(message)

        access_token = body.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise CredentialExchangeFailed(f"token exchange for {ref} returned no access token")

        expires_in = body.get("expires_in")
        ttl = (
            timedelta(seconds=expires_in)
            if isinstance(expires_in, (int, float)) and expires_in > 0
            else DEFAULT_ACCESS_TOKEN_TTL
        )
        granted = body.get("scope")
        return AccessToken(
            credential_ref=ref,
            secret=SecretValue(access_token),
            expires_at=self._clock() + ttl,
            scopes=tuple(granted.split()) if isinstance(granted, str) else tuple(scopes),
            token_type=body.get("token_type") or "Bearer",
        )


class ApiKeyExchanger:
    """Leases a static API key for one execution.

    A job-board API key cannot be traded for anything shorter lived, so there
    is no exchange to make. What this preserves is the shape: the key still
    lives only in the secret store, still reaches a tool only through the
    broker at the moment of the call, and still arrives wrapped and expiring.
    """

    def __init__(
        self,
        *,
        lease_ttl: timedelta = API_KEY_LEASE_TTL,
        clock: Callable[[], datetime] = _utcnow,
    ):
        """Set how long a leased key may be held."""
        self.lease_ttl = lease_ttl
        self._clock = clock

    async def exchange(
        self, ref: CredentialRef, secret: SecretValue, scopes: Sequence[str]
    ) -> AccessToken:
        """Wrap the key as an expiring lease."""
        return AccessToken(
            credential_ref=ref,
            secret=secret,
            expires_at=self._clock() + self.lease_ttl,
            scopes=tuple(scopes),
        )


__all__ = [
    "GOOGLE_TOKEN_URL",
    "DEFAULT_ACCESS_TOKEN_TTL",
    "API_KEY_LEASE_TTL",
    "TokenExchanger",
    "GoogleOAuthTokenExchanger",
    "ApiKeyExchanger",
]

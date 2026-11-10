"""Google's OAuth 2.0 Authorization Code flow with PKCE, for an installed app.

The half of OAuth that `GoogleOAuthTokenExchanger` does not do: getting the
*first* refresh token. Three calls, each to a Google endpoint that a test
replaces through `client_factory`:

- `authorization_url` -- where the user's browser goes to consent. It carries
  the PKCE challenge, never the verifier.
- `exchange_code` -- trades the code the redirect brought back, plus the
  verifier, for tokens.
- `revoke` -- withdraws the grant at Google.

Like the exchanger, nothing here logs a token, a code or a verifier, and no
error message carries a response body: an OAuth endpoint's error text
routinely repeats what it rejected. Every secret that comes back is wrapped
in a `SecretValue` before it leaves this module.
"""

import base64
import hashlib
import json
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlencode

import httpx

from personalos.domain.credentials import (
    CredentialExchangeFailed,
    CredentialRevoked,
    SecretValue,
)

from .exchange import GOOGLE_TOKEN_URL

#: Google's consent page.
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"

#: Google's revocation endpoint.
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

#: RFC 7636 allows 43-128 characters; 64 random bytes base64url-encode to 86.
_VERIFIER_BYTES = 64

#: Error codes that mean the code or client was refused for good.
_TERMINAL_ERRORS = frozenset({"invalid_grant", "invalid_client", "unauthorized_client"})


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


@dataclass(frozen=True)
class PkcePair:
    """A PKCE verifier and its S256 challenge (RFC 7636)."""

    verifier: SecretValue
    challenge: str

    @classmethod
    def generate(cls) -> "PkcePair":
        """A fresh pair from the OS random source."""
        verifier = _b64url(secrets.token_bytes(_VERIFIER_BYTES))
        return cls.from_verifier(SecretValue(verifier))

    @classmethod
    def from_verifier(cls, verifier: SecretValue) -> "PkcePair":
        """Rebuild the pair for a stored verifier."""
        digest = hashlib.sha256(verifier.reveal().encode("ascii")).digest()
        return cls(verifier=verifier, challenge=_b64url(digest))


def new_state() -> str:
    """An unguessable `state` value, safe as a credential-reference name."""
    return secrets.token_hex(16)


@dataclass(frozen=True)
class GrantedTokens:
    """What a successful code exchange returned. Secrets stay wrapped."""

    refresh_token: SecretValue
    access_token: SecretValue
    scopes: tuple[str, ...]
    #: The account the consent was for, from the ID token.
    account: str
    #: Seconds until the refresh token stops working, when Google says.
    refresh_token_expires_in: timedelta | None = None

    def __reduce__(self):
        raise TypeError("GrantedTokens cannot be pickled, copied or serialized")


class GoogleOAuthClient:
    """The three Google OAuth endpoints a consent flow needs."""

    def __init__(
        self,
        client_id: str,
        client_secret: SecretValue,
        *,
        auth_url: str = GOOGLE_AUTH_URL,
        token_url: str = GOOGLE_TOKEN_URL,
        revoke_url: str = GOOGLE_REVOKE_URL,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ):
        """Take the OAuth client this deployment is registered as."""
        self.client_id = client_id
        self._client_secret = client_secret
        self.auth_url = auth_url
        self.token_url = token_url
        self.revoke_url = revoke_url
        self._client_factory = client_factory or (lambda: httpx.AsyncClient(timeout=10.0))

    def authorization_url(
        self,
        *,
        redirect_uri: str,
        scopes: Sequence[str],
        state: str,
        code_challenge: str,
        login_hint: str | None = None,
    ) -> str:
        """The consent page URL.

        `access_type=offline` and `prompt=consent` make Google issue a refresh
        token every time, not only on an account's first consent;
        `include_granted_scopes` makes a later, wider consent a superset of the
        earlier one rather than a replacement for it.
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true",
        }
        if login_hint:
            params["login_hint"] = login_hint
        return f"{self.auth_url}?{urlencode(params)}"

    async def exchange_code(
        self, *, code: SecretValue, verifier: SecretValue, redirect_uri: str
    ) -> GrantedTokens:
        """Trade an authorization code for tokens."""
        form = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self._client_secret.reveal(),
            "code": code.reveal(),
            "code_verifier": verifier.reveal(),
            "redirect_uri": redirect_uri,
        }
        response = await self._post(self.token_url, form, "code exchange")
        body = _json_object(response)
        if response.status_code != 200:
            code_ = _error_code(body)
            message = f"Google code exchange failed: HTTP {response.status_code} ({code_})"
            if code_ in _TERMINAL_ERRORS:
                raise CredentialRevoked(message)
            raise CredentialExchangeFailed(message)

        refresh_token = body.get("refresh_token")
        access_token = body.get("access_token")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise CredentialExchangeFailed(
                "Google returned no refresh token; the consent did not grant offline access"
            )
        if not isinstance(access_token, str) or not access_token:
            raise CredentialExchangeFailed("Google returned no access token")

        account = _account_from_id_token(body.get("id_token"))
        if account is None:
            raise CredentialExchangeFailed(
                "Google returned no verified account email; the 'email' scope is required"
            )
        granted = body.get("scope")
        expires = body.get("refresh_token_expires_in")
        return GrantedTokens(
            refresh_token=SecretValue(refresh_token),
            access_token=SecretValue(access_token),
            scopes=tuple(granted.split()) if isinstance(granted, str) else (),
            account=account,
            refresh_token_expires_in=(
                timedelta(seconds=expires)
                if isinstance(expires, (int, float)) and expires > 0
                else None
            ),
        )

    async def revoke(self, token: SecretValue) -> None:
        """Withdraw the grant a token belongs to.

        Revoking a refresh token withdraws the whole grant. A token Google no
        longer knows (`invalid_token`) is already revoked, which is the
        outcome asked for, so that is success too.
        """
        response = await self._post(self.revoke_url, {"token": token.reveal()}, "revocation")
        if response.status_code == 200:
            return
        code = _error_code(_json_object(response))
        if response.status_code == 400 and code == "invalid_token":
            return
        raise CredentialExchangeFailed(
            f"Google revocation failed: HTTP {response.status_code} ({code})"
        )

    async def _post(self, url: str, form: dict[str, str], what: str) -> httpx.Response:
        try:
            async with self._client_factory() as client:
                return await client.post(url, data=form)
        except httpx.HTTPError as error:
            # `from None`: the chained exception holds the request, and the
            # request holds the form.
            raise CredentialExchangeFailed(
                f"Google {what} endpoint unreachable ({type(error).__name__})"
            ) from None


def _json_object(response: httpx.Response) -> dict:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _error_code(body: dict) -> str:
    code = body.get("error")
    return code if isinstance(code, str) and 0 < len(code) <= 64 else "unknown_error"


def _account_from_id_token(id_token: object) -> str | None:
    """The verified email an ID token names.

    The signature is not checked, and need not be: the token came straight
    from Google's token endpoint over TLS, in answer to a request this client
    authenticated (OpenID Connect Core 3.1.3.7).
    """
    if not isinstance(id_token, str) or id_token.count(".") != 2:
        return None
    payload = id_token.split(".")[1]
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (ValueError, TypeError):
        return None
    if not isinstance(claims, dict):
        return None
    email = claims.get("email")
    if not isinstance(email, str) or not email or claims.get("email_verified") is False:
        return None
    return email.lower()


__all__ = [
    "GOOGLE_AUTH_URL",
    "GOOGLE_REVOKE_URL",
    "PkcePair",
    "new_state",
    "GrantedTokens",
    "GoogleOAuthClient",
]

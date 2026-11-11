"""Connecting a Google account: consent, store the refresh token, and disconnect.

`GoogleConnector` is the only code that *writes* a Google refresh token. It
runs the OAuth Authorization Code flow with PKCE against a loopback redirect
and ends with exactly two writes:

- the refresh token, into the OS secret store, under `cred://google/<account>`;
- a `credentials` row naming that reference, its granted scopes and when the
  consent expires -- and no token.

Between `begin` and `complete` the PKCE verifier is a secret too, so it waits
in the secret store as well, filed under the flow's `state`
(`cred://google-oauth-pending/<state>`). That keeps the flow resumable by any
process -- the CLI's own listener, or whichever API worker the browser's
redirect happens to reach -- without the verifier touching the database. The
entry is single-use: `complete` deletes it before doing anything else.

Every secret the flow handles (verifier, code, refresh and access token) is
registered with the redactor the moment it exists, so the Phase D sinks --
logs, traces, checkpoints, audit rows -- drop it even if a provider echoes it.

Consent is split on purpose (see `personalos.domain.integrations`): `begin`
with no `enable` asks for read-only scopes, and a write capability can only be
asked for by naming it, on top of an existing connection.
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone

from personalos.domain.credentials import (
    CredentialExchangeFailed,
    CredentialKind,
    CredentialRef,
    SecretValue,
)
from personalos.domain.errors import Conflict, NotFound, ValidationFailed
from personalos.domain.integrations import (
    GOOGLE_PROVIDER,
    WRITE_CAPABILITIES,
    ConsentRequest,
    GoogleCapability,
    GoogleConnection,
    IntegrationStatus,
    RevokeResult,
    capabilities_for,
    consent_scopes,
    data_scopes,
)
from personalos.domain.redaction import Redactor, default_redactor
from personalos.persistence.integration_health import DEGRADED, IntegrationHealthStore
from personalos.persistence.repositories import CredentialRepository
from personalos.secrets.google_oauth import GoogleOAuthClient, PkcePair, new_state
from personalos.secrets.oauth_loopback import LoopbackRedirectReceiver
from personalos.secrets.store import SecretStore

logger = logging.getLogger(__name__)

#: Provider name the in-flight consent flows are filed under in the secret store.
PENDING_CONSENT_PROVIDER = "google-oauth-pending"

#: How long a begun consent may wait for its redirect.
CONSENT_TTL = timedelta(minutes=10)

#: Lifetime Google gives refresh tokens of an app in "Testing" publishing status.
TESTING_APP_CONSENT_TTL = timedelta(days=7)

#: `integration_health.reason` once the user disconnects.
DISCONNECTED_REASON = "revoked_auth"

_STATE_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class ConsentFlowFailed(ValidationFailed):
    """A consent redirect could not be completed: unknown state, expired, or declined."""

    default_message = "the Google consent flow could not be completed"


class WriteConsentNeedsConnection(Conflict):
    """A write capability was asked for before any read-only connection exists."""

    default_message = (
        "connect Google read-only first (`personalos.cli auth google`), then enable write "
        "capabilities as a separate step"
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _naive(value: datetime | None) -> datetime | None:
    """A UTC datetime as the naive UTC the schema stores."""
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class GoogleConnectionReader:
    """Reads which Google accounts are connected. Needs no keychain and no OAuth client.

    So `auth google --status` and `GET /v1/integrations` work on a machine
    where neither is set up yet, and say so instead of failing.
    """

    def __init__(
        self,
        session_factory: Callable[[], object],
        health: IntegrationHealthStore,
        *,
        clock: Callable[[], datetime] = _utcnow,
    ):
        """Take the database the `credentials` and `integration_health` rows are in."""
        self._session_factory = session_factory
        self._health = health
        self._clock = clock

    @property
    def health(self) -> IntegrationHealthStore:
        """The health store the status is read from."""
        return self._health

    def connections(self, *, active_only: bool = True) -> list[GoogleConnection]:
        """Connected Google accounts, most recently changed first."""
        session = self._session_factory()
        try:
            rows = CredentialRepository(session).list_by_provider(
                GOOGLE_PROVIDER, active_only=active_only
            )
            return [_connection(row) for row in rows]
        finally:
            session.close()

    def status(self) -> IntegrationStatus:
        """Whether Google is connected, as which accounts, and whether it needs reconnecting."""
        every = self.connections(active_only=False)
        active = [c for c in every if c.status == "active"]
        health = self._health.get(GOOGLE_PROVIDER)
        now = self._clock()
        expired = any(
            c.consent_expires_at is not None and c.consent_expires_at <= now for c in active
        )
        degraded = health is not None and (health.reconnect_required or health.status == DEGRADED)
        reconnect = expired or degraded or (not active and bool(every))
        return IntegrationStatus(
            provider=GOOGLE_PROVIDER,
            connected=bool(active),
            reconnect_required=reconnect,
            health=health.status if health else None,
            health_reason=health.reason if health else None,
            action="reconnect" if reconnect else (health.action if health else None),
            last_error=health.last_error if health else None,
            last_success_at=health.last_success_at if health else None,
            last_checked_at=health.last_checked_at if health else None,
            accounts=active,
        )


class GoogleConnector:
    """Begins, completes, reports on and revokes Google consent."""

    def __init__(
        self,
        oauth: GoogleOAuthClient,
        store: SecretStore,
        session_factory: Callable[[], object],
        health: IntegrationHealthStore,
        *,
        redactor: Redactor | None = None,
        clock: Callable[[], datetime] = _utcnow,
        consent_ttl: timedelta = CONSENT_TTL,
        testing_app: bool = True,
    ):
        """Take the OAuth client, the secret store and the database.

        `testing_app` is whether the OAuth client is unverified and in
        "Testing" status, in which case Google expires its refresh tokens after
        7 days and the stored consent is given that expiry.
        """
        self._oauth = oauth
        self._store = store
        self._session_factory = session_factory
        self._health = health
        self._redactor = redactor or default_redactor()
        self._clock = clock
        self._consent_ttl = consent_ttl
        self._testing_app = testing_app
        self._reader = GoogleConnectionReader(session_factory, health, clock=clock)

    # --- Consent ------------------------------------------------------------

    def begin(
        self, *, redirect_uri: str, enable: Iterable[GoogleCapability] = ()
    ) -> ConsentRequest:
        """Start a consent flow; return the URL the user's browser should open.

        With no `enable`, the request is read-only. Naming a write capability
        requires an active connection already, so write consent is always a
        second, explicit step; the consent page is pre-filled with that
        connection's account.
        """
        enable = tuple(GoogleCapability(c) for c in enable)
        includes_write = any(c in WRITE_CAPABILITIES for c in enable)
        login_hint = None
        if includes_write:
            existing = self.connections()
            if not existing:
                raise WriteConsentNeedsConnection()
            login_hint = existing[0].account

        scopes = consent_scopes(enable)
        state = new_state()
        pkce = PkcePair.generate()
        self._redactor.register_secret(pkce.verifier.reveal())
        expires_at = self._clock() + self._consent_ttl
        pending = {
            "code_verifier": pkce.verifier.reveal(),
            "redirect_uri": redirect_uri,
            "scopes": list(scopes),
            "expires_at": expires_at.isoformat(),
        }
        self._store.put(_pending_ref(state), SecretValue(json.dumps(pending)))

        url = self._oauth.authorization_url(
            redirect_uri=redirect_uri,
            scopes=scopes,
            state=state,
            code_challenge=pkce.challenge,
            login_hint=login_hint,
        )
        logger.info("began Google consent (write=%s, scopes=%s)", includes_write, " ".join(scopes))
        return ConsentRequest(
            authorization_url=url,
            state=state,
            redirect_uri=redirect_uri,
            requested_scopes=list(scopes),
            includes_write=includes_write,
            expires_at=expires_at,
        )

    async def complete(
        self,
        *,
        state: str | None,
        code: SecretValue | str | None,
        error: str | None = None,
    ) -> GoogleConnection:
        """Finish a consent flow from what the redirect carried.

        Stores the refresh token under `cred://google/<account>` and records
        the reference. Raises `ConsentFlowFailed` for an unknown, reused or
        expired `state`, or a consent the user declined.
        """
        if isinstance(code, str):
            code = SecretValue(code) if code else None
        if code is not None:
            self._redactor.register_secret(code.reveal())
        if not state or not _STATE_PATTERN.match(state):
            raise ConsentFlowFailed("the consent redirect carried no valid state")

        pending_ref = _pending_ref(state)
        raw = self._store.get(pending_ref)
        # Single use, whatever happens next: a replayed redirect finds nothing.
        self._store.delete(pending_ref)
        if raw is None:
            raise ConsentFlowFailed("unknown or already completed consent flow")
        pending = json.loads(raw.reveal())
        verifier = SecretValue(pending["code_verifier"])
        self._redactor.register_secret(verifier.reveal())

        now = self._clock()
        if datetime.fromisoformat(pending["expires_at"]) <= now:
            raise ConsentFlowFailed("the consent flow expired; start it again")
        if error:
            raise ConsentFlowFailed(f"Google consent was not given ({error})")
        if code is None:
            raise ConsentFlowFailed("the consent redirect carried no authorization code")

        tokens = await self._oauth.exchange_code(
            code=code, verifier=verifier, redirect_uri=pending["redirect_uri"]
        )
        self._redactor.register_secret(tokens.refresh_token.reveal())
        self._redactor.register_secret(
            tokens.access_token.reveal(), expires_at=now + timedelta(hours=1)
        )

        granted = data_scopes(tokens.scopes or pending["scopes"])
        if tokens.refresh_token_expires_in is not None:
            consent_expires_at = now + tokens.refresh_token_expires_in
        elif self._testing_app:
            consent_expires_at = now + TESTING_APP_CONSENT_TTL
        else:
            consent_expires_at = None

        ref = CredentialRef(GOOGLE_PROVIDER, tokens.account)
        self._store.put(ref, tokens.refresh_token)
        session = self._session_factory()
        try:
            row = CredentialRepository(session).record_connected(
                ref=ref,
                kind=CredentialKind.OAUTH_REFRESH_TOKEN,
                scopes=granted,
                consent_expires_at=_naive(consent_expires_at),
            )
            connection = _connection(row)
        finally:
            session.close()
        self._health.mark_healthy(GOOGLE_PROVIDER)

        missing = sorted(set(data_scopes(pending["scopes"])) - set(granted))
        logger.info(
            "connected %s (scopes=%s%s)",
            ref,
            " ".join(granted) or "-",
            f"; not granted: {' '.join(missing)}" if missing else "",
        )
        return connection

    # --- Status -------------------------------------------------------------

    def connections(self, *, active_only: bool = True) -> list[GoogleConnection]:
        """Connected Google accounts, most recently changed first."""
        return self._reader.connections(active_only=active_only)

    def status(self) -> IntegrationStatus:
        """See `GoogleConnectionReader.status`."""
        return self._reader.status()

    # --- Revocation ---------------------------------------------------------

    async def revoke(self, account: str | None = None) -> list[RevokeResult]:
        """Disconnect: revoke the grant at Google and delete the keychain entry.

        Revokes every active Google connection, or only `account`'s. The local
        secret is deleted even if Google cannot be reached -- the point of a
        revoke is that this machine can no longer act -- and the result says
        so, so the user can also withdraw access at
        https://myaccount.google.com/permissions.
        """
        targets = self.connections()
        if account is not None:
            targets = [c for c in targets if c.account == account.lower()]
        if not targets:
            raise NotFound(
                f"no Google account {account} is connected"
                if account
                else "Google is not connected"
            )

        results = []
        for connection in targets:
            ref = CredentialRef.parse(connection.credential_ref)
            secret = self._store.get(ref)
            revoked_at_provider, provider_error = False, None
            if secret is not None:
                self._redactor.register_secret(secret.reveal())
                try:
                    await self._oauth.revoke(secret)
                    revoked_at_provider = True
                except CredentialExchangeFailed as failure:
                    provider_error = failure.message
            self._store.delete(ref)
            session = self._session_factory()
            try:
                CredentialRepository(session).revoke(ref)
            finally:
                session.close()
            results.append(
                RevokeResult(
                    account=connection.account,
                    credential_ref=str(ref),
                    revoked_at_provider=revoked_at_provider,
                    secret_deleted=self._store.get(ref) is None,
                    provider_error=provider_error,
                )
            )
            logger.info(
                "disconnected %s (revoked at Google: %s)",
                ref,
                "yes" if revoked_at_provider else "no",
            )

        if not self.connections():
            self._health.mark_degraded(
                GOOGLE_PROVIDER, reason=DISCONNECTED_REASON, detail="disconnected by the user"
            )
        return results


async def run_loopback_consent(
    connector: GoogleConnector,
    *,
    enable: Iterable[GoogleCapability] = (),
    open_browser: Callable[[str], object],
    timeout: float = 300.0,
    port: int = 0,
) -> GoogleConnection:
    """Run a whole consent on this machine: listen on 127.0.0.1, open the browser, finish.

    `open_browser` is handed the consent URL (the CLI passes something that
    tries `webbrowser.open` and prints it); the listener catches Google's
    redirect, and the flow completes in this process.
    """
    with LoopbackRedirectReceiver(port=port) as receiver:
        request = connector.begin(redirect_uri=receiver.redirect_uri, enable=enable)
        open_browser(request.authorization_url)
        result = await asyncio.to_thread(receiver.wait, timeout)
    return await connector.complete(state=result.state, code=result.code, error=result.error)


def _pending_ref(state: str) -> CredentialRef:
    return CredentialRef(PENDING_CONSENT_PROVIDER, state)


def _connection(row) -> GoogleConnection:
    scopes = list(row.scopes or [])
    return GoogleConnection(
        account=CredentialRef.parse(row.credential_ref).name,
        credential_ref=row.credential_ref,
        status=row.status,
        granted_scopes=scopes,
        capabilities=capabilities_for(scopes),
        consent_expires_at=_aware(row.consent_expires_at),
        updated_at=_aware(row.updated_at),
        last_exchanged_at=_aware(row.last_exchanged_at),
    )


__all__ = [
    "PENDING_CONSENT_PROVIDER",
    "CONSENT_TTL",
    "TESTING_APP_CONSENT_TTL",
    "ConsentFlowFailed",
    "WriteConsentNeedsConnection",
    "GoogleConnectionReader",
    "GoogleConnector",
    "run_loopback_consent",
]

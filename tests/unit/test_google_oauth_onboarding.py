"""Tests for Google OAuth onboarding: `auth google`, its API, and what it leaves behind.

The acceptance criteria this file exists for:

1. **With a fake OAuth server, the flow stores a refresh token under a
   `cred://` reference, and no token value appears in the DB, the logs or a
   serialized checkpoint.**
   `test_the_flow_files_the_refresh_token_under_a_reference_and_nowhere_else`.
2. **`--revoke` removes the keychain entry, and later broker calls fail with a
   recoverable "reconnect required" failure.**
   `test_revoking_deletes_the_keychain_entry_and_the_broker_then_asks_for_reconnect`.
3. **Asking for a write scope is a separate, explicit step from the default
   read-only consent.** `test_the_default_consent_is_read_only` and
   `test_a_write_scope_is_its_own_consent_on_top_of_a_read_only_connection`.

Everything on this machine is real: the loopback listener binds 127.0.0.1 and
the "browser" follows Google's redirect to it over a socket, the refresh token
goes through `KeyringSecretStore` (over an in-memory keyring backend), and
the database is a SQLite file. Only Google is fake: `FakeGoogle` plays the
consent page, the token endpoint (checking the PKCE verifier against the
challenge it was shown) and the revocation endpoint.
"""

import base64
import hashlib
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import uuid4

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.routes.integrations import get_google_connection_reader, get_google_connector
from personalos.domain.credentials import (
    CredentialNotFound,
    CredentialReconnectRequired,
    CredentialRef,
    SecretValue,
)
from personalos.domain.errors import ProviderPermanentFailure
from personalos.domain.failures import RecoveryAction, assess_failure
from personalos.domain.integrations import (
    CALENDAR_EVENTS_SCOPE,
    CALENDAR_READONLY_SCOPE,
    GMAIL_COMPOSE_SCOPE,
    GMAIL_READONLY_SCOPE,
    GMAIL_SEND_SCOPE,
    GoogleCapability,
)
from personalos.domain.redaction import REDACTED, Redactor
from personalos.executor.credentials import CredentialBroker
from personalos.executor.google_connect import (
    PENDING_CONSENT_PROVIDER,
    ConsentFlowFailed,
    GoogleConnectionReader,
    GoogleConnector,
    WriteConsentNeedsConnection,
    run_loopback_consent,
)
from personalos.persistence.checkpointer import SqlAlchemyCheckpointSaver, WorkflowThreadRegistry
from personalos.persistence.integration_health import IntegrationHealthStore
from personalos.persistence.models import CredentialModel
from personalos.secrets.exchange import GoogleOAuthTokenExchanger
from personalos.secrets.google_oauth import GoogleOAuthClient
from personalos.secrets.store import KeyringSecretStore
from tests.fixtures.durable_workflow import session_factory

ACCOUNT = "alice@example.com"
REF = CredentialRef("google", ACCOUNT)
CLIENT_ID = "1234-fake.apps.googleusercontent.com"
CLIENT_SECRET = "fake-client-secret-onboarding-0123"
SERVICE = "personalos-test"
WRITE_SCOPES = {GMAIL_COMPOSE_SCOPE, GMAIL_SEND_SCOPE, CALENDAR_EVENTS_SCOPE}

FAKE_AUTH_URL = "https://accounts.google.test/o/oauth2/v2/auth"
FAKE_TOKEN_URL = "https://oauth2.google.test/token"
FAKE_REVOKE_URL = "https://oauth2.google.test/revoke"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class FakeKeyring:
    """A `keyring` backend in a dict."""

    def __init__(self):
        self.passwords: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.passwords.get((service, username))

    def set_password(self, service, username, password):
        self.passwords[(service, username)] = password

    def delete_password(self, service, username):
        del self.passwords[(service, username)]


class FakeGoogle:
    """Google's consent page, token endpoint and revocation endpoint, in memory.

    Every secret it hands out is distinct and recorded in `secrets_issued`, so
    a test can look for each one wherever it must not be. None has the shape
    of a real Google token (`1//...`, `ya29.`, `4/...`), so the pattern rules
    of Phase D redaction cannot hide a leak from these tests: only the flow
    itself never writing a secret down can make them pass.
    """

    def __init__(self, account: str = ACCOUNT, *, reachable: bool = True):
        self.account = account
        self.reachable = reachable
        self.consents: list[dict[str, str]] = []
        self._codes: dict[str, dict[str, str]] = {}
        self.active_refresh_tokens: dict[str, list[str]] = {}
        self.revoked: list[str] = []
        self.secrets_issued: list[str] = [CLIENT_SECRET]
        self.verifiers_seen: list[str] = []

    # --- The browser -------------------------------------------------------

    def consent(self, url: str, *, deny: bool = False) -> str:
        """The user consents (or declines); return where Google redirects the browser."""
        assert url.startswith(FAKE_AUTH_URL)
        params = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        self.consents.append(params)
        if deny:
            query = {"error": "access_denied", "state": params["state"]}
        else:
            code = f"fake-authcode-{len(self.consents):02d}-{uuid4().hex}"
            self._codes[code] = params
            self.secrets_issued.append(code)
            query = {"code": code, "state": params["state"], "scope": params["scope"]}
        return f"{params['redirect_uri']}?{urlencode(query)}"

    def browse(self, url: str) -> None:
        """A browser: consent, then follow the redirect to the loopback listener."""
        with urllib.request.urlopen(self.consent(url), timeout=5) as response:
            assert response.status == 200

    def decline(self, url: str) -> None:
        with urllib.request.urlopen(self.consent(url, deny=True), timeout=5) as response:
            assert response.status == 200

    # --- The endpoints -----------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        if not self.reachable:
            raise httpx.ConnectError("unreachable", request=request)
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        if str(request.url) == FAKE_REVOKE_URL:
            token = form.get("token")
            if token in self.active_refresh_tokens:
                del self.active_refresh_tokens[token]
                self.revoked.append(token)
                return httpx.Response(200, json={})
            return httpx.Response(400, json={"error": "invalid_token"})
        assert str(request.url) == FAKE_TOKEN_URL
        if form.get("client_id") != CLIENT_ID or form.get("client_secret") != CLIENT_SECRET:
            return httpx.Response(401, json={"error": "invalid_client"})
        if form["grant_type"] == "refresh_token":
            if form.get("refresh_token") not in self.active_refresh_tokens:
                return httpx.Response(400, json={"error": "invalid_grant"})
            access = self._mint("fake-refreshed-access")
            return httpx.Response(200, json={"access_token": access, "expires_in": 3599})
        return self._exchange_code(form)

    def _exchange_code(self, form: dict[str, str]) -> httpx.Response:
        consent = self._codes.pop(form.get("code", ""), None)  # single use
        if consent is None or form.get("redirect_uri") != consent["redirect_uri"]:
            return httpx.Response(400, json={"error": "invalid_grant"})
        verifier = form.get("code_verifier", "")
        self.verifiers_seen.append(verifier)
        challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        if consent["code_challenge_method"] != "S256" or challenge != consent["code_challenge"]:
            return httpx.Response(400, json={"error": "invalid_grant"})

        requested = consent["scope"].split()
        earlier = set()
        if consent.get("include_granted_scopes") == "true":
            for scopes in self.active_refresh_tokens.values():
                earlier |= set(scopes)
        granted = sorted(
            {
                "https://www.googleapis.com/auth/userinfo.email" if s == "email" else s
                for s in requested
            }
            | earlier
        )
        refresh = self._mint("fake-refresh")
        self.active_refresh_tokens[refresh] = granted
        claims = {"email": self.account, "email_verified": True, "sub": "1234"}
        id_token = ".".join(
            [
                _b64url(json.dumps({"alg": "RS256"}).encode()),
                _b64url(json.dumps(claims).encode()),
                _b64url(b"signature"),
            ]
        )
        return httpx.Response(
            200,
            json={
                "access_token": self._mint("fake-access"),
                "refresh_token": refresh,
                "expires_in": 3599,
                "scope": " ".join(granted),
                "token_type": "Bearer",
                "id_token": id_token,
            },
        )

    def _mint(self, prefix: str) -> str:
        value = f"{prefix}-{len(self.secrets_issued):02d}-{uuid4().hex}"
        self.secrets_issued.append(value)
        return value

    def client_factory(self):
        return lambda: httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def oauth_client(self) -> GoogleOAuthClient:
        return GoogleOAuthClient(
            CLIENT_ID,
            SecretValue(CLIENT_SECRET),
            auth_url=FAKE_AUTH_URL,
            token_url=FAKE_TOKEN_URL,
            revoke_url=FAKE_REVOKE_URL,
            client_factory=self.client_factory(),
        )

    def all_secrets(self) -> list[str]:
        return [*self.secrets_issued, *self.verifiers_seen]


class Onboarding:
    """One PersonalOS install: a database file, a keychain, and a fake Google."""

    def __init__(self, tmp_path: Path, google: FakeGoogle | None = None):
        self.db_path = tmp_path / "onboarding.db"
        self.factory = session_factory(self.db_path)
        self.keyring = FakeKeyring()
        self.store = KeyringSecretStore(SERVICE, backend=self.keyring)
        self.google = google or FakeGoogle()
        self.redactor = Redactor()
        self.health = IntegrationHealthStore(self.factory)
        self.connector = GoogleConnector(
            self.google.oauth_client(),
            self.store,
            self.factory,
            self.health,
            redactor=self.redactor,
        )

    async def connect(self, enable=()):
        return await run_loopback_consent(
            self.connector, enable=enable, open_browser=self.google.browse, timeout=10
        )

    def broker(self) -> CredentialBroker:
        exchanger = GoogleOAuthTokenExchanger(
            CLIENT_ID,
            SecretValue(CLIENT_SECRET),
            token_url=FAKE_TOKEN_URL,
            client_factory=self.google.client_factory(),
        )
        return CredentialBroker(self.store, {"google": exchanger}, redactor=self.redactor)

    def row(self) -> CredentialModel:
        session = self.factory()
        try:
            return session.query(CredentialModel).filter_by(credential_ref=str(REF)).one()
        finally:
            session.close()

    def keychain_refs(self) -> list[str]:
        return [name for (service, name) in self.keyring.passwords if service == SERVICE]

    def database_bytes(self) -> bytes:
        return self.db_path.read_bytes()


@pytest.fixture
def install(tmp_path) -> Onboarding:
    return Onboarding(tmp_path)


def _assert_absent(haystack: str | bytes, secrets: list[str], where: str) -> None:
    for secret in secrets:
        needle = secret.encode() if isinstance(haystack, bytes) else secret
        assert needle not in haystack, f"a secret ({secret[:12]}...) leaked into {where}"


# --- 1. The refresh token lands in the keychain and nowhere else ----------------


async def test_the_flow_files_the_refresh_token_under_a_reference_and_nowhere_else(
    install, caplog, capfd
):
    with caplog.at_level("DEBUG"):
        connection = await install.connect()

    # The keychain holds the refresh token, under the reference.
    assert connection.credential_ref == "cred://google/alice@example.com"
    refresh = install.keyring.passwords[(SERVICE, str(REF))]
    assert refresh in install.google.active_refresh_tokens
    # ...and nothing else: the in-flight PKCE verifier was single-use.
    assert install.keychain_refs() == [str(REF)]
    assert not any(PENDING_CONSENT_PROVIDER in ref for ref in install.keychain_refs())

    # The database records the reference, the granted scopes and the expiry.
    row = install.row()
    assert row.kind == "oauth_refresh_token" and row.status == "active"
    assert set(row.scopes) == {GMAIL_READONLY_SCOPE, CALENDAR_READONLY_SCOPE}
    assert row.consent_expires_at is not None  # an unverified app: 7-day tokens
    assert connection.consent_expires_at - datetime.now(timezone.utc) > timedelta(days=6)

    # A serialized checkpoint holding what the flow returned.
    registry = WorkflowThreadRegistry(install.factory)
    registry.register(thread_id="onboarding", workflow_name="supervisor")
    saver = SqlAlchemyCheckpointSaver(install.factory, registry)
    checkpoint = {
        "v": 1,
        "id": "1f000000-0000-6000-8000-000000000001",
        "ts": datetime.utcnow().isoformat(),
        "channel_values": {"google": connection.model_dump(mode="json")},
        "channel_versions": {"google": 1},
        "versions_seen": {},
        "pending_sends": [],
    }
    saver.put(
        {"configurable": {"thread_id": "onboarding", "checkpoint_ns": ""}},
        checkpoint,
        {"source": "loop", "step": 0},
        {"google": 1},
    )

    # No token, code, verifier or client secret anywhere it could be written down.
    secrets = install.google.all_secrets()
    assert len(install.google.verifiers_seen) == 1 and len(secrets) >= 5
    database = install.database_bytes()
    assert str(REF).encode() in database
    _assert_absent(database, secrets, "the database (credentials, checkpoints, health)")
    _assert_absent(caplog.text, secrets, "the logs")
    captured = capfd.readouterr()
    _assert_absent(captured.out + captured.err, secrets, "stdout/stderr")
    _assert_absent(connection.model_dump_json(), secrets, "the returned connection")
    assert "connected cred://google/alice@example.com" in caplog.text

    # Every secret the flow touched is registered for redaction (Phase D).
    for secret in secrets[1:]:
        assert install.redactor.redact_text(f"echo {secret}") == f"echo {REDACTED}"


async def test_the_default_consent_is_read_only(install):
    await install.connect()

    (consent,) = install.google.consents
    scopes = set(consent["scope"].split())
    assert scopes == {"openid", "email", GMAIL_READONLY_SCOPE, CALENDAR_READONLY_SCOPE}
    assert not scopes & WRITE_SCOPES
    # Authorization Code + PKCE on a loopback redirect, asking for offline access.
    assert consent["response_type"] == "code"
    assert consent["code_challenge_method"] == "S256" and "code_verifier" not in consent
    assert consent["redirect_uri"].startswith("http://127.0.0.1:")
    assert consent["access_type"] == "offline"
    assert install.connector.connections()[0].capabilities == [
        GoogleCapability.GMAIL_READ,
        GoogleCapability.CALENDAR_READ,
    ]


# --- 3. Write consent is its own, explicit step -----------------------------------


async def test_a_write_scope_is_its_own_consent_on_top_of_a_read_only_connection(install):
    # Not before a read-only connection exists: nothing is begun, nothing stored.
    with pytest.raises(WriteConsentNeedsConnection):
        install.connector.begin(
            redirect_uri="http://127.0.0.1:1/callback", enable=[GoogleCapability.GMAIL_SEND]
        )
    assert install.google.consents == [] and install.keychain_refs() == []

    await install.connect()
    connection = await install.connect(enable=[GoogleCapability.GMAIL_SEND])

    read, write = install.google.consents
    assert not set(read["scope"].split()) & WRITE_SCOPES
    assert GMAIL_SEND_SCOPE in write["scope"].split()
    assert not set(write["scope"].split()) & (WRITE_SCOPES - {GMAIL_SEND_SCOPE})
    # The second consent builds on the first, for the same account.
    assert write["include_granted_scopes"] == "true"
    assert write["login_hint"] == ACCOUNT
    assert GoogleCapability.GMAIL_SEND in connection.capabilities
    assert {GMAIL_READONLY_SCOPE, CALENDAR_READONLY_SCOPE} <= set(connection.granted_scopes)
    # Still one credential, now holding the wider grant.
    assert install.keychain_refs() == [str(REF)]


def test_a_begun_consent_says_whether_it_asks_for_write(install):
    read = install.connector.begin(redirect_uri="http://127.0.0.1:1/callback")
    assert read.includes_write is False
    assert not set(read.requested_scopes) & WRITE_SCOPES
    # The verifier is not in anything the caller is handed.
    assert "code_verifier" not in read.model_dump_json()


# --- 2. Revoke, then the broker asks for a reconnect --------------------------------


async def test_revoking_deletes_the_keychain_entry_and_the_broker_then_asks_for_reconnect(
    install,
):
    await install.connect()
    broker = install.broker()
    token = await broker.exchange(REF, [GMAIL_READONLY_SCOPE])
    assert token.secret.reveal().startswith("fake-refreshed-access-")
    refresh = install.keyring.passwords[(SERVICE, str(REF))]

    (result,) = await install.connector.revoke()

    assert result.revoked_at_provider and result.secret_deleted
    assert install.google.revoked == [refresh]
    assert install.keychain_refs() == []
    assert install.row().status == "revoked"

    with pytest.raises(CredentialReconnectRequired) as raised:
        await broker.exchange(REF, [GMAIL_READONLY_SCOPE])
    error = raised.value
    assert isinstance(error, ProviderPermanentFailure) and isinstance(error, CredentialNotFound)
    assert error.provider == "google" and error.reason == "revoked_auth"

    # Recoverable as a workflow failure: offered a reconnect, not a retry.
    failure = assess_failure(
        workflow_id=uuid4(),
        thread_id="t",
        step="send",
        checkpoint_id="c",
        task_id="k",
        stored_error=repr(error),
    )
    assert failure.retryable is False
    assert RecoveryAction.RECONNECT in failure.actions
    assert failure.integration.provider == "google" and failure.integration.reconnect_required

    status = install.connector.status()
    assert status.connected is False and status.reconnect_required is True
    assert status.health == "degraded" and status.health_reason == "revoked_auth"


async def test_reconnecting_after_a_revoke_makes_the_integration_healthy_again(install):
    await install.connect()
    await install.connector.revoke()
    await install.connect()

    status = install.connector.status()
    assert status.connected and not status.reconnect_required and status.health == "healthy"
    assert install.row().status == "active"
    assert await install.broker().exchange(REF)


async def test_a_revoke_google_cannot_hear_still_deletes_the_local_secret(install):
    await install.connect()
    install.google.reachable = False

    (result,) = await install.connector.revoke()

    assert result.secret_deleted and not result.revoked_at_provider
    assert "unreachable" in result.provider_error
    assert install.keychain_refs() == []
    with pytest.raises(CredentialReconnectRequired):
        await install.broker().exchange(REF)


# --- A consent redirect is checked ---------------------------------------------------


async def test_a_forged_replayed_declined_or_late_redirect_is_refused(install):
    with pytest.raises(ConsentFlowFailed):
        await install.connector.complete(state="0" * 32, code="4/forged-code-0123456789")
    with pytest.raises(ConsentFlowFailed):
        await install.connector.complete(state="not-a-state", code="x")

    # Declined: nothing is stored and the in-flight entry is gone.
    with pytest.raises(ConsentFlowFailed, match="access_denied"):
        await run_loopback_consent(
            install.connector, open_browser=install.google.decline, timeout=10
        )
    assert install.keychain_refs() == []

    # A redirect replayed after it completed.
    request = install.connector.begin(redirect_uri="http://127.0.0.1:1/callback")
    redirect = parse_qs(urlsplit(install.google.consent(request.authorization_url)).query)
    await install.connector.complete(state=redirect["state"][0], code=redirect["code"][0])
    with pytest.raises(ConsentFlowFailed, match="already completed"):
        await install.connector.complete(state=redirect["state"][0], code=redirect["code"][0])

    # A redirect that arrives after the flow expired.
    late = GoogleConnector(
        install.google.oauth_client(),
        install.store,
        install.factory,
        install.health,
        redactor=install.redactor,
        clock=lambda: datetime.now(timezone.utc) - timedelta(hours=1),
    )
    request = late.begin(redirect_uri="http://127.0.0.1:1/callback")
    redirect = parse_qs(urlsplit(install.google.consent(request.authorization_url)).query)
    with pytest.raises(ConsentFlowFailed, match="expired"):
        await install.connector.complete(state=redirect["state"][0], code=redirect["code"][0])


# --- The API --------------------------------------------------------------------------


def _api(install: Onboarding) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_google_connector] = lambda: install.connector
    app.dependency_overrides[get_google_connection_reader] = lambda: GoogleConnectionReader(
        install.factory, install.health
    )
    return TestClient(app)


def test_the_api_connects_reports_and_revokes_without_returning_token_material(install):
    client = _api(install)

    before = client.get("/v1/integrations")
    assert before.status_code == 200
    assert before.json()["integrations"][0] == {
        "provider": "google",
        "connected": False,
        "reconnect_required": False,
        "health": None,
        "health_reason": None,
        "action": None,
        "last_error": None,
        "last_success_at": None,
        "last_checked_at": None,
        "accounts": [],
    }

    # Write consent before any connection: refused, as its own step.
    refused = client.post("/v1/integrations/google/connect", json={"enable": ["gmail.send"]})
    assert refused.status_code == 409

    begun = client.post("/v1/integrations/google/connect")
    assert begun.status_code == 201
    body = begun.json()
    assert body["includes_write"] is False
    assert body["redirect_uri"].startswith("http://127.0.0.1")
    assert body["redirect_uri"].endswith("/v1/integrations/google/callback")

    # The browser consents, and Google redirects it to the callback.
    redirect = urlsplit(install.google.consent(body["authorization_url"]))
    callback = client.get(f"/v1/integrations/google/callback?{redirect.query}")
    assert callback.status_code == 200, callback.text
    assert callback.json()["credential_ref"] == str(REF)

    status = client.get("/v1/integrations")
    (google,) = status.json()["integrations"]
    assert google["connected"] is True
    (account,) = google["accounts"]
    assert account["account"] == ACCOUNT
    assert set(account["granted_scopes"]) == {GMAIL_READONLY_SCOPE, CALENDAR_READONLY_SCOPE}
    assert account["consent_expires_at"] is not None

    revoked = client.post("/v1/integrations/google/revoke")
    assert revoked.status_code == 200
    assert revoked.json()["results"][0]["secret_deleted"] is True

    for response in (before, begun, callback, status, revoked):
        _assert_absent(response.text, install.google.all_secrets(), "an API response")
    assert install.keychain_refs() == []


def test_the_api_refuses_an_unknown_capability(install):
    response = _api(install).post(
        "/v1/integrations/google/connect", json={"enable": ["gmail.everything"]}
    )
    assert response.status_code == 422


# --- The CLI --------------------------------------------------------------------------


def test_cli_auth_google_connects_shows_status_and_revokes(install, monkeypatch):
    import webbrowser

    import personalos.bootstrap as bootstrap
    from personalos.cli import cli

    monkeypatch.setattr(bootstrap, "build_google_connector", lambda: install.connector)
    monkeypatch.setattr(
        bootstrap,
        "build_google_connection_reader",
        lambda: GoogleConnectionReader(install.factory, install.health),
    )
    monkeypatch.setattr(webbrowser, "open", install.google.browse)
    runner = CliRunner()

    connected = runner.invoke(cli, ["auth", "google", "--timeout", "10"])
    assert connected.exit_code == 0, connected.output
    assert "Google: connected" in connected.output
    assert "account:  alice@example.com" in connected.output
    assert "(read-only)" in connected.output
    assert install.keychain_refs() == [str(REF)]

    status = runner.invoke(cli, ["auth", "google", "--status", "--json"])
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["accounts"][0]["credential_ref"] == str(REF)

    revoked = runner.invoke(cli, ["auth", "google", "--revoke"])
    assert revoked.exit_code == 0, revoked.output
    assert "keychain entry deleted, revoked at Google" in revoked.output
    assert install.keychain_refs() == []

    after = runner.invoke(cli, ["auth", "google", "--status"])
    assert "not connected" in after.output and "reconnect required" in after.output

    for result in (connected, status, revoked, after):
        _assert_absent(result.output, install.google.all_secrets(), "CLI output")


def test_cli_auth_google_refuses_mixed_or_unknown_options():
    from personalos.cli import cli

    runner = CliRunner()
    assert runner.invoke(cli, ["auth", "google", "--status", "--revoke"]).exit_code == 2
    assert runner.invoke(cli, ["auth", "google", "--enable", "gmail.everything"]).exit_code == 2
    assert runner.invoke(cli, ["auth", "google", "--account", ACCOUNT]).exit_code == 2

"""Tests for credential references, the secret store, exchangers and the broker.

The property under test throughout: a long-lived secret is read in exactly one
place (the broker), leaves it only as a short-lived `AccessToken`, and has no
representation -- `repr`, pickle, database row, exception message -- that
could be written down by accident.
"""

import copy
import json
import pickle
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

import httpx
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from personalos.domain.credentials import (
    AccessToken,
    AccessTokenProvider,
    CredentialExchangeFailed,
    CredentialKind,
    CredentialNotFound,
    CredentialRef,
    CredentialRevoked,
    InvalidCredentialRef,
    SecretValue,
)
from personalos.domain.redaction import REDACTED, Redactor
from personalos.executor.credentials import CredentialBroker
from personalos.persistence.models import Base, CredentialModel
from personalos.persistence.repositories import CredentialRepository
from personalos.secrets.exchange import ApiKeyExchanger, GoogleOAuthTokenExchanger
from personalos.secrets.store import (
    InMemorySecretStore,
    KeyringSecretStore,
    SecretStore,
    SecretStoreUnavailable,
)

REFRESH_TOKEN = "1//0gFAKE-refresh-token-FAKEFAKEFAKE-0123456789"
ACCESS_TOKEN = "ya29.FAKE-access-token-FAKEFAKEFAKE-0123456789"
CLIENT_SECRET = "GOCSPX-FAKE-client-secret-0123456789"
API_KEY = "gh-board-FAKE-api-key-0123456789abcdef"

GMAIL = CredentialRef("google", "alice@example.com")
BOARD = CredentialRef("greenhouse", "default")
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


# --- CredentialRef -----------------------------------------------------------


def test_a_reference_round_trips_through_its_string_form():
    assert str(GMAIL) == "cred://google/alice@example.com"
    assert CredentialRef.parse(str(GMAIL)) == GMAIL
    assert CredentialRef.parse(GMAIL) is GMAIL


@pytest.mark.parametrize(
    "value",
    [
        REFRESH_TOKEN,
        "google/alice@example.com",
        "cred://google",
        "cred://google/",
        "cred://Google/alice",
        "cred://google/has space",
        "cred://google/a/b",
        f"cred://google/{'x' * 201}",
        None,
    ],
)
def test_a_string_that_is_not_a_reference_is_refused(value):
    with pytest.raises(InvalidCredentialRef):
        CredentialRef.parse(value)


def test_refusing_a_reference_does_not_quote_the_offered_value():
    """What was offered as a reference may well be the token itself."""
    with pytest.raises(InvalidCredentialRef) as raised:
        CredentialRef.parse(REFRESH_TOKEN)
    assert REFRESH_TOKEN not in str(raised.value)

    with pytest.raises(InvalidCredentialRef) as raised:
        CredentialRef("google", REFRESH_TOKEN + "/")
    assert REFRESH_TOKEN not in str(raised.value)


# --- SecretValue / AccessToken ----------------------------------------------


def test_a_secret_value_has_no_textual_form():
    secret = SecretValue(REFRESH_TOKEN)

    for rendered in (repr(secret), str(secret), f"{secret}", f"{secret!r:>40}"):
        assert REFRESH_TOKEN not in rendered
    assert str(secret) == REDACTED
    assert secret.reveal() == REFRESH_TOKEN


def test_a_secret_value_cannot_be_serialized_or_copied():
    secret = SecretValue(REFRESH_TOKEN)

    with pytest.raises(TypeError):
        pickle.dumps(secret)
    with pytest.raises(TypeError):
        copy.deepcopy(secret)
    with pytest.raises(TypeError):
        json.dumps({"secret": secret})


def test_a_secret_value_in_graph_state_fails_the_checkpoint_serializer():
    """Stray into state and the write fails loudly, rather than persisting."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    with pytest.raises(TypeError):
        JsonPlusSerializer().dumps_typed({"token": SecretValue(REFRESH_TOKEN)})


def test_secret_values_compare_by_content_and_are_unhashable():
    assert SecretValue("abc-123") == SecretValue("abc-123")
    assert SecretValue("abc-123") != SecretValue("abc-124")
    with pytest.raises(TypeError):
        hash(SecretValue("abc-123"))
    with pytest.raises(ValueError):
        SecretValue("")


def test_an_access_token_shows_its_provenance_but_not_its_value():
    token = AccessToken(GMAIL, SecretValue(ACCESS_TOKEN), NOW + timedelta(minutes=5), ("gmail.send",))

    assert ACCESS_TOKEN not in repr(token)
    assert "cred://google/alice@example.com" in repr(token) or "alice@example.com" in repr(token)
    assert token.authorization_header() == {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    assert not token.is_expired(NOW)
    assert token.is_expired(NOW + timedelta(minutes=5))
    with pytest.raises(TypeError):
        pickle.dumps(token)


# --- Secret stores -----------------------------------------------------------


class FakeKeyring:
    """Stands in for the `keyring` module: the three calls the store makes."""

    def __init__(self):
        self.entries: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.entries.get((service, username))

    def set_password(self, service, username, password):
        self.entries[(service, username)] = password

    def delete_password(self, service, username):
        del self.entries[(service, username)]


def test_the_keyring_store_files_secrets_by_reference():
    backend = FakeKeyring()
    store = KeyringSecretStore("personalos-test", backend=backend)
    assert isinstance(store, SecretStore)

    assert store.get(GMAIL) is None
    store.put(GMAIL, SecretValue(REFRESH_TOKEN))

    assert backend.entries == {("personalos-test", "cred://google/alice@example.com"): REFRESH_TOKEN}
    assert store.get(GMAIL).reveal() == REFRESH_TOKEN

    store.delete(GMAIL)
    store.delete(GMAIL)  # already gone: not an error
    assert store.get(GMAIL) is None


def test_storing_a_secret_does_not_log_it(caplog):
    with caplog.at_level("DEBUG"):
        KeyringSecretStore(backend=FakeKeyring()).put(GMAIL, SecretValue(REFRESH_TOKEN))
    assert REFRESH_TOKEN not in caplog.text


def test_the_keyring_store_refuses_a_backend_that_protects_nothing(monkeypatch):
    """`keyring` falls back to a plaintext or null store; that must not pass."""
    import sys
    import types

    class Keyring:
        pass

    Keyring.__module__ = "keyrings.alt.file"
    fake = types.ModuleType("keyring")
    fake.get_keyring = lambda: Keyring()
    monkeypatch.setitem(sys.modules, "keyring", fake)

    with pytest.raises(SecretStoreUnavailable):
        KeyringSecretStore()


def test_the_keyring_store_fails_closed_when_keyring_is_not_installed(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "keyring", None)
    with pytest.raises(SecretStoreUnavailable):
        KeyringSecretStore()


# --- Exchangers --------------------------------------------------------------


def _google(handler, **kwargs) -> GoogleOAuthTokenExchanger:
    return GoogleOAuthTokenExchanger(
        "client-id.apps.example",
        SecretValue(CLIENT_SECRET),
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        clock=lambda: NOW,
        **kwargs,
    )


async def test_the_google_exchanger_trades_a_refresh_token_for_an_access_token():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["form"] = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        return httpx.Response(
            200,
            json={
                "access_token": ACCESS_TOKEN,
                "expires_in": 3599,
                "scope": "https://www.googleapis.com/auth/gmail.send",
                "token_type": "Bearer",
            },
        )

    token = await _google(handler).exchange(GMAIL, SecretValue(REFRESH_TOKEN), ["gmail.send"])

    assert seen["url"] == "https://oauth2.googleapis.com/token"
    assert seen["form"] == {
        "grant_type": "refresh_token",
        "client_id": "client-id.apps.example",
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "scope": "gmail.send",
    }
    assert token.secret.reveal() == ACCESS_TOKEN
    assert token.credential_ref == GMAIL
    assert token.expires_at == NOW + timedelta(seconds=3599)
    assert token.scopes == ("https://www.googleapis.com/auth/gmail.send",)


async def test_a_failed_exchange_reports_the_error_code_and_nothing_from_the_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "error": "temporarily_unavailable",
                "error_description": f"could not use refresh_token={REFRESH_TOKEN}",
            },
        )

    with pytest.raises(CredentialExchangeFailed) as raised:
        await _google(handler).exchange(GMAIL, SecretValue(REFRESH_TOKEN), [])

    message = str(raised.value)
    assert "temporarily_unavailable" in message and "503" in message
    assert REFRESH_TOKEN not in message and CLIENT_SECRET not in message
    assert raised.value.retryable is True


async def test_a_revoked_grant_is_terminal_not_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    with pytest.raises(CredentialRevoked) as raised:
        await _google(handler).exchange(GMAIL, SecretValue(REFRESH_TOKEN), [])
    assert raised.value.retryable is False


async def test_a_transport_failure_does_not_chain_the_request_that_held_the_secret():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed sending {request.content.decode()}", request=request)

    with pytest.raises(CredentialExchangeFailed) as raised:
        await _google(handler).exchange(GMAIL, SecretValue(REFRESH_TOKEN), [])

    assert raised.value.__cause__ is None and raised.value.__suppress_context__
    assert REFRESH_TOKEN not in str(raised.value)


async def test_a_response_without_an_access_token_is_a_failed_exchange():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>captive portal</html>")

    with pytest.raises(CredentialExchangeFailed):
        await _google(handler).exchange(GMAIL, SecretValue(REFRESH_TOKEN), [])


async def test_an_api_key_is_leased_with_an_expiry():
    token = await ApiKeyExchanger(clock=lambda: NOW).exchange(BOARD, SecretValue(API_KEY), ["read"])

    assert token.secret.reveal() == API_KEY
    assert token.expires_at == NOW + timedelta(minutes=5)
    assert token.scopes == ("read",)


# --- Broker ------------------------------------------------------------------


class StubExchanger:
    """Returns a fixed token and records what it was handed."""

    def __init__(self, *, ttl: timedelta = timedelta(minutes=30), error: Exception | None = None):
        self.ttl = ttl
        self.error = error
        self.calls: list[tuple[CredentialRef, str, tuple[str, ...]]] = []

    async def exchange(self, ref, secret, scopes):
        self.calls.append((ref, secret.reveal(), tuple(scopes)))
        if self.error:
            raise self.error
        return AccessToken(ref, SecretValue(ACCESS_TOKEN), NOW + self.ttl, tuple(scopes))


def _broker(exchanger=None, *, redactor=None, store=None) -> tuple[CredentialBroker, StubExchanger]:
    store = store or InMemorySecretStore()
    if store.get(GMAIL) is None:
        store.put(GMAIL, SecretValue(REFRESH_TOKEN))
    exchanger = exchanger or StubExchanger()
    broker = CredentialBroker(
        store, {"google": exchanger}, redactor=redactor or Redactor(), clock=lambda: NOW
    )
    return broker, exchanger


async def test_the_broker_exchanges_a_reference_for_a_short_lived_token():
    broker, exchanger = _broker()
    assert isinstance(broker, AccessTokenProvider)

    token = await broker.exchange("cred://google/alice@example.com", ["gmail.send"])

    assert exchanger.calls == [(GMAIL, REFRESH_TOKEN, ("gmail.send",))]
    assert token.secret.reveal() == ACCESS_TOKEN
    assert token.secret.reveal() != REFRESH_TOKEN
    assert token.expires_at == NOW + timedelta(minutes=30)


async def test_the_broker_exchanges_just_in_time_and_caches_nothing():
    broker, exchanger = _broker()

    await broker.exchange(GMAIL)
    await broker.exchange(GMAIL)

    assert len(exchanger.calls) == 2
    # Nothing on the broker holds a token or a secret between calls.
    held = {name: value for name, value in vars(broker).items() if name != "_store"}
    assert ACCESS_TOKEN not in repr(held) and REFRESH_TOKEN not in repr(held)
    assert not any(isinstance(value, (AccessToken, SecretValue)) for value in held.values())


async def test_the_broker_caps_the_lifetime_a_provider_claims():
    broker, _ = _broker(StubExchanger(ttl=timedelta(days=30)))
    token = await broker.exchange(GMAIL)
    assert token.expires_at == NOW + timedelta(hours=1)


async def test_a_lease_scopes_the_token_to_a_block():
    broker, _ = _broker()
    async with broker.lease(GMAIL, ["calendar.events"]) as token:
        assert token.authorization_header() == {"Authorization": f"Bearer {ACCESS_TOKEN}"}
        assert token.scopes == ("calendar.events",)


async def test_an_unknown_reference_or_provider_is_refused():
    broker, exchanger = _broker()

    with pytest.raises(CredentialNotFound):
        await broker.exchange("cred://google/nobody@example.com")
    with pytest.raises(CredentialExchangeFailed):
        await broker.exchange("cred://unknownprovider/default")
    with pytest.raises(InvalidCredentialRef):
        await broker.exchange(REFRESH_TOKEN)
    assert exchanger.calls == []


async def test_the_broker_makes_everything_it_touches_redactable():
    """Shapeless secrets on purpose: only the known-value rule can catch these."""
    long_lived, minted = "shapeless-long-lived-secret-value", "shapeless-minted-access-value"

    class Shapeless(StubExchanger):
        async def exchange(self, ref, secret, scopes):
            return AccessToken(ref, SecretValue(minted), NOW + self.ttl, tuple(scopes))

    store = InMemorySecretStore()
    store.put(GMAIL, SecretValue(long_lived))
    redactor = Redactor()
    broker, _ = _broker(Shapeless(), redactor=redactor, store=store)
    echo = f"provider echoed {long_lived} then {minted}"
    assert redactor.redact_text(echo) == echo

    await broker.exchange(GMAIL)

    assert redactor.redact_text(echo) == f"provider echoed {REDACTED} then {REDACTED}"


async def test_the_refresh_token_is_redactable_even_when_the_exchange_fails():
    """A failed exchange is the likeliest moment for the secret to be echoed."""
    store = InMemorySecretStore()
    shapeless = CredentialRef("google", "shapeless")
    store.put(shapeless, SecretValue("shapeless-long-lived-secret-value"))
    redactor = Redactor()
    broker = CredentialBroker(
        store,
        {"google": StubExchanger(error=RuntimeError("echo: shapeless-long-lived-secret-value"))},
        redactor=redactor,
    )

    with pytest.raises(RuntimeError) as raised:
        await broker.exchange(shapeless)

    assert redactor.redact_text(str(raised.value)) == f"echo: {REDACTED}"


async def test_the_broker_logs_the_reference_and_never_the_token(caplog):
    broker, _ = _broker()
    with caplog.at_level("DEBUG"):
        await broker.exchange(GMAIL, ["gmail.send"])

    assert "cred://google/alice@example.com" in caplog.text
    assert ACCESS_TOKEN not in caplog.text and REFRESH_TOKEN not in caplog.text


# --- Database rows hold references only --------------------------------------


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    yield db
    db.close()


def test_a_credential_row_stores_the_reference_and_no_token(session):
    repo = CredentialRepository(session)

    row = repo.create(ref=GMAIL, kind=CredentialKind.OAUTH_REFRESH_TOKEN, scopes=["gmail.send"])

    assert row.credential_ref == "cred://google/alice@example.com"
    assert row.provider == "google" and row.kind == "oauth_refresh_token"
    assert repo.get_by_ref(GMAIL).id == row.id
    assert set(row.to_dict()) == {
        "id",
        "user_id",
        "provider",
        "kind",
        "credential_ref",
        "scopes",
        "status",
        "last_exchanged_at",
        "created_at",
        "updated_at",
    }


def test_the_credentials_table_has_no_column_that_could_hold_a_token():
    columns = set(CredentialModel.__table__.columns.keys())
    assert not {c for c in columns if "token" in c or "secret" in c or "value" in c or "key" in c}


def test_the_repository_refuses_a_bare_string_where_a_reference_goes(session):
    with pytest.raises(TypeError):
        CredentialRepository(session).create(ref=REFRESH_TOKEN, kind=CredentialKind.API_KEY)
    with pytest.raises(TypeError):
        CredentialRepository(session).create(ref=str(GMAIL), kind=CredentialKind.API_KEY)


def test_the_schema_itself_refuses_a_non_reference(session):
    """A row written around the repository still cannot hold an arbitrary string."""
    with pytest.raises(IntegrityError):
        session.execute(
            text(
                "INSERT INTO credentials (id, provider, kind, credential_ref, scopes, status, "
                "created_at, updated_at) VALUES ('x', 'google', 'api_key', :ref, '[]', 'active', "
                "'2026-10-01', '2026-10-01')"
            ),
            {"ref": REFRESH_TOKEN},
        )


def test_a_reference_is_unique_and_can_be_revoked(session):
    repo = CredentialRepository(session)
    repo.create(ref=BOARD, kind=CredentialKind.API_KEY)

    with pytest.raises(IntegrityError):
        repo.create(ref=BOARD, kind=CredentialKind.API_KEY)
    session.rollback()

    assert repo.mark_exchanged(BOARD).last_exchanged_at is not None
    assert repo.revoke(BOARD).status == "revoked"

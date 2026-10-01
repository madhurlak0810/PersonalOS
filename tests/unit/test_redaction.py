"""Tests for the redaction layer: the redactor itself, then logs, then traces.

The redactor is a backstop, so the tests are as much about what it must *not*
touch as what it must: a redactor that rewrites `error_code` or
`idempotency_key` corrupts state on its way to a checkpoint, and one that eats
ordinary prose makes logs useless.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from personalos.domain.credentials import SecretValue
from personalos.domain.redaction import REDACTED, Redactor, default_redactor
from personalos.observability.log_redaction import (
    RedactingFilter,
    install_log_redaction,
    scrub_record,
)
from personalos.observability.trace_redaction import RedactingSpanExporter

# Fakes, shaped like the real thing so the format rules are exercised too.
BEARER = "ya29.a0AfH6SMFAKEFAKEFAKEaccess-token_0123456789"
REFRESH = "1//0gFAKEFAKEFAKErefresh-token_0123456789abcdef"
OPAQUE = "zq7-opaque-secret-with-no-recognisable-shape"


@pytest.fixture
def redactor() -> Redactor:
    return Redactor()


# --- Text patterns -----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        f"Authorization: Bearer {OPAQUE}",
        f"authorization: Basic {OPAQUE}",
        f"{{'Authorization': 'Bearer {OPAQUE}'}}",
        f'"Authorization": "Bearer {OPAQUE}"',
        f"Proxy-Authorization: Digest {OPAQUE}",
        f"Cookie: session={OPAQUE}; theme=dark",
        f"Set-Cookie: sid={OPAQUE}; HttpOnly; Secure",
        f"sent header bearer {OPAQUE} upstream",
        f"POST /token refresh_token={OPAQUE}&grant_type=refresh_token",
        f'{{"client_secret": "{OPAQUE}"}}',
        f"GOOGLE_API_KEY={OPAQUE}",
        f"X-Api-Key: {OPAQUE}",
        f"password={OPAQUE}",
        f"https://app.example/callback?code={OPAQUE}&state=xyz",
    ],
)
def test_secret_bearing_text_is_redacted(redactor: Redactor, text: str):
    redacted = redactor.redact_text(text)
    assert OPAQUE not in redacted
    assert REDACTED in redacted


@pytest.mark.parametrize(
    "token",
    [
        BEARER,
        REFRESH,
        "4/0AfJohXFAKEFAKEFAKEFAKEauthcode_0123456789",
        "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE012",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlRkFLRQ",
        "sk-FAKEFAKEFAKEFAKEFAKEFAKE0123",
        "ghp_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE0123",
        "AKIAFAKEFAKEFAKE0123",
    ],
)
def test_well_known_token_shapes_are_redacted_with_no_surrounding_context(
    redactor: Redactor, token: str
):
    redacted = redactor.redact_text(f"upstream said: {token} was rejected")
    assert token not in redacted
    assert redacted == f"upstream said: {REDACTED} was rejected"


def test_a_pem_private_key_is_removed_whole(redactor: Redactor):
    pem = "-----BEGIN PRIVATE KEY-----\nMIIFAKEFAKE\nFAKEFAKE==\n-----END PRIVATE KEY-----"
    assert redactor.redact_text(f"key was {pem} end") == f"key was {REDACTED} end"


@pytest.mark.parametrize(
    "text",
    [
        "tool 'jobs.search_jobs' failed: upstream 503",
        "error_code=tool_failure",
        "status code=200",
        "idempotency_key=apply-acme-2026-10-01",
        "lease_token=3b1e0c9e-5f0e-4c58-9f0e-6f8b7c2d1a11",
        "The candidate passed the token ring networking interview",
        "authorization is required for this action",
        "workflow_id=11111111-1111-1111-1111-111111111111 actor_id=system",
        "",
    ],
)
def test_ordinary_text_is_left_alone(redactor: Redactor, text: str):
    assert redactor.redact_text(text) == text


def test_redaction_is_idempotent(redactor: Redactor):
    """Sinks stack -- gateway, then checkpoint, then log -- so a second pass is routine."""
    redactor.register_secret(OPAQUE)
    text = (
        f"Authorization: Bearer {OPAQUE}\nCookie: sid={OPAQUE}; theme=dark\n"
        f"export TOKEN={OPAQUE} api_key={OPAQUE} https://x/cb?code={OPAQUE}&state=s"
    )

    once = redactor.redact_text(text)

    assert OPAQUE not in once
    assert redactor.redact_text(once) == once
    assert "]]" not in once


# --- Structures --------------------------------------------------------------


def test_sensitive_keys_are_dropped_without_inspecting_the_value(redactor: Redactor):
    payload = {
        "headers": {"Authorization": "x", "Set-Cookie": ["a=b"], "Accept": "application/json"},
        "refresh_token": OPAQUE,
        "nested": [{"client_secret": OPAQUE, "keep": 1}],
    }

    assert redactor.redact(payload) == {
        "headers": {"Authorization": REDACTED, "Set-Cookie": REDACTED, "Accept": "application/json"},
        "refresh_token": REDACTED,
        "nested": [{"client_secret": REDACTED, "keep": 1}],
    }


def test_a_code_key_is_an_oauth_code_only_inside_an_oauth_exchange(redactor: Redactor):
    exchange = {"grant_type": "authorization_code", "code": OPAQUE, "redirect_uri": "https://x"}
    error = {"code": "tool_failure", "message": "upstream 503"}

    assert redactor.redact(exchange)["code"] == REDACTED
    assert redactor.redact(error) == error


def test_keys_that_merely_resemble_sensitive_ones_survive(redactor: Redactor):
    state = {
        "idempotency_key": "apply-acme",
        "lease_token": "abc",
        "error_code": "tool_failure",
        "token_count": 1200,
    }
    assert redactor.redact(state) is state


def test_an_unchanged_value_is_returned_as_the_same_object(redactor: Redactor):
    state = {"shortlist": [{"rank": 1, "tags": ("python", "remote")}], "query": "python"}
    assert redactor.redact(state) is state


def test_redaction_never_mutates_its_input(redactor: Redactor):
    inner = {"detail": f"Authorization: Bearer {OPAQUE}"}
    state = {"receipts": [inner], "untouched": {"a": 1}}

    redacted = redactor.redact(state)

    assert inner["detail"] == f"Authorization: Bearer {OPAQUE}"
    assert OPAQUE not in str(redacted)
    # Siblings that needed no change are shared, not copied.
    assert redacted["untouched"] is state["untouched"]


def test_tuples_sets_and_bytes_are_followed(redactor: Redactor):
    value = (f"Bearer {OPAQUE}", frozenset({f"api_key={OPAQUE}"}), f"token={OPAQUE}".encode())
    redacted = redactor.redact(value)

    assert isinstance(redacted, tuple) and isinstance(redacted[1], frozenset)
    assert OPAQUE not in repr(redacted)


def test_dataclass_and_pydantic_fields_are_followed(redactor: Redactor):
    @dataclass(frozen=True)
    class Parked:
        value: dict
        id: str

    class Receipt(BaseModel):
        ok: bool
        detail: str | None = None

    parked = redactor.redact(Parked(value={"detail": f"Cookie: sid={OPAQUE}"}, id="i-1"))
    receipt = redactor.redact(Receipt(ok=False, detail=f"Authorization: Bearer {OPAQUE}"))

    assert isinstance(parked, Parked) and parked.id == "i-1"
    assert OPAQUE not in repr(parked)
    assert isinstance(receipt, Receipt) and receipt.ok is False
    assert OPAQUE not in receipt.detail


def test_a_secret_value_object_is_replaced_by_the_marker(redactor: Redactor):
    assert redactor.redact({"leaked": SecretValue(OPAQUE)}) == {"leaked": REDACTED}


# --- Known values ------------------------------------------------------------


def test_a_registered_secret_is_removed_wherever_it_appears(redactor: Redactor):
    """The one rule that cannot miss: no pattern describes this string."""
    text = f"provider echoed {OPAQUE} back in a sentence"
    assert redactor.redact_text(text) == text

    redactor.register_secret(OPAQUE)

    assert redactor.redact_text(text) == f"provider echoed {REDACTED} back in a sentence"
    assert redactor.redact({"deep": [{"x": f"<<{OPAQUE}>>"}]}) == {"deep": [{"x": f"<<{REDACTED}>>"}]}


def test_a_short_value_is_not_registered(redactor: Redactor):
    redactor.register_secret("abc")
    assert redactor.redact_text("abc abcdef") == "abc abcdef"


def test_an_expired_token_is_forgotten_on_the_next_registration(redactor: Redactor):
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    redactor.register_secret(OPAQUE, expires_at=past)
    assert REDACTED in redactor.redact_text(OPAQUE)

    redactor.register_secret("another-long-lived-secret")

    assert redactor.redact_text(OPAQUE) == OPAQUE


def test_a_long_lived_registration_is_not_shortened_by_a_later_one(redactor: Redactor):
    redactor.register_secret(OPAQUE)
    redactor.register_secret(OPAQUE, expires_at=datetime.now(timezone.utc) - timedelta(hours=1))
    redactor.register_secret("another-long-lived-secret")

    assert redactor.redact_text(OPAQUE) == REDACTED


# --- Logs --------------------------------------------------------------------


def _record(msg: str, *args, exc_info=None, **extra) -> logging.LogRecord:
    record = logging.LogRecord("test", logging.ERROR, __file__, 1, msg, args, exc_info)
    record.__dict__.update(extra)
    return record


def test_log_message_arguments_are_redacted():
    record = scrub_record(_record("call failed: %s", f"Authorization: Bearer {OPAQUE}"), Redactor())

    assert OPAQUE not in record.getMessage()
    assert record.args is None


def test_a_logged_traceback_is_redacted_and_the_live_exception_dropped():
    try:
        raise RuntimeError(f"401 from provider; request headers: Cookie: sid={OPAQUE}")
    except RuntimeError:
        import sys

        record = scrub_record(_record("boom", exc_info=sys.exc_info()), Redactor())

    formatted = logging.Formatter().format(record)
    assert OPAQUE not in formatted
    assert "RuntimeError" in formatted and REDACTED in formatted
    # A handler holding the exception object could format it for itself.
    assert record.exc_info is None


def test_the_filter_redacts_extra_fields():
    record = _record("request", headers={"Authorization": f"Bearer {OPAQUE}"}, attempt=2)

    assert RedactingFilter(Redactor()).filter(record) is True
    assert record.headers == {"Authorization": REDACTED}
    assert record.attempt == 2


def test_importing_personalos_installs_log_redaction(caplog):
    """No entry point has to remember: any process that loads the package redacts."""
    import personalos  # noqa: F401 - imported for its side effect

    with caplog.at_level(logging.INFO):
        logging.getLogger("some.third.party").info("refresh_token=%s", OPAQUE)

    assert OPAQUE not in caplog.text
    assert f"refresh_token={REDACTED}" in caplog.text


def test_installing_log_redaction_twice_does_not_stack_factories():
    before = logging.getLogRecordFactory()
    install_log_redaction()
    install_log_redaction()
    assert logging.getLogRecordFactory() is before


def test_the_installed_factory_uses_secrets_registered_later(caplog):
    """The broker registers a token long after logging was configured."""
    default_redactor().register_secret(OPAQUE)
    try:
        with caplog.at_level(logging.INFO):
            logging.getLogger("provider").warning("rejected credential %s", OPAQUE)
    finally:
        default_redactor().forget_secrets()

    assert OPAQUE not in caplog.text


# --- Traces ------------------------------------------------------------------


def test_exported_spans_are_redacted():
    exported = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(exported, Redactor())))
    tracer = provider.get_tracer("test")

    with pytest.raises(RuntimeError):
        with tracer.start_as_current_span("gmail.send") as span:
            span.set_attribute("http.request.header.authorization", f"Bearer {OPAQUE}")
            span.set_attribute("http.url", f"https://x/cb?code={OPAQUE}")
            span.set_attribute("http.status_code", 401)
            raise RuntimeError(f"401: Authorization: Bearer {OPAQUE}")

    (span,) = exported.get_finished_spans()
    dumped = span.to_json()
    assert OPAQUE not in dumped
    assert REDACTED in dumped
    assert span.attributes["http.status_code"] == 401
    assert span.name == "gmail.send"
    assert [event.name for event in span.events] == ["exception"]
    assert OPAQUE not in (span.status.description or "")

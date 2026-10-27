"""Structured logging: the JSON shape, the log scope, and redaction on the way out.

The acceptance criterion this file holds: **a log call containing an
Authorization header is emitted with the value redacted** --
`test_an_authorization_header_is_redacted_from_the_emitted_line` and its
neighbours, which cover the header in the message, in an `extra=` mapping,
in an exception, and in the log scope itself.
"""

import io
import json
import logging
from collections.abc import Callable, Iterator
from typing import Any
from uuid import uuid4

import pytest

from personalos.config import Settings
from personalos.domain.context import ExecutionContext
from personalos.domain.redaction import REDACTED
from personalos.observability.structured_logging import (
    LogEvent,
    component_for,
    configure_logging,
    configure_logging_from_settings,
    current_log_scope,
    log_scope,
)
from personalos.policy import ToolIntent
from personalos.policy.engine import PolicyEngine
from personalos.policy.errors import PolicyDenied
from personalos.policy.rules import default_rules

BEARER = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.c2lnbmF0dXJlLXZhbHVl"
OPAQUE = "opaque-bearer-0123456789abcdef"

logger = logging.getLogger("personalos.executor.test")


@pytest.fixture
def emitted() -> Iterator[Callable[[], list[dict[str, Any]]]]:
    """Reads back every line the structured handler wrote, parsed."""
    stream = io.StringIO()
    handler = configure_logging(level="DEBUG", stream=stream)

    def read() -> list[dict[str, Any]]:
        return [json.loads(line) for line in stream.getvalue().splitlines() if line]

    try:
        yield read
    finally:
        logging.getLogger().removeHandler(handler)


def _raw(read: Callable[[], list[dict[str, Any]]]) -> str:
    return json.dumps(read())


# --- Shape -------------------------------------------------------------------------


def test_every_line_has_the_required_fields(emitted):
    logger.info("hello %s", "world")

    (line,) = emitted()
    assert line["message"] == "hello world"
    assert line["level"] == "INFO"
    assert line["component"] == "executor"
    assert line["event"] == "log"
    assert line["timestamp"].endswith("+00:00")
    # Present, if null, outside any workflow, so a query can rely on the key.
    assert line["workflow_id"] is None
    assert line["correlation_id"] is None
    # Optional ids are absent rather than null when nothing bound them.
    assert "action_id" not in line


def test_bound_ids_are_stamped_and_an_explicit_extra_wins(emitted):
    workflow_id, correlation_id, action_id = uuid4(), uuid4(), uuid4()
    other = uuid4()

    with log_scope(workflow_id=workflow_id, correlation_id=correlation_id):
        with log_scope(action_id=action_id, correlation_id=None):
            logger.info("inner")
        logger.info("override", extra={"workflow_id": other, "event": "test.override"})
    logger.info("outside")

    inner, override, outside = emitted()
    assert inner["workflow_id"] == str(workflow_id)
    # `None` does not clear an enclosing value.
    assert inner["correlation_id"] == str(correlation_id)
    assert inner["action_id"] == str(action_id)
    assert override["workflow_id"] == str(other)
    assert override["correlation_id"] == str(correlation_id)
    assert override["event"] == "test.override"
    assert "action_id" not in override
    assert outside["workflow_id"] is None
    assert current_log_scope() == {}


def test_an_exception_is_formatted_into_the_line(emitted):
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.exception("it failed")

    (line,) = emitted()
    assert line["level"] == "ERROR"
    assert "RuntimeError: boom" in line["exception"]


@pytest.mark.parametrize(
    ("name", "component"),
    [
        ("personalos.policy.engine", "policy"),
        ("apps.worker.workflow_runner", "worker"),
        ("apps.api.routes.workflows", "api"),
        ("mcp_servers.google.calendar", "google"),
        ("uvicorn.error", "uvicorn"),
        ("", "root"),
    ],
)
def test_component_is_the_layer_that_logged(name, component):
    assert component_for(name) == component


def test_reconfiguring_replaces_the_handler_instead_of_stacking():
    root = logging.getLogger()
    first = configure_logging(stream=io.StringIO())
    second = configure_logging(stream=io.StringIO())
    try:
        assert first not in root.handlers
        assert second in root.handlers
    finally:
        root.removeHandler(second)


def test_text_format_appends_the_scope():
    stream = io.StringIO()
    handler = configure_logging(fmt="text", stream=stream)
    workflow_id = uuid4()
    try:
        with log_scope(workflow_id=workflow_id):
            logger.info("readable")
    finally:
        logging.getLogger().removeHandler(handler)

    assert stream.getvalue().rstrip().endswith(f"readable [workflow_id={workflow_id}]")


def test_settings_choose_the_format(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr("sys.stderr", stream)
    handler = configure_logging_from_settings(Settings(log_format="text", log_level="WARNING"))
    try:
        logger.warning("plain")
    finally:
        logging.getLogger().removeHandler(handler)
    assert "WARNING personalos.executor.test: plain" in stream.getvalue()


# --- Redaction ---------------------------------------------------------------------


def test_an_authorization_header_is_redacted_from_the_emitted_line(emitted):
    logger.warning("provider rejected the call: Authorization: Bearer %s", BEARER)

    (line,) = emitted()
    assert BEARER not in _raw(emitted)
    assert line["message"] == f"provider rejected the call: Authorization: {REDACTED}"


def test_an_authorization_header_in_extra_fields_is_redacted(emitted):
    logger.info(
        "outbound request",
        extra={
            "headers": {"Authorization": f"Bearer {OPAQUE}", "Accept": "application/json"},
            "cookie": "session=abc123def456",
        },
    )

    (line,) = emitted()
    assert OPAQUE not in _raw(emitted)
    assert line["headers"] == {"Authorization": REDACTED, "Accept": "application/json"}
    assert line["cookie"] == REDACTED


def test_an_authorization_header_in_an_exception_is_redacted(emitted):
    try:
        raise ConnectionError(
            f"401 for request with headers {{'Authorization': 'Bearer {OPAQUE}'}}"
        )
    except ConnectionError:
        logger.exception("provider call failed")

    (line,) = emitted()
    assert OPAQUE not in _raw(emitted)
    assert REDACTED in line["exception"]


def test_oauth_codes_and_tokens_in_a_message_are_redacted(emitted):
    code = "4/0AeaYSHB" + "x" * 30
    logger.info("redirected to https://app/callback?code=%s&state=s1", code)
    logger.info("token response refresh_token=%s", OPAQUE)

    assert code not in _raw(emitted)
    assert OPAQUE not in _raw(emitted)


def test_a_secret_bound_into_the_scope_is_redacted(emitted):
    with log_scope(actor_id=f"Bearer {OPAQUE}"):
        logger.info("scoped")

    assert OPAQUE not in _raw(emitted)


# --- Audit cross-reference ---------------------------------------------------------


def _intent(tool: str, context: ExecutionContext) -> ToolIntent:
    server, _, name = tool.partition(".")
    return ToolIntent(server=server, tool=name, arguments={}, context=context)


def test_policy_decisions_log_as_policy_decision_events(emitted):
    """Every verdict: one `policy.decision` line, citing the ids the audit trail uses."""
    context = ExecutionContext.new(actor_id="user-1")
    engine = PolicyEngine(default_rules())

    with pytest.raises(PolicyDenied):
        engine.authorize(_intent("unknown.tool", context))

    (line,) = [line for line in emitted() if line["event"] == LogEvent.POLICY_DECISION]
    assert line["level"] == "WARNING"
    assert line["component"] == "policy"
    assert line["policy_decision"] == "deny"
    assert line["policy_rule"]
    assert line["workflow_id"] == str(context.workflow_id)
    assert line["correlation_id"] == str(context.correlation_id)
    assert line["actor_id"] == "user-1"
    assert line["tool_name"] == "unknown.tool"

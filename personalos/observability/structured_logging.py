"""Structured logging: one JSON object per line, scoped to the workflow it served.

Every log line names the workflow and request it belongs to without the code
that wrote it having to know either. The ids live in a `ContextVar` (the *log
scope*), which is bound where the ids are known -- the API's correlation
middleware, `tracing.workflow_span`, `tracing.node_span`, every `tracing.span`
that carries an action, command or event id -- and read back by
`LogContextFilter` when a record is emitted. A graph node, a policy check or a
ledger write deep inside a run therefore logs with the run's `workflow_id` and
`correlation_id` attached, the same way its spans carry them as attributes.

Each line is a JSON object with, always:

- `timestamp` (UTC, ISO 8601), `level`, `component` (the layer that logged:
  `api`, `worker`, `policy`, `executor`, ...), `event` (a stable dotted name,
  `"log"` when the call site gave none) and `message`;
- `workflow_id` and `correlation_id`, `null` outside any workflow;

and, when they are bound, `action_id`, `operation_id` (the command or event
being processed), `thread_id`, `actor_id`, `node`, ... plus anything the call
passed as `extra=`. An explicit `extra=` value wins over the scope.

**Event names and levels.** The records an operator cross-references with the
audit trail use the names in `LogEvent` at fixed levels, so a query for
`event="policy.decision"` finds every verdict and a query for `level="ERROR"`
finds every failure and nothing routine:

==========================  =========  ==========================================
event                       level      extra fields
==========================  =========  ==========================================
``policy.decision``         INFO       ``policy_decision``, ``policy_rule``,
                            WARNING    ``policy_decision_id`` (WARNING on deny)
``approval.requested``      INFO       ``approval_id``
``approval.decided``        INFO       ``approval_id``, ``approval_outcome``
``approval.expired``        INFO       ``approval_ids``
``retry.scheduled``         WARNING    ``attempt``, ``max_attempts``, ``retry_delay_seconds``
``retry.abandoned``         WARNING    ``attempt``, ``max_attempts``
``action.failed``           ERROR      ``failure_class``, ``error_code``
``workflow.failed``         ERROR
==========================  =========  ==========================================

`policy_decision_id` and `approval_id` are the ids of the `policy_decisions`
and `approvals` rows, and the audit trail cites the same ids, so a log line
and the audit row about the same verdict join on them.

**Redaction.** Records are already scrubbed by `log_redaction` when they are
created; the formatter redacts the finished object once more before writing
it, so a scope value or an `extra` field that reached the record some other
way still cannot carry a credential out.
"""

import json
import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import IO, Any
from uuid import UUID

from personalos.config import Settings, settings
from personalos.domain.redaction import Redactor, default_redactor
from personalos.observability.log_redaction import (
    STANDARD_RECORD_ATTRIBUTES,
    RedactingFilter,
    install_log_redaction,
)


class LogEvent:
    """`event` names for the records the audit trail is cross-referenced with."""

    POLICY_DECISION = "policy.decision"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_DECIDED = "approval.decided"
    APPROVAL_EXPIRED = "approval.expired"
    RETRY_SCHEDULED = "retry.scheduled"
    RETRY_ABANDONED = "retry.abandoned"
    ACTION_FAILED = "action.failed"
    WORKFLOW_FAILED = "workflow.failed"


#: `event` of a record whose call site did not name one.
DEFAULT_EVENT = "log"

#: Fields every line has, `null` when unbound.
REQUIRED_SCOPE_FIELDS = ("workflow_id", "correlation_id")

#: Scope fields written right after the required ones, when bound, so the ids
#: an operator scans for sit in the same place on every line.
SCOPE_FIELDS = (
    *REQUIRED_SCOPE_FIELDS,
    "action_id",
    "operation_id",
    "thread_id",
    "actor_id",
    "node",
    "graph",
    "tool_name",
    "command_id",
    "event_id",
    "approval_id",
    "attempt",
)

#: Top-level packages whose second segment names the component that logged.
_LAYERED_PACKAGES = frozenset({"personalos", "apps", "mcp_servers"})

_EMPTY: Mapping[str, Any] = MappingProxyType({})

_LOG_SCOPE: ContextVar[Mapping[str, Any]] = ContextVar("personalos_log_scope", default=_EMPTY)

#: Marks the handler `configure_logging` installed, so a second call replaces it.
_HANDLER_MARKER = "_personalos_structured"


# --- Scope ---------------------------------------------------------------------------


def _scope_value(value: Any) -> Any:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, UUID):
        return str(value)
    return value


def current_log_scope() -> Mapping[str, Any]:
    """The ids bound in this context, innermost binding winning."""
    return _LOG_SCOPE.get()


@contextmanager
def log_scope(**fields: Any) -> Iterator[Mapping[str, Any]]:
    """Bind ids to every log record emitted in this block (and tasks it starts).

    Merged over the enclosing scope; a `None` value leaves an enclosing value
    in place rather than clearing it.
    """
    updates = {key: _scope_value(value) for key, value in fields.items() if value is not None}
    if not updates:
        yield current_log_scope()
        return
    merged = MappingProxyType({**current_log_scope(), **updates})
    token = _LOG_SCOPE.set(merged)
    try:
        yield merged
    finally:
        _LOG_SCOPE.reset(token)


class LogContextFilter(logging.Filter):
    """Copies the bound log scope onto each record, leaving `extra=` values alone.

    A handler filter rather than part of the record factory: `logging` refuses
    an `extra=` key that the factory already set, so stamping the scope there
    would make every `extra={"workflow_id": ...}` call raise.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Stamp the scope; never drops a record."""
        for key, value in current_log_scope().items():
            if key not in record.__dict__:
                record.__dict__[key] = value
        return True


# --- Formatting ----------------------------------------------------------------------


def component_for(logger_name: str) -> str:
    """The layer a logger belongs to: `personalos.policy.engine` -> `policy`."""
    parts = logger_name.split(".")
    if parts[0] in _LAYERED_PACKAGES and len(parts) > 1:
        return parts[1]
    return parts[0] or "root"


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    return str(value)


class JsonFormatter(logging.Formatter):
    """Formats a record as one line of JSON. See the module docstring for the fields."""

    def __init__(self, redactor: Redactor | None = None):
        """Use the given redactor, or the process-wide one."""
        super().__init__()
        self._redactor = redactor

    def to_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        """The record as the mapping `format` serializes, redacted."""
        fields = record.__dict__
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "component": fields.get("component") or component_for(record.name),
            "event": fields.get("event") or DEFAULT_EVENT,
            "message": record.getMessage(),
            "logger": record.name,
        }
        for key in SCOPE_FIELDS:
            if key in REQUIRED_SCOPE_FIELDS or fields.get(key) is not None:
                payload[key] = _scope_value(fields.get(key))
        for key, value in fields.items():
            if key in STANDARD_RECORD_ATTRIBUTES or key in payload or key.startswith("_"):
                continue
            payload[key] = _scope_value(value)

        exc_text = record.exc_text
        if record.exc_info and not exc_text:
            exc_text = self.formatException(record.exc_info)
        if exc_text:
            payload["exception"] = exc_text
        if record.stack_info:
            payload["stack"] = record.stack_info
        return (self._redactor or default_redactor()).redact(payload)

    def format(self, record: logging.LogRecord) -> str:
        """One line of JSON."""
        return json.dumps(self.to_dict(record), default=_json_default, ensure_ascii=False)


class _ScopedTextFormatter(logging.Formatter):
    """The plain-text format, with the bound scope appended, for reading at a terminal."""

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        scope = " ".join(
            f"{key}={record.__dict__[key]}"
            for key in SCOPE_FIELDS
            if record.__dict__.get(key) is not None
        )
        return f"{line} [{scope}]" if scope else line


# --- Installing ----------------------------------------------------------------------


def configure_logging(
    *,
    level: str | int = "INFO",
    fmt: str = "json",
    stream: IO[str] | None = None,
    redactor: Redactor | None = None,
) -> logging.Handler:
    """Send every record to `stream` (stderr) as JSON, or as text with `fmt="text"`.

    Replaces the handler an earlier call installed, and leaves any other
    handler alone. Returns the handler, so a test can remove it again.
    """
    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_MARKER, False):
            root.removeHandler(existing)
            existing.close()

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    setattr(handler, _HANDLER_MARKER, True)
    # Scope first, so the redacting filter sees -- and scrubs -- what it added.
    handler.addFilter(LogContextFilter())
    handler.addFilter(RedactingFilter(redactor))
    if fmt == "text":
        handler.setFormatter(
            _ScopedTextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    else:
        handler.setFormatter(JsonFormatter(redactor))
    root.addHandler(handler)
    root.setLevel(level.upper() if isinstance(level, str) else level)
    install_log_redaction(redactor)
    return handler


def configure_logging_from_settings(config: Settings = settings) -> logging.Handler:
    """Install logging as the environment says: `LOG_LEVEL` and `LOG_FORMAT`."""
    return configure_logging(level=config.log_level, fmt=config.log_format)


__all__ = [
    "DEFAULT_EVENT",
    "JsonFormatter",
    "LogContextFilter",
    "LogEvent",
    "REQUIRED_SCOPE_FIELDS",
    "SCOPE_FIELDS",
    "component_for",
    "configure_logging",
    "configure_logging_from_settings",
    "current_log_scope",
    "log_scope",
]

"""Exception text that is safe to write to a span or a log: no echoed input, no credential.

Redaction (Phase D) removes credentials. It cannot remove *content*: a
recruiter's email body or a line of a resume is not a pattern. Most
exception messages carry neither. The one that reliably does is pydantic's
`ValidationError`, which quotes the value that failed to validate
(`input_value='Hi Jordan, thanks for applying...'`). That value is very often
model output or a message body, because validating model output is what the
structured-output layer does all day.

So a validation error is described by *where* and *how* it failed -- field
locations and error types -- and never by what the input was. Every other
exception keeps its message, redacted. `tracing.mark_failed` records
exceptions on spans this way, `log_redaction.scrub_record` formats exception
arguments and tracebacks this way, and `failures.describe_exception` -- the
one-line description the worker logs a failed step with -- uses it too.

In `domain` because it is pure, and because `domain.failures` needs it.
"""

import re
import traceback
from typing import Any

from personalos.domain.redaction import Redactor, default_redactor

try:  # pydantic is a hard dependency; the guard keeps this module import-safe.
    from pydantic import ValidationError
except ImportError:  # pragma: no cover
    ValidationError = ()  # type: ignore[assignment,misc]

#: A location part shown as itself: a field name or an index. Anything else
#: (a dict key taken from the input) is shown as `*`.
_LOCATION_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
#: At most this many errors of one `ValidationError` are listed.
_MAX_ERRORS = 10


def _location(parts: Any) -> str:
    shown = [
        str(part) if isinstance(part, int) or _LOCATION_PART.match(str(part)) else "*"
        for part in parts or ()
    ]
    return ".".join(shown) or "(root)"


def describe_validation_error(error: Any) -> str:
    """`N validation error(s) for Model: field: type; ...`, without the input values."""
    details = error.errors(include_url=False, include_input=False, include_context=False)
    listed = "; ".join(
        f"{_location(item.get('loc'))}: {item.get('type', 'invalid')}"
        for item in details[:_MAX_ERRORS]
    )
    more = f"; and {len(details) - _MAX_ERRORS} more" if len(details) > _MAX_ERRORS else ""
    count = error.error_count()
    noun = "error" if count == 1 else "errors"
    return f"{count} validation {noun} for {error.title}: {listed}{more}"


def is_input_echoing(error: BaseException) -> bool:
    """True for an exception whose message quotes its input (a pydantic `ValidationError`)."""
    return bool(ValidationError) and isinstance(error, ValidationError)


def safe_exception_message(error: BaseException, redactor: Redactor | None = None) -> str:
    """The exception's message, without echoed input, redacted."""
    redactor = redactor or default_redactor()
    if is_input_echoing(error):
        return describe_validation_error(error)
    try:
        text = str(error)
    except Exception:  # noqa: BLE001 - a broken __str__ must not stop the description
        text = ""
    return redactor.redact_text(text)


def _chain(error: BaseException) -> list[BaseException]:
    """`error` and the exceptions it was raised from or during, outermost last."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return list(reversed(chain))


def safe_traceback(error: BaseException, redactor: Redactor | None = None) -> str:
    """The traceback `traceback.format_exception` would give, with every message made safe."""
    redactor = redactor or default_redactor()
    parts: list[str] = []
    chain = _chain(error)
    for index, current in enumerate(chain):
        if index:
            parts.append(
                "\nThe above exception was the direct cause of the following exception:\n\n"
                if chain[index].__cause__ is chain[index - 1]
                else "\nDuring handling of the above exception, another exception occurred:\n\n"
            )
        if current.__traceback__ is not None:
            parts.append("Traceback (most recent call last):\n")
            parts.extend(
                redactor.redact_text(line) for line in traceback.format_tb(current.__traceback__)
            )
        message = safe_exception_message(current, redactor)
        name = type(current).__qualname__
        parts.append(f"{name}: {message}\n" if message else f"{name}\n")
    return "".join(parts).rstrip("\n")


def chain_echoes_input(error: BaseException) -> bool:
    """True when `error` or anything in its chain quotes its input."""
    return any(is_input_echoing(item) for item in _chain(error))


__all__ = [
    "chain_echoes_input",
    "describe_validation_error",
    "is_input_echoing",
    "safe_exception_message",
    "safe_traceback",
]

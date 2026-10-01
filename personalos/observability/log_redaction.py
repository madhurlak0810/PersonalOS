"""Log redaction: no credential reaches a log handler.

Call sites cannot be trusted to redact their own messages -- the one that
matters is the one nobody thought about, `logger.error(..., exc_info=True)`
around a provider call whose exception text includes the request headers. So
redaction is applied to the `LogRecord` itself, in two places:

- a **record factory**, which rewrites every record as it is created. This is
  process-wide and independent of which handlers exist, so it also covers
  handlers attached later and by other libraries;
- a **filter**, which does the same to a record and additionally covers
  `extra={...}` fields (which `logging` attaches *after* the factory has run).

Both format the message eagerly and drop `args` and `exc_info`, keeping the
redacted traceback as `exc_text`. That is deliberate: a handler that received
the live exception object could format it for itself and bypass all of this.
"""

import logging
import traceback

from personalos.domain.redaction import Redactor, default_redactor

#: Attributes every `LogRecord` has. Anything else on a record arrived through
#: `extra=` and is redacted as data.
_STANDARD_RECORD_ATTRIBUTES = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}

_FACTORY_MARKER = "_personalos_redacting"


def scrub_record(record: logging.LogRecord, redactor: Redactor | None = None) -> logging.LogRecord:
    """Redact one log record in place, and return it."""
    redactor = redactor or default_redactor()

    try:
        message = record.getMessage()
    except Exception:
        # A malformed format string must not become a reason to skip
        # redaction; fall back to the unformatted parts.
        message = f"{record.msg} {record.args!r}"
    record.msg = redactor.redact_text(message)
    record.args = None

    if record.exc_info:
        if not record.exc_text and isinstance(record.exc_info, tuple):
            record.exc_text = "".join(traceback.format_exception(*record.exc_info)).rstrip("\n")
        record.exc_info = None
    if record.exc_text:
        record.exc_text = redactor.redact_text(record.exc_text)
    if record.stack_info:
        record.stack_info = redactor.redact_text(record.stack_info)

    for name, value in list(record.__dict__.items()):
        if name not in _STANDARD_RECORD_ATTRIBUTES:
            record.__dict__[name] = redactor.redact(value)
    return record


class RedactingFilter(logging.Filter):
    """A logging filter that redacts every record and lets all of them through."""

    def __init__(self, redactor: Redactor | None = None):
        """Use the given redactor, or the process-wide one."""
        super().__init__()
        self._redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact the record; never drops it."""
        scrub_record(record, self._redactor)
        return True


def install_log_redaction(redactor: Redactor | None = None) -> None:
    """Redact every log record this process emits from now on.

    Idempotent. Installs the record factory, and adds a `RedactingFilter` to
    the handlers currently on the root logger so `extra` fields are covered as
    well.
    """
    current = logging.getLogRecordFactory()
    if not getattr(current, _FACTORY_MARKER, False):

        def factory(*args, **kwargs) -> logging.LogRecord:
            return scrub_record(current(*args, **kwargs), redactor)

        setattr(factory, _FACTORY_MARKER, True)
        logging.setLogRecordFactory(factory)

    for handler in logging.getLogger().handlers:
        if not any(isinstance(existing, RedactingFilter) for existing in handler.filters):
            handler.addFilter(RedactingFilter(redactor))


__all__ = ["scrub_record", "RedactingFilter", "install_log_redaction"]

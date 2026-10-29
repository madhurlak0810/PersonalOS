"""Schema-constrained model calls with a bounded repair step.

Every model boundary in this package binds a chat model to one Pydantic schema
and validates what comes back. A provider can still return something that will
not validate -- a field outside the schema, an enum value it made up, a
parser error. `invoke_structured` / `ainvoke_structured` answer that the same
way everywhere: ask again, telling the model which kind of error it made, up to
`max_attempts` times, and then raise the last error for the caller's own
fallback to handle. A provider error is not a schema error and is raised at
once; retrying the transport is the client's job, not this one's.

What the fallback *is* differs per boundary and is decided by the caller:
clarification for intent routing, keyword-level matches for job scoring, no
draft at all for artifact drafting.
"""

from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from personalos.observability.tracing import Attr, model_span, set_attributes

#: Errors that mean "the output was not the schema". `ValueError` covers
#: LangChain's `OutputParserException` and the domain's own contract errors.
SCHEMA_ERRORS: tuple[type[Exception], ...] = (ValidationError, ValueError, TypeError)

#: The first answer and one repair.
DEFAULT_MAX_ATTEMPTS = 2

REPAIR_PROMPT = (
    "Your previous answer did not match the required schema ({error}). "
    "Answer again with only the schema's fields."
)

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)

Turns = Sequence[tuple[str, str]]


def _coerce(schema: type[_SchemaT], raw: Any) -> _SchemaT:
    # Providers may hand back the decoded model or the plain dict behind it;
    # validating both ways means only a real instance leaves here.
    return raw if isinstance(raw, schema) else schema.model_validate(raw)


def repair_turns(turns: Turns, error: Exception) -> list[tuple[str, str]]:
    """The original system and human turns, plus the request to try again."""
    return [*turns[:2], ("human", REPAIR_PROMPT.format(error=type(error).__name__))]


def _check_attempts(max_attempts: int) -> None:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")


def invoke_structured(
    runnable: Any,
    turns: Turns,
    schema: type[_SchemaT],
    *,
    span: str,
    model: str | None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    **span_attributes: Any,
) -> _SchemaT:
    """Invoke `runnable` until its output validates as `schema`, or raise."""
    _check_attempts(max_attempts)
    current = list(turns)
    for attempt in range(1, max_attempts + 1):
        try:
            with model_span(span, model=model, attempt=attempt, **span_attributes) as attempt_span:
                try:
                    raw = runnable.invoke(current)
                    result = _coerce(schema, raw)
                except SCHEMA_ERRORS:
                    set_attributes(attempt_span, {Attr.SCHEMA_VALID: False})
                    raise
                set_attributes(attempt_span, {Attr.SCHEMA_VALID: True})
            return result
        except SCHEMA_ERRORS as exc:
            if attempt == max_attempts:
                raise
            current = repair_turns(turns, exc)
    raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises


async def ainvoke_structured(
    runnable: Any,
    turns: Turns,
    schema: type[_SchemaT],
    *,
    span: str,
    model: str | None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    **span_attributes: Any,
) -> _SchemaT:
    """`invoke_structured`, for a runnable called with `ainvoke`."""
    _check_attempts(max_attempts)
    current = list(turns)
    for attempt in range(1, max_attempts + 1):
        try:
            with model_span(span, model=model, attempt=attempt, **span_attributes) as attempt_span:
                try:
                    raw = await runnable.ainvoke(current)
                    result = _coerce(schema, raw)
                except SCHEMA_ERRORS:
                    set_attributes(attempt_span, {Attr.SCHEMA_VALID: False})
                    raise
                set_attributes(attempt_span, {Attr.SCHEMA_VALID: True})
            return result
        except SCHEMA_ERRORS as exc:
            if attempt == max_attempts:
                raise
            current = repair_turns(turns, exc)
    raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises


__all__ = [
    "SCHEMA_ERRORS",
    "DEFAULT_MAX_ATTEMPTS",
    "REPAIR_PROMPT",
    "repair_turns",
    "invoke_structured",
    "ainvoke_structured",
]

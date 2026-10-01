"""Redaction: strip credentials from anything about to be written down.

The primary defence against a leaked credential is structural -- a refresh
token lives in the OS keychain and is only ever handed to the credential
broker (see `personalos.domain.credentials`), so it has no route into graph
state in the first place. This module is the second line, for the routes
structure cannot close: a provider SDK that echoes the request headers into
its exception message, an HTTP error whose body repeats the token it rejected,
a redirect URL with an OAuth `code=` still on it. Those arrive as ordinary
strings inside ordinary errors, and from there flow into logs, traces,
checkpoints and audit rows like any other string.

So every sink that persists free-form data passes it through a `Redactor`
first. Three kinds of match, in decreasing order of certainty:

1. **Known values.** The broker registers every secret it reads or mints, and
   an exact occurrence of one is removed wherever it appears, whatever it is
   wrapped in. This is the only rule that cannot miss a credential this
   process actually handled.
2. **Sensitive keys.** In a mapping, the value under `Authorization`,
   `Cookie`, `refresh_token`, `client_secret` and the like is dropped without
   looking at it.
3. **Patterns.** Header lines, `key=value` pairs, OAuth codes and well-known
   token formats inside free text.

It lives in `domain` because it is pure -- regexes over values, no I/O -- and
because `domain` is the one layer every sink (`persistence`, `mcp`, `tools`,
`observability`) is allowed to import.
"""

import dataclasses
import re
import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

#: What a removed value is replaced with. A fixed marker rather than a hash or
#: a prefix: any function of the secret is a little of the secret.
REDACTED = "[REDACTED]"

#: A known secret shorter than this is not registered: scrubbing every
#: occurrence of a three-character string would shred unrelated text, and a
#: real token is never that short.
MIN_KNOWN_SECRET_LENGTH = 8

#: Mapping keys whose value is a credential by definition. Compared after
#: lower-casing and folding `-` to `_`, so `Set-Cookie` and `set_cookie` match.
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "refresh_token",
        "access_token",
        "id_token",
        "auth_token",
        "session_token",
        "bearer_token",
        "token",
        "client_secret",
        "api_key",
        "apikey",
        "x_api_key",
        "password",
        "passwd",
        "secret",
        "secret_key",
        "private_key",
        "oauth_code",
        "authorization_code",
    }
)

#: Keys that mark a mapping as an OAuth exchange. A bare `code` key is far too
#: common to redact on sight (error codes, status codes), so it is only treated
#: as an authorization code when it sits next to one of these.
_OAUTH_SIBLING_KEYS: frozenset[str] = frozenset(
    {"grant_type", "redirect_uri", "client_id", "code_verifier"}
)

_SCHEMES = r"(?:(?:bearer|basic|digest|token|oauth)\s+)?"


def _separator(before_value: str = "") -> str:
    """The `:`/`=` between a name and its value, refusing an already-redacted value.

    Without the guard a second pass would match `[REDACTED` and leave a stray
    `]` -- and second passes are routine: a tool error is redacted at the
    gateway and again on its way into a checkpoint. The guard sits directly
    after the `:`/`=` so the optional whitespace cannot backtrack around it.
    """
    return (
        r"""(["']?\s*[:=](?!\s*["']?"""
        + before_value
        + r"""\[REDACTED\])\s*["']?)"""
    )


_NAME_VALUE_SEPARATOR = _separator()

#: `(pattern, replacement)` applied in order to every string.
_TEXT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # A PEM private key, whole block.
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        REDACTED,
    ),
    # `Authorization: Bearer abc`, `'Authorization': 'Basic abc'`, `authorization=abc`.
    (
        re.compile(
            r"(?i)\b((?:proxy-)?authorization)"
            + _separator(_SCHEMES)
            + _SCHEMES
            + r"[^\s\"',;}\]]+"
        ),
        rf"\1\2{REDACTED}",
    ),
    # `Cookie: a=b; c=d` -- to the end of the line or the closing quote, since
    # a cookie header is itself a `;`-separated list.
    (
        re.compile(r"(?i)\b(set-cookie|cookie)" + _NAME_VALUE_SEPARATOR + r"[^\"'\r\n}]+"),
        rf"\1\2{REDACTED}",
    ),
    # A bearer token with no header name in front of it.
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), f"Bearer {REDACTED}"),
    # `refresh_token=...`, `"client_secret": "..."`, `GOOGLE_API_KEY=...`.
    (
        re.compile(
            r"(?i)(?<![A-Za-z0-9])"
            r"((?:refresh|access|id|auth|session|bearer)[_-]?token"
            r"|client[_-]?secret|(?:x[_-])?api[_-]?key|password|passwd"
            r"|secret[_-]?key|private[_-]?key)"
            + _NAME_VALUE_SEPARATOR
            + r"[^\s\"'&,;}\]]+"
        ),
        rf"\1\2{REDACTED}",
    ),
    # A bare `token=` / `secret=`. Anchored so `lease_token=` and
    # `idempotency-token=` -- identifiers, not credentials -- are left alone.
    (
        re.compile(
            r"(?i)(?<![A-Za-z0-9_-])(token|secret)"
            + _NAME_VALUE_SEPARATOR
            + r"[^\s\"'&,;}\]]{8,}"
        ),
        rf"\1\2{REDACTED}",
    ),
    # An OAuth authorization code on a redirect URL or in a form body.
    (
        re.compile(r"(?i)(?<![A-Za-z0-9_-])(code)(=)(?!\[REDACTED\])[^\s\"'&,;}\]]{8,}"),
        rf"\1\2{REDACTED}",
    ),
    # Well-known token shapes, wherever they turn up.
    (re.compile(r"\bya29\.[0-9A-Za-z._-]{10,}"), REDACTED),  # Google access token
    (re.compile(r"\b1//[0-9A-Za-z_-]{20,}"), REDACTED),  # Google refresh token
    (re.compile(r"\b4/[0-9A-Za-z_-]{20,}"), REDACTED),  # Google authorization code
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35,}"), REDACTED),  # Google API key
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]+"),
        REDACTED,
    ),  # JWT
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"), REDACTED),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), REDACTED),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),
)


def _normalize_key(key: Any) -> str | None:
    if not isinstance(key, str):
        return None
    return key.strip().lower().replace("-", "_")


class Redactor:
    """Removes credentials from strings and from structures that hold them.

    `redact` returns the *same object* when nothing in it needed removing, and
    a rebuilt copy when something did. The input is never mutated -- a caller
    redacting graph state on its way to a checkpoint must not change the state
    the running graph still holds.
    """

    def __init__(self) -> None:
        """Start with the pattern rules only; known values are registered later."""
        self._known: dict[str, datetime | None] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Known values
    # ------------------------------------------------------------------

    def register_secret(self, value: str, *, expires_at: datetime | None = None) -> None:
        """Remember a literal secret so every later occurrence of it is removed.

        `expires_at` lets a short-lived access token be forgotten once it can
        no longer be used, so a long-running worker does not accumulate every
        token it ever minted. A long-lived secret is registered without one.
        """
        if not isinstance(value, str) or len(value) < MIN_KNOWN_SECRET_LENGTH:
            return
        with self._lock:
            now = datetime.now(timezone.utc)
            self._known = {
                known: expiry
                for known, expiry in self._known.items()
                if expiry is None or expiry > now
            }
            current = self._known.get(value, expires_at)
            # Re-registering never shortens a lifetime: a value known to be
            # long-lived stays long-lived.
            self._known[value] = None if current is None or expires_at is None else expires_at

    def forget_secrets(self) -> None:
        """Drop every registered value. For tests; production never needs it."""
        with self._lock:
            self._known = {}

    # ------------------------------------------------------------------
    # Redaction
    # ------------------------------------------------------------------

    def redact_text(self, text: str) -> str:
        """Remove credentials from one string."""
        if not text:
            return text
        # Longest first, so a secret that contains another is removed whole.
        for known in sorted(self._known, key=len, reverse=True):
            if known in text:
                text = text.replace(known, REDACTED)
        for pattern, replacement in _TEXT_RULES:
            text = pattern.sub(replacement, text)
        return text

    def redact(self, value: Any) -> Any:
        """Remove credentials from a value, descending into containers."""
        return self._walk(value)[0]

    def contains_secret(self, value: Any) -> bool:
        """True when `redact` would change the value."""
        return self._walk(value)[1]

    def _walk(self, value: Any) -> tuple[Any, bool]:
        """Return `(redacted, changed)`; `redacted is value` when unchanged."""
        if value is None or isinstance(value, (bool, int, float, BaseException)):
            return value, False
        if isinstance(value, str):
            redacted = self.redact_text(value)
            return (redacted, True) if redacted != value else (value, False)
        if isinstance(value, (bytes, bytearray)):
            return self._walk_bytes(value)
        # Recognised by a marker attribute rather than by importing the type,
        # which keeps this module free of any dependency on `credentials`.
        if getattr(type(value), "__personalos_secret__", False):
            return REDACTED, True
        if isinstance(value, Mapping):
            return self._walk_mapping(value)
        if isinstance(value, (list, tuple, set, frozenset)):
            return self._walk_collection(value)
        return self._walk_object(value)

    def _walk_bytes(self, value: bytes | bytearray) -> tuple[Any, bool]:
        try:
            text = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return value, False
        redacted = self.redact_text(text)
        return (redacted.encode("utf-8"), True) if redacted != text else (value, False)

    def _walk_collection(self, value: Any) -> tuple[Any, bool]:
        items = [self._walk(item) for item in value]
        if not any(changed for _, changed in items):
            return value, False
        redacted = [item for item, _ in items]
        if isinstance(value, tuple) and hasattr(value, "_fields"):  # namedtuple
            return type(value)(*redacted), True
        if isinstance(value, list):
            return redacted, True
        return type(value)(redacted), True

    def _walk_mapping(self, value: Mapping[Any, Any]) -> tuple[Any, bool]:
        keys = {_normalize_key(key) for key in value}
        oauth_exchange = bool(keys & _OAUTH_SIBLING_KEYS)

        redacted: dict[Any, Any] = {}
        changed = False
        for key, item in value.items():
            normalized = _normalize_key(key)
            sensitive = normalized in SENSITIVE_KEYS or (oauth_exchange and normalized == "code")
            if sensitive and item is not None and item != "" and item != REDACTED:
                redacted[key] = REDACTED
                changed = True
                continue
            new_item, item_changed = self._walk(item)
            redacted[key] = new_item
            changed = changed or item_changed

        if not changed:
            return value, False
        if type(value) is dict:
            return redacted, True
        try:
            return type(value)(redacted), True
        except Exception:
            # A mapping type that cannot be rebuilt from a dict degrades to a
            # plain one: losing the subclass is better than keeping the secret.
            return redacted, True

    def _walk_object(self, value: Any) -> tuple[Any, bool]:
        """Redact the fields of a dataclass or pydantic model instance.

        Graph state is JSON by convention, but LangGraph's own values are not
        (an `Interrupt` carries the approval request it parked on), so objects
        with declared fields are followed rather than trusted. Anything else
        is opaque and passes through.
        """
        if isinstance(value, type):
            return value, False
        if dataclasses.is_dataclass(value):
            names = [f.name for f in dataclasses.fields(value) if f.init]
            return self._rebuild(value, names, lambda updates: dataclasses.replace(value, **updates))
        model_fields = getattr(type(value), "model_fields", None)
        if isinstance(model_fields, Mapping) and hasattr(value, "model_copy"):
            return self._rebuild(
                value, list(model_fields), lambda updates: value.model_copy(update=updates)
            )
        return value, False

    def _rebuild(
        self, value: Any, names: list[str], rebuild: Callable[[dict[str, Any]], Any]
    ) -> tuple[Any, bool]:
        updates: dict[str, Any] = {}
        for name in names:
            new_item, changed = self._field(name, getattr(value, name, None))
            if changed:
                updates[name] = new_item
        if not updates:
            return value, False
        try:
            return rebuild(updates), True
        except Exception:
            # An object that will not take the redacted field cannot be made
            # safe by this walker; the caller's serializer decides what happens
            # to it. Known secret *types* never get here (see `_walk`).
            return value, False

    def _field(self, name: str, current: Any) -> tuple[Any, bool]:
        if (
            _normalize_key(name) in SENSITIVE_KEYS
            and isinstance(current, str)
            and current
            and current != REDACTED
        ):
            return REDACTED, True
        return self._walk(current)


#: The process-wide redactor. Shared deliberately: the broker registers a
#: secret on it once, and every sink -- logging, tracing, the checkpointer, the
#: repositories -- then removes that value without being told about it.
_DEFAULT_REDACTOR = Redactor()


def default_redactor() -> Redactor:
    """The process-wide `Redactor` every sink uses unless handed another."""
    return _DEFAULT_REDACTOR


def redact(value: Any) -> Any:
    """Remove credentials from a value using the process-wide redactor."""
    return _DEFAULT_REDACTOR.redact(value)


def redact_text(text: str) -> str:
    """Remove credentials from a string using the process-wide redactor."""
    return _DEFAULT_REDACTOR.redact_text(text)


__all__ = [
    "REDACTED",
    "MIN_KNOWN_SECRET_LENGTH",
    "SENSITIVE_KEYS",
    "Redactor",
    "default_redactor",
    "redact",
    "redact_text",
]

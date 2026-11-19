"""Path sandbox for the files MCP server.

A path reaching a file tool is model-authored text. Nothing about it is
trusted: not that it is relative, not that it stays where its prefix says it
does, not that the thing at the end of it is a file. `PathSandbox.resolve` is
the one place that turns such a string into a path the server will touch, and
it decides on the *canonical* path -- after `..` is collapsed and every symlink
is followed -- never on how the string looks.

Three checks, in this order:

1. The requested path is anchored server-side. A relative path is joined onto
   the primary allowed root, never onto the process's working directory.
2. The canonical path must sit inside a canonical allowed root.
3. No component below that root may be hidden or name a credential/config
   directory (`.ssh`, a browser profile, a cloud CLI's config). This holds
   even inside an allowed root, and is checked on the path as requested as
   well as on what it resolves to, so neither a symlink into such a directory
   nor a symlink out of one gets through.
"""

import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from personalos.domain.errors import PolicyDeniedError, ValidationFailed
from personalos.domain.redaction import redact_text

logger = logging.getLogger(__name__)


class PathNotAllowed(PolicyDeniedError):
    """The path resolves outside the allowed roots, or into a denied directory."""

    default_message = "path is not allowed"


class StaleWrite(ValidationFailed):
    """The file changed since the caller read it; writing would lose an update."""

    http_status = 409
    default_message = "file changed since it was read"


#: Directory and file names refused anywhere below an allowed root, compared
#: case-insensitively. Dot-prefixed names are refused wholesale by
#: `PathSandbox`; they are listed here as well because this set is also what an
#: allowed *root* is checked against, where the blanket dot rule does not apply.
DEFAULT_DENIED_NAMES: frozenset[str] = frozenset(
    {
        # SSH / GPG / generic credential files
        ".ssh",
        ".gnupg",
        ".password-store",
        ".netrc",
        ".env",
        ".git-credentials",
        "keychains",
        "keyrings",
        # Cloud and cluster credentials
        ".aws",
        ".azure",
        ".gcloud",
        "gcloud",
        ".kube",
        ".docker",
        ".oci",
        # Config directories
        ".config",
        ".local",
        ".git",
        "appdata",
        # Browser profiles
        ".mozilla",
        "google-chrome",
        "chromium",
        "bravesoftware",
        "microsoft edge",
        "microsoft-edge",
        "user data",
    }
)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class PathSandbox:
    """Resolves untrusted paths against a fixed set of allowed roots."""

    def __init__(
        self,
        allowed_roots: Iterable[str | os.PathLike[str]],
        denied_names: Iterable[str] = DEFAULT_DENIED_NAMES,
    ):
        """Canonicalize the roots once, at construction.

        With no roots the sandbox is valid and rejects every path: an
        unconfigured deployment has no file surface, rather than a default one.

        Raises `ValueError` for a root that is relative, is the filesystem
        root, or is itself a credential/config directory -- a misconfiguration
        that should stop startup, not surface as a denied call later.
        """
        self.denied_names = frozenset(name.casefold() for name in denied_names)
        roots: list[Path] = []
        for raw in allowed_roots:
            root = Path(raw)
            if not root.is_absolute():
                raise ValueError(f"allowed root must be an absolute path: {raw!r}")
            canonical = Path(os.path.realpath(root))
            if canonical == Path(canonical.anchor):
                raise ValueError(f"allowed root must not be the filesystem root: {raw!r}")
            denied = [part for part in canonical.parts if part.casefold() in self.denied_names]
            if denied:
                raise ValueError(
                    f"allowed root {raw!r} is inside a denied directory ({denied[0]!r})"
                )
            if canonical not in roots:
                roots.append(canonical)
        self.roots: tuple[Path, ...] = tuple(roots)

    def _is_denied(self, part: str) -> bool:
        return part.startswith(".") or part.casefold() in self.denied_names

    def _root_of(self, path: Path) -> Path | None:
        for root in self.roots:
            if _is_within(path, root):
                return root
        return None

    def resolve(self, requested: str, *, allow_root: bool = False) -> Path:
        """Return the canonical path for `requested`, or raise `PathNotAllowed`.

        The target does not have to exist (a write may be creating it), but
        every symlink on the way to it is followed, including a dangling one.
        `allow_root` admits an allowed root itself, for a directory listing.
        """
        if not isinstance(requested, str) or not requested.strip():
            raise PathNotAllowed("path must be a non-empty string")
        if "\x00" in requested:
            raise PathNotAllowed("path contains a NUL byte")
        if not self.roots:
            raise PathNotAllowed("no allowed roots are configured for file tools")

        path = Path(requested)
        anchored = path if path.is_absolute() else self.roots[0] / path

        # The path as written, with `..` collapsed but symlinks left alone:
        # this is what catches `<root>/.ssh/link-to-somewhere-harmless`.
        lexical = Path(os.path.normpath(anchored))
        lexical_root = self._root_of(lexical)
        if lexical_root is not None:
            self._refuse_denied_parts(requested, lexical.relative_to(lexical_root).parts)

        canonical = Path(os.path.realpath(anchored))
        root = self._root_of(canonical)
        if root is None:
            raise PathNotAllowed(f"path {requested!r} resolves outside the allowed roots")
        self._refuse_denied_parts(requested, canonical.relative_to(root).parts)
        if canonical == root and not allow_root:
            raise PathNotAllowed(f"path {requested!r} is an allowed root, not a file")
        return canonical

    def is_denied_name(self, name: str) -> bool:
        """True for a directory or file name no listing may descend into or report."""
        return self._is_denied(name)

    def _refuse_denied_parts(self, requested: str, parts: tuple[str, ...]) -> None:
        for part in parts:
            if self._is_denied(part):
                raise PathNotAllowed(f"path {requested!r} is inside a denied directory ({part!r})")


@dataclass(frozen=True)
class PathDenial:
    """One path the sandbox refused, as the audit trail records it.

    `requested_path` is what the caller sent, redacted: it is model-authored
    text and is about to be written to a log and an audit row.
    """

    tool: str
    requested_path: str
    reason: str
    actor_id: str = "system"
    workflow_id: UUID | None = None
    correlation_id: UUID | None = None


#: Receives every `PathDenial`. Must not raise; the call is refused either way.
DenialSink = Callable[[PathDenial], None]


def log_path_denial(denial: PathDenial) -> None:
    """The default sink: one structured WARNING per refused path."""
    logger.warning(
        "files.%s refused path %r for %s: %s",
        denial.tool,
        denial.requested_path,
        denial.actor_id,
        denial.reason,
        extra={
            key: value
            for key, value in {
                "event": "files.path_denied",
                "tool_name": f"files.{denial.tool}",
                "actor_id": denial.actor_id,
                "workflow_id": denial.workflow_id,
                "correlation_id": denial.correlation_id,
            }.items()
            if value is not None
        },
    )


def path_denial(
    tool: str,
    requested: object,
    reason: str,
    *,
    actor_id: str = "system",
    workflow_id: UUID | None = None,
    correlation_id: UUID | None = None,
) -> PathDenial:
    """Build a `PathDenial`, redacting what the caller sent."""
    return PathDenial(
        tool=tool,
        requested_path=redact_text(str(requested))[:500],
        reason=redact_text(reason)[:500],
        actor_id=actor_id,
        workflow_id=workflow_id,
        correlation_id=correlation_id,
    )


__all__ = [
    "DEFAULT_DENIED_NAMES",
    "DenialSink",
    "PathDenial",
    "PathNotAllowed",
    "PathSandbox",
    "StaleWrite",
    "log_path_denial",
    "path_denial",
]

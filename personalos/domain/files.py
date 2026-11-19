"""The File domain: typed plans over job-search documents, and what came of them.

The File subgraph (`personalos.graphs.file_task`) finds, reads, summarizes and
organizes resumes, cover letters and other application documents. Everything
it is allowed to do is written down here, before it does any of it, as a
`FilePlan`: a closed list of `FileStep`s, each naming its operation and the
exact paths it touches. A plan is the only thing the executor runs, and it is
fixed before a single document is opened -- so nothing written *inside* a
document can add a step, change a path, or reach a tool.

Three properties the rest of the subgraph leans on:

**A plan is validated twice, against a JSON schema and against the world.**
`parse_file_plan` checks a raw plan against `FILE_PLAN_JSON_SCHEMA` (the
schema a model is constrained to) and then rebuilds it as a `FilePlan`;
`check_plan_against_inventory` then checks it names only files discovery
actually saw, and writes only where discovery's roots are.

**Approval is decided from the plan, not from the caller.**
`FileApprovalPolicy.reasons` returns why a plan needs a person -- any delete,
any write over an existing file, or more files touched than the configured
limit -- and the approval request lists every affected path.

**Every mutation is idempotent by key and reconcilable by hash.** Each
mutating step carries `step_idempotency_key(plan, step)`, derived from the
plan's hash so a rerun of the same plan reuses it, and the content hash it
expects to find. A rerun after a crash compares the file on disk with the
step's goal state (`FileReceipt.outcome == RECONCILED`) instead of writing
again.

Pure: no I/O, no clocks beyond defaults, nothing internal imported but
`domain`.
"""

import hashlib
import json
import posixpath
import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from personalos.domain.errors import ValidationFailed
from personalos.domain.job_search import ActionIntent, ApprovalDecision
from personalos.domain.models import validate_idempotency_key

#: The most steps one plan may hold. A plan is a bounded unit of work, not a
#: script; anything larger is several requests.
MAX_PLAN_STEPS = 50

#: The most files a plan may touch with mutating steps before a person has to
#: sign it off, when the deployment does not configure its own limit.
DEFAULT_MAX_FILES_WITHOUT_APPROVAL = 3

#: The most entries discovery records. More than this many documents is not a
#: job search's worth of material, and the inventory is part of a checkpoint.
MAX_INVENTORY_ENTRIES = 500

#: Longest highlight a summary may carry, and how many it may carry.
MAX_HIGHLIGHT_CHARS = 300
MAX_HIGHLIGHTS = 8

#: Extensions of documents the File subgraph treats as job-search material.
DOCUMENT_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".txt", ".pdf", ".docx", ".doc", ".rtf", ".odt"}
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FilePlanInvalid(ValidationFailed, ValueError):
    """A plan failed its schema or does not fit the files it names."""

    default_message = "file plan is invalid"


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    ).hexdigest()


# --- Inventory -----------------------------------------------------------------


class FileCategory(str, Enum):
    """What kind of job-search document a file is, as read off its name."""

    RESUME = "resume"
    COVER_LETTER = "cover_letter"
    APPLICATION = "application"
    OTHER = "other"


_CATEGORY_PATTERNS: tuple[tuple[FileCategory, re.Pattern[str]], ...] = (
    (FileCategory.COVER_LETTER, re.compile(r"cover[\s_\-.]*letter|\bcl\b|motivation")),
    (FileCategory.RESUME, re.compile(r"resume|résumé|\bcv\b|curriculum")),
    (
        FileCategory.APPLICATION,
        re.compile(r"application|offer|reference|transcript|portfolio|writing[\s_\-]*sample"),
    ),
)


def categorize(name: str) -> FileCategory:
    """The category a file's *name* suggests. Never its content: content is data."""
    lowered = re.sub(r"[_\-.]+", " ", name.casefold())
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(lowered):
            return category
    return FileCategory.OTHER


def is_document(name: str) -> bool:
    """True for a file with a document extension; hidden and temp files never are."""
    if name.startswith(".") or name.startswith("~$"):
        return False
    return posixpath.splitext(name.casefold())[1] in DOCUMENT_EXTENSIONS


class FileEntry(_Value):
    """One file discovery saw: where it is and what version it is. Never its content."""

    path: str
    name: str
    size: int = Field(ge=0)
    modified_at: datetime
    sha256: str
    category: FileCategory = FileCategory.OTHER

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("sha256 must be a lowercase hex SHA-256 digest")
        return value


class FileInventory(_Value):
    """What discovery found: the roots it searched and the documents in them."""

    roots: tuple[str, ...] = ()
    entries: tuple[FileEntry, ...] = ()
    #: True when more documents existed than `MAX_INVENTORY_ENTRIES` allowed.
    truncated: bool = False

    def by_path(self) -> dict[str, FileEntry]:
        return {entry.path: entry for entry in self.entries}

    def latest(self, category: FileCategory | None = None) -> FileEntry | None:
        """The most recently modified entry, optionally of one category."""
        candidates = [e for e in self.entries if category is None or e.category == category]
        if not candidates:
            return None
        return max(candidates, key=lambda e: (e.modified_at, e.path))


class FileStat(_Value):
    """Whether a file exists, and which version it is. No content."""

    path: str
    exists: bool
    size: int | None = None
    sha256: str | None = None
    modified_at: datetime | None = None


class FileContent(_Value):
    """What a read returned: the document's text, if it has any, and its version.

    `text` is untrusted data written by whoever wrote the document. It is
    already redacted of credentials by the tool gateway, and it is never
    checkpointed whole: nodes keep its hash, a summary and a short excerpt.
    """

    path: str
    sha256: str
    size: int = Field(ge=0)
    text: str | None = None


# --- Plans ---------------------------------------------------------------------


class FileOp(str, Enum):
    """The closed set of operations a plan step may name."""

    #: Report a file's location and metadata. Reads nothing.
    LOCATE = "locate"
    #: Read a file and record its hash and a short, redacted excerpt.
    READ = "read"
    #: Read a file and summarize it, as data.
    SUMMARIZE = "summarize"
    #: Create a file, or replace one discovery saw (an overwrite: needs approval).
    WRITE = "write"
    #: Move or rename a file. Reversible, with an undo record.
    MOVE = "move"
    #: Delete a file. Destructive: always needs approval.
    DELETE = "delete"


#: Operations that change the filesystem.
MUTATING_OPS: frozenset[FileOp] = frozenset({FileOp.WRITE, FileOp.MOVE, FileOp.DELETE})

#: Operations whose `path` must be a file discovery saw.
_EXISTING_SOURCE_OPS: frozenset[FileOp] = frozenset(
    {FileOp.LOCATE, FileOp.READ, FileOp.SUMMARIZE, FileOp.MOVE, FileOp.DELETE}
)


class FileStep(_Value):
    """One operation in a plan, on explicit paths."""

    step_id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_\-]+$")
    op: FileOp
    #: The file the step acts on. For a write, the file it creates or replaces.
    path: str = Field(min_length=1, max_length=1024)
    #: Where a move puts the file. Required for, and only for, `MOVE`.
    destination: str | None = Field(default=None, max_length=1024)
    #: What a write puts in the file. Required for, and only for, `WRITE`.
    content: str | None = Field(default=None, max_length=200_000)
    #: Why the planner chose this step, for the reviewer. Not acted on.
    reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _fields_match_op(self) -> "FileStep":
        if self.op == FileOp.MOVE and not self.destination:
            raise ValueError("a move step must name its destination")
        if self.op != FileOp.MOVE and self.destination is not None:
            raise ValueError(f"a {self.op.value} step must not name a destination")
        if self.op == FileOp.WRITE and self.content is None:
            raise ValueError("a write step must carry the content it writes")
        if self.op != FileOp.WRITE and self.content is not None:
            raise ValueError(f"a {self.op.value} step must not carry content")
        return self

    @property
    def mutating(self) -> bool:
        return self.op in MUTATING_OPS

    def touched_paths(self) -> tuple[str, ...]:
        """Every path this step changes, in order. Empty for a read."""
        if not self.mutating:
            return ()
        return (self.path, self.destination) if self.destination else (self.path,)

    def content_sha256(self) -> str | None:
        """SHA-256 of what a write puts on disk: the hash its goal state has."""
        if self.content is None:
            return None
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class FilePlan(_Value):
    """A bounded, typed plan: what the File subgraph will do, path by path."""

    goal: str = Field(min_length=1, max_length=200)
    steps: tuple[FileStep, ...] = Field(min_length=1, max_length=MAX_PLAN_STEPS)

    @model_validator(mode="after")
    def _step_ids_unique(self) -> "FilePlan":
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step ids must be unique within a plan")
        return self

    def plan_hash(self) -> str:
        """Stable hash of everything the plan would do. What an approval is bound to."""
        return _canonical_hash(self.model_dump(mode="json"))

    @property
    def mutating_steps(self) -> tuple[FileStep, ...]:
        return tuple(step for step in self.steps if step.mutating)

    def affected_paths(self) -> tuple[str, ...]:
        """Every path a mutating step changes, deduplicated, in plan order."""
        seen: dict[str, None] = {}
        for step in self.mutating_steps:
            for path in step.touched_paths():
                seen.setdefault(path, None)
        return tuple(seen)

    def changed_files(self) -> tuple[str, ...]:
        """The files mutating steps act on (a move's source, not its destination).

        What the unattended limit counts: reorganizing three files is three
        files, however many paths the moves name.
        """
        return tuple(dict.fromkeys(step.path for step in self.mutating_steps))

    @property
    def has_destructive_step(self) -> bool:
        return any(step.op == FileOp.DELETE for step in self.steps)


#: The JSON schema a plan is validated against, and a model is constrained to.
FILE_PLAN_JSON_SCHEMA: dict[str, Any] = FilePlan.model_json_schema()


def parse_file_plan(raw: Any) -> FilePlan:
    """Validate a raw plan against `FILE_PLAN_JSON_SCHEMA`, then build the typed plan.

    The JSON-schema pass is what a model was constrained to, checked rather
    than trusted; the typed pass adds what a schema cannot say (a move has a
    destination, ids are unique). Raises `FilePlanInvalid` either way.
    """
    if isinstance(raw, FilePlan):
        raw = raw.model_dump(mode="json")
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema ships with the mcp dependency
        jsonschema = None
    if jsonschema is not None:
        try:
            jsonschema.validate(raw, FILE_PLAN_JSON_SCHEMA)
        except jsonschema.ValidationError as exc:
            raise FilePlanInvalid(f"plan does not match its schema: {exc.message}") from exc
    try:
        return FilePlan.model_validate(raw)
    except ValueError as exc:
        raise FilePlanInvalid(f"plan is not a valid FilePlan: {exc}") from exc


def _has_unsafe_parts(path: str) -> str | None:
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part == "..":
            return "'..' segments are not allowed"
        if part.startswith(".") and part not in ("", "."):
            return f"hidden component {part!r} is not allowed"
    if "\x00" in path:
        return "NUL bytes are not allowed"
    return None


def _within_roots(path: str, roots: tuple[str, ...]) -> bool:
    normalized = posixpath.normpath(path)
    return any(
        normalized == posixpath.normpath(root)
        or normalized.startswith(posixpath.normpath(root).rstrip("/") + "/")
        for root in roots
    )


def check_plan_against_inventory(plan: FilePlan, inventory: FileInventory) -> FilePlan:
    """Check the plan names only files discovery saw and writes only inside its roots.

    A lexical pre-check: the files server re-resolves every path canonically,
    following symlinks, and refuses anything outside its roots whatever this
    says. Its job is to reject a bad plan *before* a reviewer is asked about
    it, not to be the sandbox.
    """
    known = inventory.by_path()
    touched: set[str] = set()
    for step in plan.steps:
        _check_step_paths(step)
        _check_step_targets(step, known, inventory.roots)
        for path in step.touched_paths():
            if path in touched:
                raise FilePlanInvalid(
                    f"step {step.step_id!r} touches {path!r}, which an earlier step already "
                    f"changes; one change per file per plan"
                )
            touched.add(path)
    return plan


def _check_step_paths(step: FileStep) -> None:
    """Every path a step names is absolute, with no `..` and no hidden component."""
    for path in (step.path, step.destination):
        if path is None:
            continue
        problem = _has_unsafe_parts(path)
        if problem:
            raise FilePlanInvalid(f"step {step.step_id!r}: {path!r}: {problem}")
        if not path.startswith("/"):
            raise FilePlanInvalid(f"step {step.step_id!r}: {path!r} must be an absolute path")


def _check_step_targets(
    step: FileStep, known: dict[str, FileEntry], roots: tuple[str, ...]
) -> None:
    """A step acts on a discovered file, and writes or moves only inside the roots."""
    if step.op in _EXISTING_SOURCE_OPS and step.path not in known:
        raise FilePlanInvalid(
            f"step {step.step_id!r} names {step.path!r}, which discovery did not find"
        )
    if step.op == FileOp.WRITE and step.path not in known and not _within_roots(step.path, roots):
        raise FilePlanInvalid(
            f"step {step.step_id!r} writes {step.path!r}, outside the searched roots"
        )
    if step.op == FileOp.MOVE:
        destination = step.destination or ""
        if not _within_roots(destination, roots):
            raise FilePlanInvalid(
                f"step {step.step_id!r} moves to {destination!r}, outside the searched roots"
            )
        if destination in known:
            raise FilePlanInvalid(
                f"step {step.step_id!r} would move onto {destination!r}, which exists"
            )


def step_idempotency_key(plan: FilePlan, step: FileStep) -> str:
    """The idempotency key a mutating step is sent with: stable across reruns of the plan."""
    return validate_idempotency_key(f"file-{plan.plan_hash()[:24]}-{step.step_id}")


def expected_sha256(step: FileStep, inventory: FileInventory) -> str | None:
    """The hash the file a step acts on must still have: the version discovery saw."""
    entry = inventory.by_path().get(step.path)
    return entry.sha256 if entry else None


# --- Approval ------------------------------------------------------------------


class FileApprovalPolicy(_Value):
    """When a plan needs a person to sign it off before anything runs."""

    max_files_without_approval: int = Field(default=DEFAULT_MAX_FILES_WITHOUT_APPROVAL, ge=0)

    def reasons(self, plan: FilePlan, inventory: FileInventory) -> tuple[str, ...]:
        """Why this plan needs approval; empty when it can run unattended."""
        reasons: list[str] = []
        deletes = [step.path for step in plan.steps if step.op == FileOp.DELETE]
        if deletes:
            reasons.append(f"deletes {len(deletes)} file(s)")
        known = inventory.by_path()
        overwrites = [s.path for s in plan.steps if s.op == FileOp.WRITE and s.path in known]
        if overwrites:
            reasons.append(f"replaces {len(overwrites)} existing file(s)")
        changed = plan.changed_files()
        if len(changed) > self.max_files_without_approval:
            reasons.append(
                f"changes {len(changed)} files, more than the "
                f"{self.max_files_without_approval} allowed without approval"
            )
        return tuple(reasons)


class PlanApproval(_Value):
    """A person's approval of a plan, handed to the executor with each call it covers.

    Built only by the File subgraph's apply node, after `authorize_execution`
    and the plan-hash check have both passed. The executor still checks that
    `decision` authorizes `action`, and that a person made it, before minting
    a policy grant from it.
    """

    action: ActionIntent
    decision: ApprovalDecision

    @property
    def approved_by(self) -> str:
        return self.decision.decided_by


def describe_step(step: FileStep) -> str:
    """One line a reviewer reads: the operation and its exact paths."""
    if step.op == FileOp.MOVE:
        return f"MOVE {step.path} -> {step.destination}"
    if step.op == FileOp.WRITE:
        return f"WRITE {step.path} ({len((step.content or '').encode('utf-8'))} bytes)"
    return f"{step.op.value.upper()} {step.path}"


# --- Outcomes ------------------------------------------------------------------


class StepOutcome(str, Enum):
    """What happened to one step."""

    #: The step ran and did what it said.
    APPLIED = "applied"
    #: The file was already in the step's goal state -- a rerun after a crash --
    #: so nothing was written again.
    RECONCILED = "reconciled"
    #: Not run: the plan was not approved, or an earlier step failed.
    SKIPPED = "skipped"
    #: Refused by policy or the sandbox.
    DENIED = "denied"
    #: Tried, and failed, or the file is in neither the old nor the new state.
    FAILED = "failed"


class UndoRecord(_Value):
    """How to take a reversible step back: the inverse move, and the version it expects."""

    op: FileOp = FileOp.MOVE
    path: str
    destination: str
    expected_sha256: str


class DocumentSummary(_Value):
    """A document, summarized as data.

    Deliberately has no field for instructions, tools or next steps: whatever
    a document says, the most a summary can do is quote it.
    """

    path: str
    title: str = Field(max_length=200)
    category: FileCategory = FileCategory.OTHER
    highlights: tuple[str, ...] = Field(default=(), max_length=MAX_HIGHLIGHTS)
    word_count: int = Field(default=0, ge=0)
    #: True when the document contains text that reads like instructions to an
    #: assistant. Reported to the person; it changes nothing the run does.
    contains_instructions: bool = False

    @field_validator("highlights")
    @classmethod
    def _bounded(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item[:MAX_HIGHLIGHT_CHARS] for item in value)


class FileReceipt(_Value):
    """What one step did, or why it did not."""

    step_id: str
    op: FileOp
    path: str
    destination: str | None = None
    outcome: StepOutcome
    #: The file's hash after the step (for a delete, the hash it had).
    sha256: str | None = None
    previous_sha256: str | None = None
    idempotency_key: str | None = None
    undo: UndoRecord | None = None
    entry: FileEntry | None = None
    summary: DocumentSummary | None = None
    #: A short, redacted excerpt of what a read step read.
    excerpt: str | None = Field(default=None, max_length=2000)
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome in (StepOutcome.APPLIED, StepOutcome.RECONCILED)


class FileVerification(_Value):
    """The verifier's check of one mutating step against the disk."""

    step_id: str
    ok: bool
    detail: str


#: Phrases that mark text as an attempt to instruct an assistant. Detection
#: only, for the person's information: nothing in the run branches on it.
_INSTRUCTION_PATTERNS = re.compile(
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)"
    r"|disregard (the |all )?(previous|prior|above)"
    r"|you are now|system prompt|new instructions"
    r"|(call|use|invoke) (the )?[a-z_]+\.[a-z_]+"
    r"|delete_file|move_file|write_file|send_message|tool call",
    re.IGNORECASE,
)


def looks_like_instructions(text: str) -> bool:
    """True when `text` contains phrasing aimed at an assistant rather than a reader."""
    return bool(_INSTRUCTION_PATTERNS.search(text or ""))


__all__ = [
    "DEFAULT_MAX_FILES_WITHOUT_APPROVAL",
    "DOCUMENT_EXTENSIONS",
    "FILE_PLAN_JSON_SCHEMA",
    "MAX_HIGHLIGHTS",
    "MAX_HIGHLIGHT_CHARS",
    "MAX_INVENTORY_ENTRIES",
    "MAX_PLAN_STEPS",
    "MUTATING_OPS",
    "PlanApproval",
    "DocumentSummary",
    "FileApprovalPolicy",
    "FileCategory",
    "FileContent",
    "FileEntry",
    "FileInventory",
    "FileOp",
    "FilePlan",
    "FilePlanInvalid",
    "FileReceipt",
    "FileStat",
    "FileStep",
    "FileVerification",
    "StepOutcome",
    "UndoRecord",
    "categorize",
    "check_plan_against_inventory",
    "describe_step",
    "expected_sha256",
    "is_document",
    "looks_like_instructions",
    "parse_file_plan",
    "step_idempotency_key",
]

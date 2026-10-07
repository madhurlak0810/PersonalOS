"""Tailored application artifacts, and the rule that keeps them honest.

A tailored resume or cover letter starts as a `DraftProposal`: model output,
and therefore a set of claims about the candidate rather than facts. Nothing
stores or sends a proposal. It becomes an `ArtifactDraft` only through
`finalize_draft`, which runs `validate_draft` first and raises
`UntraceableClaim` when anything in it cannot be traced to the evidence the
draft was written from.

The integrity rule is: **tailoring may rephrase supported experience, and may
not invent it.** A draft is a sequence of segments, each citing the evidence
records it restates. `validate_draft` flags a segment when

- it cites an `evidence_id` that is not among the records it was given;
- it states a checkable fact -- a number or date, a proper noun (employer,
  tool, degree, project), or a term from the posting's own skill list -- that
  does not appear in the records it cites; or
- it states such a fact and cites nothing at all.

The third case is what "substantive" means here: a greeting or a sentence of
interest in the role carries no checkable fact and needs no citation; anything
that names an employer, a metric, a tool or a date does.

The check is lexical on purpose. It is deterministic, needs no second model
to audit the first, and fails closed on exactly the inventions the rule names.
It has known limits, and they are limits in the permissive direction a
reviewer should know about: a capitalized word that *starts* a sentence is not
treated as a proper noun (it cannot be told apart from an ordinary verb), a
number is matched without its unit, and a claim with no checkable fact in it
("led a large team") passes uncited. The approval a submission still needs is
what stands behind those.

Posting text is data here as everywhere else. The posting's skill list is used
only as vocabulary to *look for* in a draft -- naming a required tool the
evidence does not mention is the commonest way a tailored resume lies -- and
never as support for a claim.
"""

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from enum import Enum

from pydantic import Field, field_validator

from personalos.domain.job_search import (
    PAYLOAD_BODY,
    PAYLOAD_PREVIOUS_BODY,
    PAYLOAD_RECIPIENT,
    ActionIntent,
    ActionKind,
    ArtifactDraft,
    EvidenceRecord,
    EvidenceRef,
    JobSearchContractError,
    _Value,
    strip_unsafe_chars,
)
from personalos.domain.models import ArtifactType

#: Longest evidence excerpt copied onto an `artifact_versions.evidence` entry.
EXCERPT_CHARS = 240

#: Credential words that are claims wherever they appear, in any case: a draft
#: may not award the candidate a degree its evidence does not mention.
DEGREE_TERMS: frozenset[str] = frozenset(
    {
        "bachelor",
        "bachelors",
        "master",
        "masters",
        "phd",
        "doctorate",
        "mba",
        "degree",
        "diploma",
        "certified",
        "certification",
        "certificate",
        "licensed",
    }
)

#: Capitalized words that are form, not fact: salutations and sign-offs.
_NEUTRAL_TERMS: frozenset[str] = frozenset(
    {
        "i",
        "i'm",
        "i've",
        "i'd",
        "i'll",
        "dear",
        "hiring",
        "manager",
        "team",
        "sincerely",
        "regards",
        "best",
        "thank",
        "thanks",
    }
)

#: A word, keeping the joiners that are part of a tool's name (`C++`, `C#`,
#: `Node.js`). `/` and `-` split, so `CI/CD` and `Python-based` are checked a
#: part at a time.
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.+#'][A-Za-z0-9+#]+)*[+#]*")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

#: Characters skipped when deciding whether a word starts a sentence.
_OPENERS = " \t\"'([-*•–—“‘"
_SENTENCE_ENDS = ".!?:;\n"


class DraftSegment(_Value):
    """One sentence, bullet or paragraph of a draft, and the records it restates.

    `evidence_ids` is empty for framing text. It is optional rather than
    required so a model that cites nothing still returns something parseable;
    `validate_draft` decides whether the omission matters.
    """

    text: str
    evidence_ids: tuple[str, ...] = ()


class DraftProposal(_Value):
    """A tailored document as a model wrote it, before anything has checked it.

    The untrusted counterpart of `ArtifactDraft`, in the way `ClaimedMatch` is
    of `MatchedRequirement`.
    """

    artifact_type: ArtifactType
    segments: tuple[DraftSegment, ...] = ()


class RetrievedEvidence(_Value):
    """One evidence record selected for a posting, with how close it scored."""

    record: EvidenceRecord
    similarity: float


class ClaimFlagReason(str, Enum):
    """Why a segment of a draft could not be traced to the candidate's record."""

    #: States a checkable fact and cites no evidence at all.
    UNCITED = "uncited_claim"
    #: Cites an evidence id that is not one of the records it was given.
    UNKNOWN_EVIDENCE = "unknown_evidence_id"
    #: States a fact the records it cites do not contain.
    UNSUPPORTED_FACT = "unsupported_fact"
    #: The draft as a whole cites nothing.
    NO_EVIDENCE = "draft_cites_no_evidence"


class ClaimFlag(_Value):
    """One untraceable claim: where it is, why, and the terms that gave it away."""

    reason: ClaimFlagReason
    segment_index: int | None = None
    text: str = ""
    terms: tuple[str, ...] = ()


class DraftValidation(_Value):
    """The validator's verdict on one proposal."""

    artifact_type: ArtifactType
    flags: tuple[ClaimFlag, ...] = ()
    #: Evidence ids the draft cites and that resolved, in first-cited order.
    cited_evidence_ids: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when every substantive claim traced to an evidence record."""
        return not self.flags


class UntraceableClaim(JobSearchContractError):
    """A draft states something its evidence does not support, and was rejected."""

    def __init__(self, validation: DraftValidation):
        self.validation = validation
        flagged = "; ".join(
            f"{flag.reason.value}"
            + (f" {list(flag.terms)}" if flag.terms else "")
            + (f" in segment {flag.segment_index}" if flag.segment_index is not None else "")
            for flag in validation.flags
        )
        super().__init__(
            f"{validation.artifact_type.value} draft rejected: {len(validation.flags)} claim(s) "
            f"not traceable to evidence ({flagged})",
            details={"flags": [flag.model_dump(mode="json") for flag in validation.flags]},
        )


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", strip_unsafe_chars(text)).replace("’", "'")


def _word(token: str) -> str:
    """A token as it is compared: case-folded, without a possessive or trailing dot."""
    word = token.casefold().rstrip(".")
    return word[:-2] if word.endswith("'s") else word


def _words(text: str) -> list[str]:
    return [_word(match.group()) for match in _TOKEN.finditer(_fold(text))]


def _numbers(text: str) -> set[str]:
    return {match.group().replace(",", "") for match in _NUMBER.finditer(_fold(text))}


def _starts_sentence(text: str, start: int) -> bool:
    before = text[:start].rstrip(_OPENERS)
    return not before or before[-1] in _SENTENCE_ENDS


class _Support:
    """What a set of texts can vouch for: its words, its numbers, its phrases."""

    def __init__(self, texts: Iterable[str]):
        words: list[str] = []
        self.numbers: set[str] = set()
        for text in texts:
            words.extend(_words(text))
            self.numbers |= _numbers(text)
        self.words = set(words)
        self._joined = f" {' '.join(words)} "

    def has_phrase(self, phrase: str) -> bool:
        return f" {phrase} " in self._joined


def _unsupported_terms(
    text: str, support: _Support, context: _Support, protected: Sequence[str]
) -> list[str]:
    """Checkable facts in `text` that neither `support` nor `context` contains."""
    folded = _fold(text)
    missing: list[str] = []

    # Numbers first, over the whole text: `1,200` is one figure, not two words.
    for match in _NUMBER.finditer(folded):
        number = match.group().replace(",", "")
        if number not in support.numbers and number not in context.numbers:
            missing.append(match.group())

    for match in _TOKEN.finditer(folded):
        token = match.group()
        word = _word(token)
        if word in context.words or word in _NEUTRAL_TERMS or word in support.words:
            continue
        if token[0].isdigit():
            continue  # a figure, already checked above
        named = (
            # `S3`, `K8s`, `PostgreSQL`, `AWS`: a name wherever it stands.
            any(char.isupper() or char.isdigit() for char in token[1:])
            or (token[0].isupper() and not _starts_sentence(folded, match.start()))
        )
        if named:
            missing.append(token)

    in_text = _Support([text])
    for phrase in protected:
        if (
            in_text.has_phrase(phrase)
            and not support.has_phrase(phrase)
            and not context.has_phrase(phrase)
        ):
            missing.append(phrase)

    return list(dict.fromkeys(missing))


def validate_draft(
    proposal: DraftProposal,
    evidence: Sequence[EvidenceRecord],
    *,
    context_terms: Iterable[str] = (),
    protected_terms: Iterable[str] = (),
) -> DraftValidation:
    """Flag every claim in `proposal` that does not trace to `evidence`.

    `evidence` is the full set the draft was written from; a citation outside
    it does not resolve, however real the id looks.

    `context_terms` are things a draft may name without that being a claim
    about the candidate: the company and role it is addressed to, the
    candidate's own name. `protected_terms` are lower-case terms that count as
    a claim wherever they appear -- pass the posting's skills, so a draft
    cannot acquire a required tool by mentioning it. `DEGREE_TERMS` are always
    protected.

    Pure: the same proposal and evidence always produce the same flags.
    """
    records = {record.evidence_id: record for record in evidence}
    context = _Support(context_terms)
    protected = list(
        dict.fromkeys(
            phrase
            for phrase in (" ".join(_words(term)) for term in (*protected_terms, *DEGREE_TERMS))
            if phrase
        )
    )
    nothing = _Support(())

    flags: list[ClaimFlag] = []
    cited: list[str] = []

    for index, segment in enumerate(proposal.segments):
        if not segment.text.strip():
            continue

        unknown = [eid for eid in segment.evidence_ids if eid not in records]
        if unknown:
            flags.append(
                ClaimFlag(
                    reason=ClaimFlagReason.UNKNOWN_EVIDENCE,
                    segment_index=index,
                    text=segment.text,
                    terms=tuple(unknown),
                )
            )
            continue

        if not segment.evidence_ids:
            terms = _unsupported_terms(segment.text, nothing, context, protected)
            if terms:
                flags.append(
                    ClaimFlag(
                        reason=ClaimFlagReason.UNCITED,
                        segment_index=index,
                        text=segment.text,
                        terms=tuple(terms),
                    )
                )
            continue

        cited.extend(segment.evidence_ids)
        support = _Support(records[eid].text for eid in segment.evidence_ids)
        terms = _unsupported_terms(segment.text, support, context, protected)
        if terms:
            flags.append(
                ClaimFlag(
                    reason=ClaimFlagReason.UNSUPPORTED_FACT,
                    segment_index=index,
                    text=segment.text,
                    terms=tuple(terms),
                )
            )

    cited_ids = tuple(dict.fromkeys(cited))
    if not cited_ids and not flags:
        flags.append(ClaimFlag(reason=ClaimFlagReason.NO_EVIDENCE))

    return DraftValidation(
        artifact_type=proposal.artifact_type, flags=tuple(flags), cited_evidence_ids=cited_ids
    )


def finalize_draft(
    proposal: DraftProposal,
    evidence: Sequence[EvidenceRecord],
    *,
    context_terms: Iterable[str] = (),
    protected_terms: Iterable[str] = (),
) -> ArtifactDraft:
    """Turn a proposal into a storable draft, or raise `UntraceableClaim`.

    The only way a `DraftProposal` becomes an `ArtifactDraft`. The draft's
    `evidence` is the records its segments cited -- type read from the record,
    never from the model -- so the `artifact_versions` row it is stored as
    links to exactly the evidence ids used.
    """
    validation = validate_draft(
        proposal, evidence, context_terms=context_terms, protected_terms=protected_terms
    )
    if not validation.ok:
        raise UntraceableClaim(validation)

    records = {record.evidence_id: record for record in evidence}
    return ArtifactDraft(
        artifact_type=proposal.artifact_type,
        content="\n".join(
            segment.text.strip() for segment in proposal.segments if segment.text.strip()
        ),
        evidence=tuple(
            EvidenceRef(
                type=records[eid].source_type.value,
                ref=eid,
                excerpt=records[eid].text[:EXCERPT_CHARS],
            )
            for eid in validation.cited_evidence_ids
        ),
    )


def content_sha256(content: str) -> str:
    """Hex SHA-256 of a document's text: the version token an overwrite names."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class DocumentOverwrite(_Value):
    """Replacing a file the candidate already has with a tailored version of it.

    `expected_sha256` is the precondition: the hash of the version the diff was
    computed against. The write is refused if the file no longer hashes to it,
    so an approval of one diff cannot be spent on a file that has since moved.
    """

    path: str
    content: str
    expected_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("path")
    @classmethod
    def _path_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise JobSearchContractError("a document overwrite must name the file it replaces")
        return value

    @classmethod
    def from_payload(cls, payload: dict) -> "DocumentOverwrite":
        """Read the overwrite back out of an `ActionIntent.payload`."""
        return cls(
            path=str(payload.get("path") or ""),
            content=str(payload.get(PAYLOAD_BODY) or ""),
            expected_sha256=str(payload.get("expected_sha256") or ""),
        )


def document_overwrite_intent(
    *,
    path: str,
    current_content: str,
    new_content: str,
    requested_by: str,
    summary: str | None = None,
) -> ActionIntent:
    """Propose overwriting an existing document. Proposes only; nothing is written.

    The payload carries both versions, so the approval request shows the
    reviewer the diff and the action's hash covers the exact text that would
    land. `current_content` must be what the file holds now: its hash becomes
    the write's precondition.
    """
    overwrite = DocumentOverwrite(
        path=path, content=new_content, expected_sha256=content_sha256(current_content)
    )
    return ActionIntent(
        kind=ActionKind.OVERWRITE_DOCUMENT,
        target=overwrite.path,
        summary=summary or f"Overwrite {overwrite.path} with a tailored version",
        payload={
            "path": overwrite.path,
            "expected_sha256": overwrite.expected_sha256,
            PAYLOAD_RECIPIENT: overwrite.path,
            PAYLOAD_BODY: new_content,
            PAYLOAD_PREVIOUS_BODY: current_content,
        },
        # Derived from the file and both versions: re-proposing the same
        # replacement is the same action, and any other one gets a new key.
        idempotency_key="overwrite-" + content_sha256(
            f"{overwrite.path}\n{overwrite.expected_sha256}\n{content_sha256(new_content)}"
        )[:48],
        requested_by=requested_by,
    )


__all__ = [
    "EXCERPT_CHARS",
    "DEGREE_TERMS",
    "DraftSegment",
    "DraftProposal",
    "RetrievedEvidence",
    "ClaimFlagReason",
    "ClaimFlag",
    "DraftValidation",
    "UntraceableClaim",
    "validate_draft",
    "finalize_draft",
    "content_sha256",
    "DocumentOverwrite",
    "document_overwrite_intent",
]

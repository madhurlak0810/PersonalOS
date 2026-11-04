"""Turn a resume file into labelled chunks of evidence.

Three steps, each a pure function of its input:

1. `detect_kind` -- PDF, DOCX, Markdown or plain text, from the extension and
   checked against the content, so a renamed binary is refused rather than
   read as text.
2. `extract_text` -- the document's text as lines. DOCX headings and list
   paragraphs come out as Markdown `#` headings and `- ` bullets, so the
   chunker reads every format the same way. PDF needs `pypdf`; without it a
   PDF is refused with `PortNotConfigured`, and nothing else is affected.
3. `chunk_resume` -- the lines cut into `ResumeChunk`s:

   - a section is opened by a Markdown heading, or by a line that names a
     usual resume section ("Experience", "TECHNICAL SKILLS:", ...);
   - under an experience section each position is one `ROLE` chunk, and
     under a projects section each project is one `PROJECT` chunk. A position
     starts at a sub-heading, or -- in text with no headings, like most PDFs
     -- at a line that carries a date after a blank line or after the last
     position's bullets;
   - a skills section is one `SKILLS` chunk, and any other section one
     `SECTION` chunk;
   - every chunk starts with its section title, and a role or project chunk
     with its own header line too, so a chunk read on its own (which is how
     retrieval hands it to a model) still says what it is about. One that is
     longer than `MAX_CHUNK_CHARS` is split between bullets, each part
     repeating that context.

Chunk text is redacted before anything else sees it: it is embedded, stored
and pasted into model context verbatim.

The heuristics are deliberately plain. A resume the chunker misreads still
yields its full text as evidence, only cut at less useful places; nothing in
the text is dropped.
"""

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePath
from xml.etree import ElementTree

from personalos.domain.errors import PortNotConfigured, ValidationFailed
from personalos.domain.evidence import ChunkKind, DocumentKind, ResumeChunk, normalize_chunk_text
from personalos.domain.redaction import redact_text

#: Longest chunk, in characters, before it is split between bullets.
MAX_CHUNK_CHARS = 1500

#: Most chunks one document may produce. A resume is a few dozen; a file that
#: cuts into more than this is not one.
MAX_CHUNKS = 400

#: Largest `word/document.xml` a DOCX may unpack to. The archive itself is
#: capped by the files tool; this caps what it expands into.
MAX_DOCX_XML_BYTES = 20 * 1024 * 1024

#: Port named when a PDF arrives and `pypdf` is not installed.
PDF_TEXT_PORT = "pdf_text"

_EXTENSIONS: dict[str, DocumentKind] = {
    ".pdf": DocumentKind.PDF,
    ".docx": DocumentKind.DOCX,
    ".md": DocumentKind.MARKDOWN,
    ".markdown": DocumentKind.MARKDOWN,
    ".txt": DocumentKind.TEXT,
    ".text": DocumentKind.TEXT,
}

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# --- 1. Format ----------------------------------------------------------------


def _sniff(content: bytes) -> DocumentKind | None:
    """The format the bytes themselves say they are, if any."""
    if content.startswith(b"%PDF-"):
        return DocumentKind.PDF
    if content.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile:
            return None
        return DocumentKind.DOCX if "word/document.xml" in names else None
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return DocumentKind.TEXT


def detect_kind(path: str, content: bytes) -> DocumentKind:
    """The document's format, or `ValidationFailed` if it is not one we read."""
    suffix = PurePath(path).suffix.lower()
    declared = _EXTENSIONS.get(suffix)
    sniffed = _sniff(content)
    if declared is None:
        if sniffed is None:
            raise ValidationFailed(
                f"{PurePath(path).name!r} is not a PDF, DOCX, Markdown or text file"
            )
        return sniffed
    binary = (DocumentKind.PDF, DocumentKind.DOCX)
    if declared in binary and sniffed != declared:
        raise ValidationFailed(
            f"{PurePath(path).name!r} has a {suffix} extension but is not a {declared.value} file"
        )
    if declared not in binary and sniffed != DocumentKind.TEXT:
        raise ValidationFailed(f"{PurePath(path).name!r} has a {suffix} extension but is not text")
    return declared


# --- 2. Text ------------------------------------------------------------------


def extract_text(content: bytes, kind: DocumentKind) -> str:
    """The document's text, newline-separated; headings as `#` lines where known."""
    if kind == DocumentKind.PDF:
        text = _pdf_text(content)
    elif kind == DocumentKind.DOCX:
        text = _docx_text(content)
    else:
        text = content.decode("utf-8")
    return text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")


def _pdf_text(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PortNotConfigured(
            "reading a PDF resume needs the 'pypdf' package; install it, or ingest "
            "a DOCX, Markdown or text export of the resume",
            port=PDF_TEXT_PORT,
        ) from exc
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValidationFailed("the PDF is password-protected")
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ValidationFailed:
        raise
    except Exception as exc:
        raise ValidationFailed("the file could not be read as a PDF") from exc


def _docx_text(content: bytes) -> str:
    root = _docx_body(content)
    return "\n".join(_docx_line(paragraph) for paragraph in root.iter(f"{_W}p"))


def _docx_body(content: bytes) -> ElementTree.Element:
    """The parsed `word/document.xml`, within `MAX_DOCX_XML_BYTES`."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > MAX_DOCX_XML_BYTES:
                raise ValidationFailed("the DOCX document body is too large to be a resume")
            with archive.open(info) as handle:
                xml = handle.read(MAX_DOCX_XML_BYTES + 1)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValidationFailed("the file could not be read as a DOCX document") from exc
    if len(xml) > MAX_DOCX_XML_BYTES:
        raise ValidationFailed("the DOCX document body is too large to be a resume")
    try:
        return ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise ValidationFailed("the DOCX document body is not well-formed XML") from exc


def _docx_line(paragraph: ElementTree.Element) -> str:
    """One paragraph as a line: `#` for a heading style, `- ` for a list item."""
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{_W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{_W}tab":
            parts.append("\t")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
    text = "".join(parts).strip()
    if not text:
        return ""
    props = paragraph.find(f"{_W}pPr")
    style = props.find(f"{_W}pStyle") if props is not None else None
    style_name = (style.get(f"{_W}val") or "") if style is not None else ""
    heading = re.fullmatch(r"(?i)heading\s*([1-6])", style_name)
    if heading or style_name.lower() == "title":
        return f"{'#' * (int(heading.group(1)) if heading else 1)} {text}"
    if props is not None and props.find(f"{_W}numPr") is not None:
        return f"- {text}"
    return text


# --- 3. Chunks ----------------------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_BULLET = re.compile(r"^(?:[-*+•·▪◦‣–—]|\d{1,2}[.)])\s+")
_DATED = re.compile(r"\b(?:19|20)\d{2}\b|\b(?:present|current)\b", re.IGNORECASE)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"(\*\*|__|`)")

#: Section headings, canonicalized, that open a section wherever they appear.
_KNOWN_SECTIONS = frozenset(
    {
        "summary",
        "professional summary",
        "profile",
        "professional profile",
        "about",
        "about me",
        "objective",
        "experience",
        "work experience",
        "professional experience",
        "relevant experience",
        "employment",
        "employment history",
        "work history",
        "career history",
        "projects",
        "personal projects",
        "selected projects",
        "side projects",
        "open source",
        "skills",
        "technical skills",
        "key skills",
        "core competencies",
        "technologies",
        "tools",
        "skills and tools",
        "education",
        "certifications",
        "certificates",
        "licenses and certifications",
        "publications",
        "awards",
        "honors",
        "honours",
        "volunteering",
        "volunteer experience",
        "leadership",
        "languages",
        "interests",
        "achievements",
        "patents",
        "talks",
        "courses",
        "training",
    }
)

#: Which kind of chunk a section's content becomes, by its heading; first match wins.
_SECTION_KINDS: tuple[tuple[re.Pattern[str], ChunkKind], ...] = (
    (re.compile(r"\bprojects?\b|\bportfolio\b|\bopen source\b"), ChunkKind.PROJECT),
    (
        re.compile(r"\b(?:experience|employment|work history|career history|positions?)\b"),
        ChunkKind.ROLE,
    ),
    (
        re.compile(r"\b(?:skills?|competenc(?:y|ies)|technolog(?:y|ies)|tools|tech stack)\b"),
        ChunkKind.SKILLS,
    ),
)

#: Experience under these headings is not a list of positions.
_NOT_ROLES = re.compile(r"\bvolunteer")


def _canonical_heading(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _section_kind(title: str) -> ChunkKind | None:
    canonical = _canonical_heading(title)
    for pattern, kind in _SECTION_KINDS:
        if pattern.search(canonical):
            if kind == ChunkKind.ROLE and _NOT_ROLES.search(canonical):
                return ChunkKind.SECTION
            return kind
    return None


def _plain_heading(line: str) -> str | None:
    """The section title if this unmarked line is a section heading."""
    if _BULLET.match(line) or len(line) > 40:
        return None
    title = line.rstrip(":").strip()
    canonical = _canonical_heading(title)
    if not canonical or len(canonical.split()) > 5:
        return None
    if canonical in _KNOWN_SECTIONS:
        return title
    # Title case alone is not enough here: "Engineer, Experience Platform" is a
    # role header, not an experience section.
    looks_like_heading = line.endswith(":") or title.isupper()
    if looks_like_heading and _section_kind(title) is not None:
        return title
    return None


def _clean_inline(text: str) -> str:
    return _MD_EMPHASIS.sub("", _MD_LINK.sub(r"\1", text)).strip()


def _slug(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-")


@dataclass
class _Item:
    text: str
    bullet: bool


@dataclass
class _Entry:
    header: list[str] = field(default_factory=list)
    items: list[_Item] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.header and not self.items


@dataclass
class _Section:
    title: str | None
    kind: ChunkKind
    level: int = 0
    entries: list[_Entry] = field(default_factory=list)

    @property
    def splits_entries(self) -> bool:
        return self.kind in (ChunkKind.ROLE, ChunkKind.PROJECT)

    def open_entry(self, header: str | None = None) -> _Entry:
        entry = _Entry(header=[header] if header else [])
        self.entries.append(entry)
        return entry

    def add(self, text: str, *, bullet: bool, after_blank: bool) -> None:
        entry = self.entries[-1] if self.entries else None
        if not self.splits_entries:
            (entry or self.open_entry()).items.append(_Item(text, bullet))
            return
        if bullet:
            (entry or self.open_entry()).items.append(_Item(text, True))
            return
        if entry is None:
            self.open_entry(text)
            return
        dated = bool(_DATED.search(text))
        if dated and (after_blank or (entry.items and not text[:1].islower())):
            self.open_entry(text)
        elif entry.items and not after_blank and entry.items[-1].bullet:
            # A bullet wrapped onto the next line, as PDF text extraction leaves it.
            entry.items[-1].text = f"{entry.items[-1].text} {text}"
        elif not entry.items and not after_blank:
            entry.header.append(text)
        elif after_blank and len(text) <= 100 and entry.items:
            self.open_entry(text)
        else:
            entry.items.append(_Item(text, False))


def _parse(text: str) -> list[_Section]:
    sections = [_Section(title=None, kind=ChunkKind.SECTION)]
    after_blank = False
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            after_blank = True
            continue
        current = sections[-1]

        heading = _MD_HEADING.match(line)
        if heading:
            level, title = len(heading.group(1)), _clean_inline(heading.group(2))
            kind = _section_kind(title)
            if _canonical_heading(title) in _KNOWN_SECTIONS or kind is not None:
                sections.append(_Section(title, kind or ChunkKind.SECTION, level))
            elif current.splits_entries and level > current.level:
                current.open_entry(title)
            else:
                sections.append(_Section(title, ChunkKind.SECTION, level))
            after_blank = False
            continue

        title = _plain_heading(line)
        if title is not None:
            sections.append(_Section(title, _section_kind(title) or ChunkKind.SECTION))
            after_blank = False
            continue

        bullet = _BULLET.match(line)
        content = _clean_inline(line[bullet.end() :] if bullet else line)
        if content:
            current.add(content, bullet=bool(bullet), after_blank=after_blank)
        after_blank = False
    return sections


def _render(item: _Item) -> str:
    return f"- {item.text}" if item.bullet else item.text


def _pack(prefix: list[str], items: list[str]) -> list[str]:
    """`prefix` plus `items`, as few texts of at most `MAX_CHUNK_CHARS` as fit."""
    budget = max(MAX_CHUNK_CHARS - sum(len(line) + 1 for line in prefix), MAX_CHUNK_CHARS // 2)
    pieces: list[str] = []
    for item in items:
        while len(item) > budget:
            cut = item.rfind(" ", 0, budget)
            cut = cut if cut > budget // 2 else budget
            pieces.append(item[:cut].rstrip())
            item = item[cut:].lstrip()
        if item:
            pieces.append(item)

    texts: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        if current and size + len(piece) + 1 > budget:
            texts.append("\n".join([*prefix, *current]))
            current, size = [], 0
        current.append(piece)
        size += len(piece) + 1
    if current:
        texts.append("\n".join([*prefix, *current]))
    elif prefix:
        texts.append("\n".join(prefix))
    return texts


def chunk_resume(text: str, *, document_name: str) -> list[ResumeChunk]:
    """Cut a resume's text into labelled chunks, in document order.

    `document_name` prefixes every `source_ref` (`resume.md#experience/acme`),
    so a citation says which file it came from. Chunks whose text repeats an
    earlier chunk's are dropped: they would have the same evidence id.
    """
    chunks: list[ResumeChunk] = []
    seen: set[str] = set()
    for section in _parse(text):
        section_slug = _slug(section.title) if section.title else "header"
        title_lines = [section.title] if section.title else []
        for entry in section.entries:
            if entry.empty:
                continue
            ref = f"{document_name}#{section_slug or 'section'}"
            if section.splits_entries and entry.header:
                ref = f"{ref}/{_slug(entry.header[0]) or 'entry'}"
            prefix = [*title_lines, *(entry.header if section.splits_entries else [])]
            items = [*([] if section.splits_entries else entry.header)]
            items += [_render(item) for item in entry.items]
            parts = _pack(prefix, items)
            for number, part in enumerate(parts, start=1):
                body = redact_text(part)
                key = normalize_chunk_text(body)
                if not key or key == normalize_chunk_text("\n".join(title_lines)) or key in seen:
                    continue
                seen.add(key)
                suffix = f"~{number}" if len(parts) > 1 else ""
                chunks.append(
                    ResumeChunk(kind=section.kind, source_ref=f"{ref}{suffix}"[:255], text=body)
                )
                if len(chunks) > MAX_CHUNKS:
                    raise ValidationFailed(
                        f"the document cuts into more than {MAX_CHUNKS} chunks; "
                        f"it does not look like a resume"
                    )
    return chunks


__all__ = [
    "MAX_CHUNK_CHARS",
    "MAX_CHUNKS",
    "MAX_DOCX_XML_BYTES",
    "PDF_TEXT_PORT",
    "detect_kind",
    "extract_text",
    "chunk_resume",
]

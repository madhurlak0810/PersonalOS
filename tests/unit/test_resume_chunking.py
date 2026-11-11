"""Tests for resume format detection, text extraction, chunking and the default embedder."""

import math

import pytest

from personalos.domain.errors import ValidationFailed
from personalos.domain.evidence import ChunkKind, DocumentKind, evidence_id_for
from personalos.domain.models import EvidenceSourceType
from personalos.models.embedding import HashingEmbedder, tokenize
from personalos.persistence.models import EMBEDDING_DIMENSION
from personalos.retrieval.resume_chunking import (
    MAX_CHUNK_CHARS,
    chunk_resume,
    detect_kind,
    extract_text,
)
from tests.fixtures import evidence_ingestion as fx
from tests.fixtures import job_search_fakes as fakes

PLAIN_TEXT_RESUME = """Jane Candidate
jane@example.test

PROFESSIONAL EXPERIENCE
Senior Software Engineer, Acme Corp
Jan 2019 - Present
• Built Python APIs backed by Postgres serving 1,200 requests
per second across three regions.
• Cut p95 latency by 40%.
Software Engineer, Globex 2016 - 2019
• Operated Kafka pipelines.

Technical Skills:
Python, Go, SQL
"""


def chunks(text: str, name: str = "resume.md"):
    return chunk_resume(text, document_name=name)


class TestDetectKind:
    @pytest.mark.parametrize(
        "name, kind",
        [
            ("resume.md", DocumentKind.MARKDOWN),
            ("resume.markdown", DocumentKind.MARKDOWN),
            ("resume.txt", DocumentKind.TEXT),
            ("RESUME.TXT", DocumentKind.TEXT),
            ("resume", DocumentKind.TEXT),
        ],
    )
    def test_text_formats_by_extension(self, name, kind):
        assert detect_kind(name, b"## Skills\nPython") == kind

    def test_binary_formats_are_checked_against_their_content(self):
        assert detect_kind("cv.pdf", fx.minimal_pdf(["Hi"])) == DocumentKind.PDF
        assert detect_kind("cv.docx", fx.minimal_docx([(None, "Hi")])) == DocumentKind.DOCX
        assert detect_kind("cv", fx.minimal_docx([(None, "Hi")])) == DocumentKind.DOCX
        with pytest.raises(ValidationFailed, match="not a pdf file"):
            detect_kind("cv.pdf", b"## Skills")
        with pytest.raises(ValidationFailed, match="not a docx file"):
            detect_kind("cv.docx", b"PK\x03\x04not really a zip")

    def test_an_unknown_binary_is_refused(self):
        with pytest.raises(ValidationFailed, match="not a PDF, DOCX"):
            detect_kind("cv.bin", b"\x00\xff\xfe\xfd")


class TestExtractText:
    def test_docx_headings_and_lists_become_markdown(self):
        content = fx.minimal_docx(
            [
                ("Title", "Jane"),
                ("Heading2", "Experience"),
                ("List", "Built <things> & more"),
                (None, "Plain"),
            ]
        )

        assert extract_text(content, DocumentKind.DOCX).split("\n") == [
            "# Jane",
            "## Experience",
            "- Built <things> & more",
            "Plain",
        ]

    def test_a_malformed_docx_is_refused(self):
        with pytest.raises(ValidationFailed):
            extract_text(b"PK\x03\x04junk", DocumentKind.DOCX)

    def test_text_loses_its_bom_and_carriage_returns(self):
        assert extract_text("﻿a\r\nb\rc".encode(), DocumentKind.TEXT) == "a\nb\nc"


class TestChunkResume:
    def test_markdown_roles_projects_skills_and_sections(self):
        result = chunks(fx.FIXTURE_RESUME.read_text())
        kinds = {chunk.source_ref: chunk.kind for chunk in result}

        assert kinds == {
            "resume.md#jane-candidate": ChunkKind.SECTION,
            "resume.md#summary": ChunkKind.SECTION,
            "resume.md#experience/senior-software-engineer-acme-corp-2019-2023": ChunkKind.ROLE,
            "resume.md#experience/software-engineer-globex-2016-2019": ChunkKind.ROLE,
            "resume.md#projects/pgsync": ChunkKind.PROJECT,
            "resume.md#skills": ChunkKind.SKILLS,
            "resume.md#education": ChunkKind.SECTION,
        }
        project = next(chunk for chunk in result if chunk.kind == ChunkKind.PROJECT)
        assert project.source_type == EvidenceSourceType.PROJECT

    def test_a_role_chunk_carries_its_section_and_header(self):
        acme = next(
            chunk for chunk in chunks(fx.FIXTURE_RESUME.read_text()) if "acme" in chunk.source_ref
        )

        assert acme.text.split("\n")[:2] == [
            "Experience",
            "Senior Software Engineer, Acme Corp (2019 - 2023)",
        ]
        assert "- Built Python APIs backed by Postgres" in acme.text
        assert "**" not in acme.text

    def test_plain_text_with_no_headings_is_cut_by_dates_and_known_sections(self):
        result = chunks(PLAIN_TEXT_RESUME, "resume.txt")
        refs = [chunk.source_ref for chunk in result]

        assert refs == [
            "resume.txt#header",
            "resume.txt#professional-experience/senior-software-engineer-acme-corp",
            "resume.txt#professional-experience/software-engineer-globex-2016-2019",
            "resume.txt#technical-skills",
        ]
        acme = result[1]
        # The bullet PDF extraction wrapped onto two lines is put back together.
        assert "requests per second across three regions." in acme.text
        assert "Jan 2019 - Present" in acme.text

    def test_a_title_case_role_line_is_not_mistaken_for_a_section(self):
        text = "## Experience\nEngineer, Experience Platform\n- Shipped things.\n"

        (chunk,) = chunks(text)

        assert chunk.kind == ChunkKind.ROLE
        assert chunk.source_ref == "resume.md#experience/engineer-experience-platform"

    def test_volunteer_experience_is_a_section_not_roles(self):
        (chunk,) = chunks("## Volunteer Experience\nTreasurer, Garden Club 2020\n- Budget.\n")

        assert chunk.kind == ChunkKind.SECTION

    def test_a_long_role_is_split_between_bullets_repeating_its_header(self):
        bullets = "\n".join(f"- Achievement {n}: " + "word " * 40 for n in range(30))
        result = chunks(f"## Experience\n### Engineer, Acme (2020)\n{bullets}\n")

        assert len(result) > 1
        for number, chunk in enumerate(result, start=1):
            assert len(chunk.text) <= MAX_CHUNK_CHARS
            assert chunk.text.startswith("Experience\nEngineer, Acme (2020)\n- Achievement")
            assert chunk.source_ref.endswith(f"~{number}")
        assert sum(chunk.text.count("- Achievement") for chunk in result) == 30

    def test_secrets_are_redacted_before_anything_sees_them(self):
        text = "## Skills\nPython\napi_key=sk-live-0123456789abcdefghijklmnop\n"

        (chunk,) = chunks(text)

        assert "sk-live-0123456789abcdefghijklmnop" not in chunk.text

    def test_repeated_text_is_one_chunk(self):
        result = chunks("## Skills\nPython\n## Languages\nEnglish\n## Skills\nPython\n")

        assert [chunk.source_ref for chunk in result] == ["resume.md#skills", "resume.md#languages"]

    def test_a_heading_with_nothing_under_it_is_no_chunk(self):
        assert chunks("## Experience\n\n## Education\n") == []


class TestStableIds:
    def test_ids_ignore_whitespace_and_depend_on_user_path_and_text(self):
        user = fakes.USER_ID
        base = evidence_id_for(user, "/r/resume.md", "Built  Python\nAPIs")

        assert base == evidence_id_for(user, "/r/resume.md", "Built Python APIs ")
        assert base != evidence_id_for(user, "/r/other.md", "Built Python APIs")
        assert base != evidence_id_for(fakes.POSTING_ID, "/r/resume.md", "Built Python APIs")
        assert base != evidence_id_for(user, "/r/resume.md", "Built Go APIs")


class TestHashingEmbedder:
    def test_vectors_are_unit_length_and_fit_the_column(self):
        vector = HashingEmbedder().vector("Built Python APIs backed by Postgres")

        assert len(vector) == EMBEDDING_DIMENSION
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)

    def test_deterministic_and_shared_words_are_closer(self):
        embedder = HashingEmbedder()

        def similarity(a: str, b: str) -> float:
            return sum(x * y for x, y in zip(embedder.vector(a), embedder.vector(b), strict=True))

        assert embedder.vector("Kafka pipelines") == embedder.vector("Kafka pipelines")
        assert similarity("Python APIs on Postgres", "Built Python APIs backed by Postgres") > (
            similarity("Python APIs on Postgres", "Volunteer treasurer for a garden")
        )

    def test_language_names_with_symbols_are_kept_whole(self):
        assert tokenize("C++, C#, Node.js and Python 3.11.") == [
            "c++",
            "c#",
            "node.js",
            "and",
            "python",
            "3.11",
        ]

    def test_text_with_no_words_is_refused(self):
        with pytest.raises(ValueError):
            HashingEmbedder().vector("  --  ")

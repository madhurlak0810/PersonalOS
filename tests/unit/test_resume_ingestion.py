"""Tests for resume ingestion: read through the sandbox, chunk, embed, index, re-index."""

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.routes.evidence import get_resume_ingestor
from personalos import bootstrap
from personalos import cli as cli_module
from personalos.cli import cli
from personalos.domain.errors import PortNotConfigured, ValidationFailed
from personalos.domain.evidence import (
    ChunkKind,
    DocumentKind,
    IngestionStatus,
    evidence_id_for,
)
from personalos.domain.models import EvidenceSourceType
from personalos.domain.search_profiles import actor_user_id
from personalos.executor.document_reader import DocumentReadRefused
from personalos.persistence.evidence import SqlEvidenceIndex, SqlEvidenceSource
from personalos.persistence.models import EMBEDDING_DIMENSION
from personalos.retrieval.embedding import HASHING_EMBEDDER_MODEL, HASHING_EMBEDDER_VERSION
from tests.fixtures import evidence_ingestion as fx
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.job_discovery import session_factory

USER = fakes.USER_ID


@pytest.fixture
def root(tmp_path):
    return fx.allowed_root(tmp_path)


@pytest.fixture
def factory(tmp_path):
    return session_factory(tmp_path / "db")


def _edit(root, old: str, new: str) -> None:
    resume = root / "resume.md"
    text = resume.read_text()
    assert old in text
    resume.write_text(text.replace(old, new))


# --- Acceptance: a fixture resume becomes stable, embedded evidence --------------------


async def test_ingesting_the_fixture_resume_indexes_embedded_evidence_with_stable_ids(
    root, factory
):
    result = await fx.ingestor(root, factory).ingest(USER, "resume.md")

    assert result.status == IngestionStatus.INDEXED
    document = result.document
    assert document.path == str((root / "resume.md").resolve())
    assert document.kind == DocumentKind.MARKDOWN
    assert len(document.sha256) == 64
    assert document.indexed_at is not None
    assert result.added == len(document.evidence) > 0

    rows = fx.snapshot(factory)["chunks"]
    assert set(rows) == {str(item.evidence_id) for item in document.evidence}
    for row in rows.values():
        assert row["document_id"] == str(document.document_id)
        assert len(row["embedding"]) == EMBEDDING_DIMENSION
        assert any(row["embedding"])
        assert row["embedding_model"] == HASHING_EMBEDDER_MODEL
        assert row["embedding_version"] == HASHING_EMBEDDER_VERSION
        # The id is a function of who, which file and what text -- nothing random.
        assert row["id"] == str(evidence_id_for(USER, document.path, row["chunk_text"]))


async def test_a_second_ingest_of_the_unchanged_file_changes_nothing(root, factory):
    embedder = fx.CountingEmbedder()
    ingestor = fx.ingestor(root, factory, embedder)
    first = await ingestor.ingest(USER, "resume.md")
    before = fx.snapshot(factory)
    calls = len(embedder.calls)

    second = await ingestor.ingest(USER, "resume.md")

    assert second.status == IngestionStatus.UNCHANGED
    assert second.document == first.document
    assert (second.added, second.kept, second.retired, second.deleted) == (0, 0, 0, 0)
    assert fx.snapshot(factory) == before
    assert len(embedder.calls) == calls


async def test_the_same_resume_gets_the_same_ids_in_a_fresh_index(root, factory, tmp_path):
    first = await fx.ingestor(root, factory).ingest(USER, "resume.md")
    elsewhere = session_factory(tmp_path / "other-db")

    second = await fx.ingestor(root, elsewhere).ingest(USER, "resume.md")

    assert [item.evidence_id for item in second.document.evidence] == [
        item.evidence_id for item in first.document.evidence
    ]


async def test_the_resume_is_chunked_into_sections_roles_projects_and_skills(root, factory):
    result = await fx.ingestor(root, factory).ingest(USER, "resume.md")
    by_ref = {item.source_ref: item for item in result.document.evidence}

    assert "resume.md#experience/senior-software-engineer-acme-corp-2019-2023" in by_ref
    assert "resume.md#experience/software-engineer-globex-2016-2019" in by_ref
    assert by_ref["resume.md#projects/pgsync"].source_type == EvidenceSourceType.PROJECT
    assert "resume.md#skills" in by_ref
    assert "resume.md#education" in by_ref
    assert [item.chunk_index for item in result.document.evidence] == list(
        range(len(result.document.evidence))
    )


async def test_ingested_evidence_is_what_job_matching_and_artifact_prep_read(root, factory):
    result = await fx.ingestor(root, factory).ingest(USER, "resume.md")
    ids = {str(item.evidence_id) for item in result.document.evidence}

    loaded = await SqlEvidenceSource(factory).load(USER)
    embedder = fx.CountingEmbedder()
    nearest = await SqlEvidenceIndex(factory).search(
        user_id=USER,
        query_embedding=embedder.vector("Kafka pipelines and Kubernetes in production"),
        embedding_model=embedder.model,
        top_k=1,
        min_similarity=0.0,
    )

    assert {record.evidence_id for record in loaded} == ids
    assert "Kafka pipelines" in nearest[0].record.text


# --- Acceptance: the sandbox bounds what can be ingested ---------------------------------


@pytest.mark.parametrize("path", ["../../outside-root.pdf", "{outside}", "../outside-root.pdf"])
async def test_a_path_outside_the_allowed_roots_is_rejected(root, factory, path):
    outside = root.parent.parent / "outside-root.pdf"
    outside.write_bytes(fx.minimal_pdf(["Not the candidate's resume."]))
    (root.parent / "outside-root.pdf").write_bytes(outside.read_bytes())

    with pytest.raises(DocumentReadRefused, match="outside the allowed roots"):
        await fx.ingestor(root, factory).ingest(USER, path.format(outside=outside))

    assert fx.snapshot(factory) == {"documents": [], "chunks": {}}


async def test_a_symlink_out_of_the_root_is_rejected(root, factory):
    secret = root.parent / "elsewhere.md"
    secret.write_text("## Experience\n- Not inside the root.\n")
    (root / "linked.md").symlink_to(secret)

    with pytest.raises(DocumentReadRefused):
        await fx.ingestor(root, factory).ingest(USER, "linked.md")

    assert fx.snapshot(factory)["chunks"] == {}


# --- Re-ingesting a changed file -------------------------------------------------------


async def test_an_edit_keeps_the_ids_of_unchanged_chunks_and_embeds_only_new_ones(root, factory):
    embedder = fx.CountingEmbedder()
    ingestor = fx.ingestor(root, factory, embedder)
    first = await ingestor.ingest(USER, "resume.md")
    embedder.calls.clear()

    _edit(root, "Docker, Terraform", "Docker, Terraform, Go")
    second = await ingestor.ingest(USER, "resume.md")

    assert second.status == IngestionStatus.UPDATED
    assert second.document.sha256 != first.document.sha256
    assert second.document.indexed_at >= first.document.indexed_at
    old = {item.source_ref: item.evidence_id for item in first.document.evidence}
    new = {item.source_ref: item.evidence_id for item in second.document.evidence}
    assert old.keys() == new.keys()
    assert {ref for ref in old if old[ref] != new[ref]} == {"resume.md#skills"}
    assert (second.added, second.kept, second.deleted, second.retired) == (
        1,
        len(old) - 1,
        1,
        0,
    )
    assert len(embedder.calls) == 1 and "Go" in embedder.calls[0]


async def test_a_dropped_chunk_an_artifact_cites_is_retired_not_deleted(root, factory):
    ingestor = fx.ingestor(root, factory)
    first = await ingestor.ingest(USER, "resume.md")
    globex = next(item for item in first.document.evidence if "globex" in item.source_ref)
    fx.cite(factory, USER, [str(globex.evidence_id)])

    _edit(root, "- Wrote the on-call runbook for the ingestion platform.\n", "")
    second = await ingestor.ingest(USER, "resume.md")

    assert second.retired == 1 and second.deleted == 0
    rows = fx.snapshot(factory)["chunks"]
    assert rows[str(globex.evidence_id)]["retired_at"] is not None
    assert "runbook" in rows[str(globex.evidence_id)]["chunk_text"]
    active = {record.evidence_id for record in await SqlEvidenceSource(factory).load(USER)}
    assert str(globex.evidence_id) not in active
    assert {str(item.evidence_id) for item in second.document.evidence} == active


async def test_restoring_retired_text_brings_back_the_same_evidence_id(root, factory):
    ingestor = fx.ingestor(root, factory)
    first = await ingestor.ingest(USER, "resume.md")
    globex = next(item for item in first.document.evidence if "globex" in item.source_ref)
    fx.cite(factory, USER, [str(globex.evidence_id)])
    runbook = "- Wrote the on-call runbook for the ingestion platform.\n"
    _edit(root, runbook, "")
    await ingestor.ingest(USER, "resume.md")

    _edit(root, "clusters in production.\n", f"clusters in production.\n{runbook}")
    third = await ingestor.ingest(USER, "resume.md")

    assert third.added == 1
    assert [item.evidence_id for item in third.document.evidence] == [
        item.evidence_id for item in first.document.evidence
    ]
    assert all(row["retired_at"] is None for row in fx.snapshot(factory)["chunks"].values())


async def test_reordering_sections_moves_chunks_without_renaming_them(root, factory):
    ingestor = fx.ingestor(root, factory)
    first = await ingestor.ingest(USER, "resume.md")
    resume = root / "resume.md"
    text = resume.read_text()
    education = "## Education\nB.S. Computer Science, State University, 2016\n"
    resume.write_text(text.replace(education, "").replace("## Summary", f"{education}\n## Summary"))

    second = await ingestor.ingest(USER, "resume.md")

    assert second.kept == len(first.document.evidence) and second.added == 0
    assert {item.evidence_id for item in second.document.evidence} == {
        item.evidence_id for item in first.document.evidence
    }
    assert second.document.evidence[1].source_ref == "resume.md#education"


async def test_a_new_embedder_re_embeds_an_unchanged_file_under_the_same_ids(root, factory):
    first = await fx.ingestor(root, factory).ingest(USER, "resume.md")
    upgraded = fx.CountingEmbedder(model="personalos-hashing", version="2")

    second = await fx.ingestor(root, factory, upgraded).ingest(USER, "resume.md")

    assert second.status == IngestionStatus.UPDATED
    assert len(upgraded.calls) == len(first.document.evidence)
    assert [item.evidence_id for item in second.document.evidence] == [
        item.evidence_id for item in first.document.evidence
    ]
    assert {row["embedding_version"] for row in fx.snapshot(factory)["chunks"].values()} == {"2"}


# --- Formats -------------------------------------------------------------------------------


async def test_a_docx_resume_is_ingested_with_its_headings_and_lists(root, factory):
    (root / "resume.docx").write_bytes(
        fx.minimal_docx(
            [
                ("Title", "Jane Candidate"),
                ("Heading1", "Experience"),
                ("Heading2", "Senior Software Engineer, Acme Corp (2019 - 2023)"),
                ("List", "Built Python APIs backed by Postgres."),
                ("Heading1", "Skills"),
                (None, "Python, Postgres"),
            ]
        )
    )

    result = await fx.ingestor(root, factory).ingest(USER, "resume.docx")

    assert result.document.kind == DocumentKind.DOCX
    refs = [item.source_ref for item in result.document.evidence]
    assert "resume.docx#experience/senior-software-engineer-acme-corp-2019-2023" in refs
    assert "resume.docx#skills" in refs


async def test_a_pdf_resume_is_ingested(root, factory):
    pytest.importorskip("pypdf")
    (root / "resume.pdf").write_bytes(
        fx.minimal_pdf(
            [
                "EXPERIENCE",
                "Senior Software Engineer, Acme Corp 2019 - 2023",
                "- Built Python APIs backed by Postgres.",
                "SKILLS",
                "Python, Postgres",
            ]
        )
    )

    result = await fx.ingestor(root, factory).ingest(USER, "resume.pdf")

    assert result.document.kind == DocumentKind.PDF
    texts = [record.text for record in await SqlEvidenceSource(factory).load(USER)]
    assert any("Built Python APIs" in text for text in texts)


async def test_a_pdf_without_pypdf_installed_is_refused_as_unconfigured(root, factory, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_pypdf(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pypdf)
    (root / "resume.pdf").write_bytes(fx.minimal_pdf(["EXPERIENCE"]))

    with pytest.raises(PortNotConfigured, match="pypdf"):
        await fx.ingestor(root, factory).ingest(USER, "resume.pdf")


async def test_a_binary_file_named_like_text_is_refused(root, factory):
    (root / "resume.txt").write_bytes(fx.minimal_pdf(["Hidden"]) + b"\xff\xfe")

    with pytest.raises(ValidationFailed, match="not text"):
        await fx.ingestor(root, factory).ingest(USER, "resume.txt")


async def test_a_resume_with_no_text_is_refused(root, factory):
    (root / "empty.md").write_text("\n\n   \n")

    with pytest.raises(ValidationFailed, match="no text"):
        await fx.ingestor(root, factory).ingest(USER, "empty.md")
    assert fx.snapshot(factory)["documents"] == []


# --- Configuration -------------------------------------------------------------------------


def test_the_default_embedder_is_the_local_hashing_one():
    embedder = bootstrap.build_embedder("hashing")

    assert embedder.model == HASHING_EMBEDDER_MODEL
    assert bootstrap.build_embedder("") is None
    with pytest.raises(ValueError, match="unknown EMBEDDING_PROVIDER"):
        bootstrap.build_embedder("openai")


def test_ingestion_without_an_embedder_is_refused(factory, monkeypatch):
    monkeypatch.setattr(bootstrap.settings, "embedding_provider", "")

    with pytest.raises(PortNotConfigured, match="EMBEDDING_PROVIDER"):
        bootstrap.build_resume_ingestor(factory)


# --- CLI and API -----------------------------------------------------------------------------


ACTOR = "jane"


@pytest.fixture
def client(root, factory):
    app = create_app()
    app.dependency_overrides[get_resume_ingestor] = lambda: fx.ingestor(root, factory)
    return TestClient(app, headers={"X-Actor-Id": ACTOR})


def test_the_api_indexes_then_reports_the_unchanged_file(client, factory):
    created = client.post("/v1/evidence/documents", json={"path": "resume.md"})
    again = client.post("/v1/evidence/documents", json={"path": "resume.md"})

    assert created.status_code == 201, created.text
    assert created.json()["status"] == "indexed"
    assert created.json()["document"]["user_id"] == str(actor_user_id(ACTOR))
    assert again.status_code == 200
    assert again.json()["status"] == "unchanged"
    assert again.json()["document"] == created.json()["document"]


def test_the_api_refuses_a_path_outside_the_roots(client, factory):
    response = client.post("/v1/evidence/documents", json={"path": "../../outside-root.pdf"})

    assert response.status_code == 403
    assert "outside the allowed roots" in response.text
    assert fx.snapshot(factory)["chunks"] == {}


def test_the_api_refuses_unknown_fields(client):
    response = client.post("/v1/evidence/documents", json={"path": "resume.md", "root": "/"})

    assert response.status_code == 422


def test_the_cli_ingests_and_prints_the_api_body(root, factory, monkeypatch):
    monkeypatch.setattr(cli_module, "init_db", lambda: None)
    monkeypatch.setattr(bootstrap, "initialize_mcp_servers", lambda: None)
    monkeypatch.setattr(bootstrap, "build_resume_ingestor", lambda **_: fx.ingestor(root, factory))
    runner = CliRunner()

    first = runner.invoke(cli, ["ingest-resume", "resume.md", "--actor", ACTOR])
    second = runner.invoke(cli, ["ingest-resume", "resume.md", "--actor", ACTOR])
    outside = runner.invoke(cli, ["ingest-resume", "../../outside-root.pdf", "--actor", ACTOR])

    assert first.exit_code == 0, first.output
    assert '"status": "indexed"' in first.output
    assert '"status": "unchanged"' in second.output
    assert outside.exit_code != 0
    assert "outside the allowed roots" in outside.output


def test_chunk_kinds_cover_the_resume():
    """Every kind the chunker can produce is one the API documents."""
    assert {kind.value for kind in ChunkKind} == {"section", "role", "project", "skills"}

"""Tests for swapping the embedder and backfilling stored vectors (`cli reembed`).

The acceptance criteria:

- swapping the configured embedder is a settings change. The same composition
  functions build ingestion and artifact prep, with no graph code involved, and
  new vectors carry the new model and version;
- the backfill resumes after a crash mid-run and leaves every cited evidence id
  intact, retired chunks included.
"""

from datetime import datetime
from uuid import UUID

import pytest
from click.testing import CliRunner

from personalos import bootstrap
from personalos import cli as cli_module
from personalos.cli import cli
from personalos.domain.embeddings import VectorTable
from personalos.domain.evidence import IngestionStatus
from personalos.models.embedding import HASHING_EMBEDDER_MODEL, HashingEmbedder
from personalos.persistence.models import (
    ApplicationModel,
    CommunicationEventModel,
    DocumentModel,
    EvidenceChunkModel,
    JobPostingEmbeddingModel,
    JobPostingModel,
    MessageEmbeddingModel,
)
from personalos.retrieval.artifact_prep import retrieval_query
from tests.fixtures import artifact_prep as prep
from tests.fixtures import evidence_ingestion as fx
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.embedders import ClientFactory, CrashingEmbedder, EmbedderCrashed
from tests.fixtures.job_discovery import session_factory

USER = fakes.USER_ID
OLD = ("old-embedder", "1")


@pytest.fixture
def root(tmp_path):
    return fx.allowed_root(tmp_path)


@pytest.fixture
def factory(tmp_path):
    return session_factory(tmp_path / "db")


@pytest.fixture
def local_settings(monkeypatch):
    for name, value in {
        "embedding_provider": "hashing",
        "embedding_model": "",
        "embedding_model_version": "",
        "embedding_base_url": "",
        "cloud_inference_allowed": False,
        "llm_gateway_config": "",
    }.items():
        monkeypatch.setattr(bootstrap.settings, name, value)


def _chunks(factory) -> dict[str, tuple[str, str, bool]]:
    """evidence id -> (model, version, retired)."""
    session = factory()
    try:
        return {
            str(row.id): (row.embedding_model, row.embedding_version, row.retired_at is not None)
            for row in session.query(EvidenceChunkModel)
        }
    finally:
        session.close()


def _documents(factory) -> list[tuple[str, str]]:
    session = factory()
    try:
        return [
            (row.embedding_model, row.embedding_version) for row in session.query(DocumentModel)
        ]
    finally:
        session.close()


def _rows(factory, model) -> list[tuple[UUID, str, str]]:
    session = factory()
    try:
        foreign_key = (
            "job_posting_id" if model is JobPostingEmbeddingModel else "communication_event_id"
        )
        return sorted(
            (getattr(row, foreign_key), row.embedding_model, row.embedding_version)
            for row in session.query(model)
        )
    finally:
        session.close()


def _cited(factory) -> list[str]:
    session = factory()
    try:
        from personalos.persistence.models import ArtifactVersionModel

        return [item["ref"] for row in session.query(ArtifactVersionModel) for item in row.evidence]
    finally:
        session.close()


async def _seed(root, factory) -> dict[str, list[str]]:
    """Evidence on the old embedder, one chunk cited and then retired, plus a posting and a message."""
    old = fx.CountingEmbedder(*OLD)
    first = await fx.ingestor(root, factory, old).ingest(USER, "resume.md")
    ids = [str(item.evidence_id) for item in first.document.evidence]
    fx.cite(factory, USER, ids[:3])
    # An edit drops the first cited chunk, which is retired rather than deleted.
    resume = root / "resume.md"
    resume.write_text(
        resume.read_text().replace("Backend engineer with eight", "Engineer with eight")
    )
    await fx.ingestor(root, factory, old).ingest(USER, "resume.md")

    session = factory()
    try:
        posting = session.query(JobPostingModel).one()
        posting.normalized_json = fakes.posting().model_dump(mode="json", exclude={"raw"})
        application = session.query(ApplicationModel).one()
        events = []
        for index, (subject, summary) in enumerate(
            [("Interview invite", "Pick a time for a call"), ("", "")]
        ):
            event = CommunicationEventModel(
                application_id=application.id,
                classification="interview_invite",
                provider_message_id=f"m-{index}",
                dedupe_key=f"m-{index}",
                occurred_at=datetime(2026, 11, 1),
                metadata_json={"subject": subject, "summary": summary},
            )
            session.add(event)
            events.append(event)
        session.flush()
        vector = HashingEmbedder().vector("old vector")
        session.add(
            JobPostingEmbeddingModel(
                job_posting_id=posting.id,
                embedding=vector,
                embedding_model=OLD[0],
                embedding_version=OLD[1],
            )
        )
        for event in events:
            session.add(
                MessageEmbeddingModel(
                    communication_event_id=event.id,
                    embedding=vector,
                    embedding_model=OLD[0],
                    embedding_version=OLD[1],
                )
            )
        session.commit()
    finally:
        session.close()
    return {"cited": ids[:3]}


# --- Acceptance: swapping the embedder is configuration only ----------------------------------


async def test_swapping_the_configured_embedder_needs_no_graph_change_and_tags_new_vectors(
    root, factory, local_settings, monkeypatch
):
    clients = ClientFactory(width=768)

    # The same composition calls the worker and `ingest-resume` make.
    def configured():
        gateway = bootstrap.build_model_gateway(embeddings_client_factory=clients)
        return bootstrap.build_embedder(gateway=gateway)

    before = configured()
    await fx.ingestor(root, factory, before).ingest(USER, "resume.md")
    assert {value[:2] for value in _chunks(factory).values()} == {(HASHING_EMBEDDER_MODEL, "1")}

    # The swap: settings only.
    monkeypatch.setattr(bootstrap.settings, "embedding_provider", "ollama")
    monkeypatch.setattr(bootstrap.settings, "embedding_model", "nomic-embed-text:v1.5")
    after = configured()
    assert (after.model, after.version) == ("nomic-embed-text:v1.5", "v1.5")

    builder = bootstrap.build_artifact_packet_builder(prep.ScriptedDraftWriter({}), after, factory)
    posting = fakes.posting()
    # Stored vectors are still the old model's, and retrieval never mixes models.
    assert await builder.selector.select(posting, USER) == []

    report = await bootstrap.build_reembedder(factory, embedder=after).run()
    assert report.complete

    found = await builder.selector.select(posting, USER)
    assert found, "evidence is retrievable under the new embedder"
    assert {value[:2] for value in _chunks(factory).values()} == {("nomic-embed-text:v1.5", "v1.5")}
    assert _documents(factory) == [("nomic-embed-text:v1.5", "v1.5")]
    assert retrieval_query(posting) in clients.texts

    # Ingesting an edit with the new embedder writes rows tagged with it, and
    # embeds only the new chunk.
    embedded_before_edit = len(clients.texts)
    resume = root / "resume.md"
    resume.write_text(resume.read_text() + "\n## Awards\nBest on-call runbook, 2022.\n")
    result = await fx.ingestor(root, factory, after).ingest(USER, "resume.md")
    assert result.status == IngestionStatus.UPDATED
    assert len(clients.texts) - embedded_before_edit == result.added
    assert {value[:2] for value in _chunks(factory).values()} == {("nomic-embed-text:v1.5", "v1.5")}


# --- Acceptance: resumable, and cited evidence ids survive ----------------------------------


async def test_the_backfill_resumes_after_a_crash_and_keeps_cited_evidence_ids(root, factory):
    seeded = await _seed(root, factory)
    ids_before = set(_chunks(factory))
    retired = {key for key, (_, _, is_retired) in _chunks(factory).items() if is_retired}
    assert retired and retired <= set(seeded["cited"]), "the fixture retires a cited chunk"

    target = HashingEmbedder()
    crashing = CrashingEmbedder(target, crash_on=3)
    store_embedder = bootstrap.build_reembedder(factory, embedder=crashing, batch_size=2)
    with pytest.raises(EmbedderCrashed):
        await store_embedder.run([VectorTable.EVIDENCE])

    # Two batches were committed before the crash, the third was lost whole.
    mid = _chunks(factory)
    assert set(mid) == ids_before
    assert sum(1 for value in mid.values() if value[:2] == ("personalos-hashing", "1")) == 4
    assert _documents(factory) == [OLD]

    recording = fx.CountingEmbedder()
    resumed = await bootstrap.build_reembedder(factory, embedder=recording, batch_size=2).run(
        [VectorTable.EVIDENCE]
    )

    (evidence,) = resumed.tables
    assert evidence.pending == len(ids_before) - 4
    assert evidence.reembedded == evidence.pending
    assert len(recording.calls) == evidence.pending, "nothing committed before the crash is redone"
    after = _chunks(factory)
    assert set(after) == ids_before
    assert {value[:2] for value in after.values()} == {("personalos-hashing", "1")}
    assert {key for key, value in after.items() if value[2]} == retired
    assert set(_cited(factory)) <= set(after), "every cited evidence id still resolves"
    assert _documents(factory) == [("personalos-hashing", "1")]

    # A further run has nothing to do, and an unchanged resume is still a no-op.
    again = await bootstrap.build_reembedder(factory, embedder=recording).run()
    assert [table.reembedded for table in again.tables] == [0, 1, 1]
    unchanged = await fx.ingestor(root, factory, recording).ingest(USER, "resume.md")
    assert unchanged.status == IngestionStatus.UNCHANGED


async def test_postings_and_messages_gain_a_row_for_the_new_model_and_keep_the_old(root, factory):
    await _seed(root, factory)
    embedder = fx.CountingEmbedder()

    report = await bootstrap.build_reembedder(factory, embedder=embedder).run(
        [VectorTable.POSTINGS, VectorTable.MESSAGES]
    )

    postings, messages = report.tables
    assert (postings.pending, postings.reembedded) == (1, 1)
    # The message with no subject or summary has nothing to embed.
    assert (messages.pending, messages.reembedded, messages.skipped) == (2, 1, 1)
    assert retrieval_query(fakes.posting()) in embedder.calls
    assert "Interview invite\nPick a time for a call" in embedder.calls
    assert sorted(row[1:] for row in _rows(factory, JobPostingEmbeddingModel)) == sorted(
        [OLD, ("personalos-hashing", "1")]
    )
    assert sorted(row[1:] for row in _rows(factory, MessageEmbeddingModel)) == sorted(
        [OLD, OLD, ("personalos-hashing", "1")]
    )

    rerun = await bootstrap.build_reembedder(factory, embedder=embedder).run([VectorTable.POSTINGS])
    assert rerun.tables[0].pending == 0


async def test_a_dry_run_counts_and_writes_nothing(root, factory):
    await _seed(root, factory)
    before = _chunks(factory)
    embedder = fx.CountingEmbedder()

    report = await bootstrap.build_reembedder(factory, embedder=embedder).run(dry_run=True)

    assert report.dry_run and not report.complete
    assert [table.pending for table in report.tables] == [len(before), 1, 2]
    assert embedder.calls == []
    assert _chunks(factory) == before


async def test_a_backfill_onto_the_model_already_stored_has_nothing_to_do(root, factory):
    await fx.ingestor(root, factory).ingest(USER, "resume.md")
    embedder = fx.CountingEmbedder()

    report = await bootstrap.build_reembedder(factory, embedder=embedder).run()

    assert [table.pending for table in report.tables] == [0, 0, 0]
    assert report.complete and embedder.calls == []


# --- CLI ------------------------------------------------------------------------------------


def test_the_cli_backfills_and_prints_the_report(root, factory, local_settings, monkeypatch):
    import asyncio

    asyncio.run(_seed(root, factory))
    monkeypatch.setattr(cli_module, "init_db", lambda: None)
    real = bootstrap.build_reembedder
    monkeypatch.setattr(bootstrap, "build_reembedder", lambda **kwargs: real(factory, **kwargs))
    runner = CliRunner()

    dry = runner.invoke(cli, ["reembed", "--only", "evidence", "--dry-run"])
    run = runner.invoke(cli, ["reembed", "--provider", "hashing", "--batch-size", "3"])
    again = runner.invoke(cli, ["reembed", "--only", "evidence"])

    assert dry.exit_code == 0, dry.output
    assert '"dry_run": true' in dry.output
    assert run.exit_code == 0, run.output
    assert '"embedding_model": "personalos-hashing"' in run.output
    assert {value[:2] for value in _chunks(factory).values()} == {("personalos-hashing", "1")}
    assert '"pending": 0' in again.output

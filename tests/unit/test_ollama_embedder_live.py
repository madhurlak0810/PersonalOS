"""Live tests of the default embedder against a real Ollama server on this machine.

Opt-in: every test here is skipped unless `langchain-ollama` is installed,
Ollama answers at `OLLAMA_TEST_URL` (default http://127.0.0.1:11434), and the
model `OLLAMA_TEST_EMBED_MODEL` (default `all-minilm`, 23M parameters and
384 dimensions, small enough for any laptop) has been pulled:

    ollama pull all-minilm
    pytest tests/unit/test_ollama_embedder_live.py

Nothing is faked here. The vectors come from `OllamaEmbeddings` through
`ModelGateway.embedder`, the same path `ingest-resume` and the worker take.
Only loopback is contacted, which the suite's network guard allows.
"""

import json
import math
import os
import urllib.error
import urllib.request

import pytest

from personalos import bootstrap
from personalos.domain.evidence import IngestionStatus
from personalos.models.embedding import EmbeddingSpec, ModelEmbedder, PrivacyTier
from personalos.models.gateway import ModelConfig, ModelGateway
from personalos.persistence.models import EMBEDDING_DIMENSION
from tests.fixtures import evidence_ingestion as fx
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.job_discovery import session_factory

URL = os.environ.get("OLLAMA_TEST_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("OLLAMA_TEST_EMBED_MODEL", "all-minilm")
USER = fakes.USER_ID


def _pulled_models() -> set[str] | None:
    try:
        with urllib.request.urlopen(f"{URL}/api/tags", timeout=2) as response:
            tags = json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    names = {model["name"] for model in tags.get("models", [])}
    return names | {name.split(":", 1)[0] for name in names if name.endswith(":latest")}


def _skip_reason() -> str | None:
    try:
        import langchain_ollama  # noqa: F401
    except ImportError:
        return "langchain-ollama is not installed (pip install 'personalos[local]')"
    models = _pulled_models()
    if models is None:
        return f"no Ollama server at {URL}"
    if MODEL not in models:
        return f"Ollama model {MODEL!r} is not pulled (ollama pull {MODEL})"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


def _cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


@pytest.fixture
def embedder() -> ModelEmbedder:
    """The real default embedder: ModelGateway -> OllamaEmbeddings -> this machine's Ollama."""
    gateway = ModelGateway(ModelConfig(name="rules"))
    built = gateway.embedder(EmbeddingSpec(provider="ollama", model=MODEL, base_url=URL))
    assert isinstance(built, ModelEmbedder)
    return built


async def test_ollama_vectors_are_padded_to_the_column_width(embedder):
    vector = await embedder.embed("Built Python APIs backed by Postgres.")

    assert embedder.tier is PrivacyTier.LOCAL
    assert (embedder.model, embedder.version) == (MODEL, "latest")
    assert len(vector) == EMBEDDING_DIMENSION
    assert any(value != 0.0 for value in vector)
    # all-minilm is 384 wide; the rest is padding.
    if MODEL.startswith("all-minilm"):
        assert vector[384:] == [0.0] * (EMBEDDING_DIMENSION - 384)


async def test_a_batch_matches_single_embeddings(embedder):
    texts = ["Operated Kafka pipelines in production.", "Volunteered at the community garden."]

    batch = await embedder.embed_many(texts)
    singles = [await embedder.embed(text) for text in texts]

    for left, right in zip(batch, singles, strict=True):
        assert _cosine(left, right) == pytest.approx(1.0, abs=1e-3)


async def test_the_model_knows_meaning_the_hashing_embedder_does_not(embedder):
    """'PostgreSQL' and 'Postgres' share no token, so hashing sees no likeness; a model does."""
    query, related, unrelated = await embedder.embed_many(
        [
            "Experience with PostgreSQL databases",
            "Tuned Postgres query plans and added read replicas",
            "Volunteered planting vegetables at the community garden",
        ]
    )

    assert _cosine(query, related) > _cosine(query, unrelated)


async def test_ingest_retrieve_and_reembed_with_the_real_model(embedder, tmp_path):
    root = fx.allowed_root(tmp_path)
    factory = session_factory(tmp_path / "db")

    # Evidence first ingested with the hashing embedder, then moved onto Ollama.
    hashed = await fx.ingestor(root, factory).ingest(USER, "resume.md")
    ids = {item.evidence_id for item in hashed.document.evidence}
    fx.cite(factory, USER, [str(next(iter(ids)))])

    report = await bootstrap.build_reembedder(factory, embedder=embedder, batch_size=3).run()
    assert report.complete
    assert report.tables[0].reembedded == len(ids)

    again = await fx.ingestor(root, factory, embedder).ingest(USER, "resume.md")
    assert again.status == IngestionStatus.UNCHANGED
    assert {item.evidence_id for item in again.document.evidence} == ids
    assert {(item.embedding_model, item.embedding_version) for item in again.document.evidence} == {
        (MODEL, "latest")
    }

    builder = bootstrap.build_artifact_packet_builder(object(), embedder, factory, top_k=3)
    found = await builder.selector.select(
        fakes.posting(
            title="Data Platform Engineer",
            description="You will run streaming pipelines on Kafka and Kubernetes.",
            skills=("kafka", "kubernetes"),
        ),
        USER,
    )
    assert found
    assert "Kafka" in found[0].record.text

"""Tests for the embedders and for how the ModelGateway selects one.

Covers the local default (Ollama), the opt-in hosted embedder (OpenAI) and the
privacy rule between them: with CLOUD_INFERENCE_ALLOWED=false, a hosted
embedder cannot be selected however it is asked for. No test opens a network
connection: the embedding server is `tests.fixtures.embedders.FakeEmbeddingsClient`.
"""

import json
import math

import pytest
from click.testing import CliRunner

from personalos import bootstrap
from personalos import cli as cli_module
from personalos.cli import cli
from personalos.config import Settings
from personalos.models.embedding import (
    DEFAULT_EMBEDDING_DIMENSION,
    HASHING_EMBEDDER_MODEL,
    CloudInferenceNotAllowed,
    EmbeddingSpec,
    HashingEmbedder,
    ModelEmbedder,
    PrivacyTier,
    fit_dimension,
)
from personalos.models.gateway import ModelConfig, ModelGateway
from personalos.persistence.models import EMBEDDING_DIMENSION
from tests.fixtures.embedders import ClientFactory, FakeEmbeddingsClient


def _cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


@pytest.fixture
def local_settings(monkeypatch):
    """The shipped embedding settings, whatever the developer's .env says."""
    for name, value in {
        "embedding_provider": "ollama",
        "embedding_model": "",
        "embedding_model_version": "",
        "embedding_base_url": "",
        "embedding_api_key_env": "OPENAI_API_KEY",
        "cloud_inference_allowed": False,
        "llm_gateway_config": "",
    }.items():
        monkeypatch.setattr(bootstrap.settings, name, value)


# --- Specs and widths ---------------------------------------------------------------------


def test_the_stored_width_is_the_column_width():
    assert DEFAULT_EMBEDDING_DIMENSION == EMBEDDING_DIMENSION


@pytest.mark.parametrize(
    ("spec", "model", "version"),
    [
        (EmbeddingSpec(), "nomic-embed-text", "latest"),
        (EmbeddingSpec(model="nomic-embed-text:v1.5"), "nomic-embed-text:v1.5", "v1.5"),
        (
            EmbeddingSpec(model="registry.local:5000/team/embed"),
            "registry.local:5000/team/embed",
            "latest",
        ),
        (
            EmbeddingSpec(model="mxbai-embed-large", version="2024-03"),
            "mxbai-embed-large",
            "2024-03",
        ),
        (EmbeddingSpec(provider="openai"), "text-embedding-3-small", "latest"),
        (EmbeddingSpec(provider="hashing", model="ignored"), HASHING_EMBEDDER_MODEL, "1"),
    ],
)
def test_a_spec_resolves_the_model_and_version_every_row_is_stored_under(spec, model, version):
    assert (spec.model_name(), spec.model_version()) == (model, version)


def test_privacy_tiers():
    assert EmbeddingSpec(provider="hashing").tier is PrivacyTier.LOCAL
    assert EmbeddingSpec(provider="ollama").tier is PrivacyTier.LOCAL
    assert EmbeddingSpec(provider="openai").tier is PrivacyTier.CLOUD


def test_padding_to_the_column_width_keeps_cosine_similarity_exactly():
    left, right = [0.3, -0.2, 0.9], [0.1, 0.4, 0.7]

    padded_left, padded_right = (fit_dimension(v, 8, model="m") for v in (left, right))

    assert len(padded_left) == 8 and padded_left[3:] == [0.0] * 5
    assert _cosine(padded_left, padded_right) == pytest.approx(_cosine(left, right))


def test_a_model_wider_than_the_column_is_refused_not_truncated():
    with pytest.raises(ValueError, match="returns 9 dimensions"):
        fit_dimension([0.1] * 9, 8, model="too-wide")


# --- The local default ---------------------------------------------------------------------


async def test_the_ollama_embedder_batches_and_pads_to_the_column_width():
    client = FakeEmbeddingsClient(width=768)
    embedder = ModelEmbedder(
        client, model="nomic-embed-text", version="v1.5", tier=PrivacyTier.LOCAL
    )

    vectors = await embedder.embed_many(["Python on Postgres", "Kafka pipelines"])
    single = await embedder.embed("Python on Postgres")

    assert client.requests == [["Python on Postgres", "Kafka pipelines"], ["Python on Postgres"]]
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in vectors)
    assert vectors[0][768:] == [0.0] * (EMBEDDING_DIMENSION - 768)
    assert single == vectors[0]


async def test_an_empty_text_is_refused_before_any_request():
    client = FakeEmbeddingsClient()
    embedder = ModelEmbedder(client, model="m", version="1", tier=PrivacyTier.LOCAL)

    with pytest.raises(ValueError, match="empty"):
        await embedder.embed("   ")
    assert client.requests == []


def test_the_default_embedder_is_the_local_ollama_one(local_settings):
    factory = ClientFactory()

    embedder = bootstrap.build_embedder(
        gateway=bootstrap.build_model_gateway(embeddings_client_factory=factory)
    )

    assert isinstance(embedder, ModelEmbedder)
    assert (embedder.model, embedder.version, embedder.tier) == (
        "nomic-embed-text",
        "latest",
        PrivacyTier.LOCAL,
    )


async def test_the_client_is_built_on_first_use_so_a_worker_starts_without_the_extra():
    factory = ClientFactory()
    gateway = ModelGateway(ModelConfig(name="rules"), embeddings_client_factory=factory)

    embedder = gateway.embedder(EmbeddingSpec(provider="ollama", base_url="http://localhost:11434"))
    assert factory.specs == []

    await embedder.embed("Python")
    await embedder.embed("Postgres")
    assert [spec.base_url for spec in factory.specs] == ["http://localhost:11434"]


def test_settings_and_overrides_reach_the_spec(local_settings, monkeypatch):
    monkeypatch.setattr(bootstrap.settings, "embedding_model", "nomic-embed-text:v1.5")
    monkeypatch.setattr(bootstrap.settings, "embedding_base_url", "http://127.0.0.1:11434")
    gateway = bootstrap.build_model_gateway(embeddings_client_factory=ClientFactory())

    configured = bootstrap.build_embedder(gateway=gateway)
    overridden = bootstrap.build_embedder(model="mxbai-embed-large", version="7", gateway=gateway)

    assert (configured.model, configured.version) == ("nomic-embed-text:v1.5", "v1.5")
    assert (overridden.model, overridden.version) == ("mxbai-embed-large", "7")


def test_an_embedding_block_in_the_model_configuration_wins_over_settings(local_settings, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"name": "local", "embedding": {"provider": "hashing"}}), encoding="utf-8"
    )

    gateway = bootstrap.build_model_gateway(str(path), embeddings_client_factory=ClientFactory())

    assert isinstance(bootstrap.build_embedder(gateway=gateway), HashingEmbedder)
    # An explicit provider is an override, and wins over both.
    assert bootstrap.build_embedder("ollama", gateway=gateway).model == "nomic-embed-text"


# --- The hosted embedder and the privacy rule ------------------------------------------------


def test_with_cloud_inference_disallowed_the_hosted_embedder_cannot_be_selected(local_settings):
    factory = ClientFactory()
    gateway = bootstrap.build_model_gateway(embeddings_client_factory=factory)

    with pytest.raises(CloudInferenceNotAllowed, match="CLOUD_INFERENCE_ALLOWED"):
        bootstrap.build_embedder("openai", gateway=gateway)
    with pytest.raises(CloudInferenceNotAllowed):
        gateway.embedder(EmbeddingSpec(provider="openai", model="text-embedding-3-large"))
    # Refused before a client existed: no text could have been sent.
    assert factory.specs == []


def test_cloud_inference_is_disallowed_by_default():
    assert ModelGateway(ModelConfig(name="rules")).cloud_inference_allowed is False
    assert Settings.model_fields["cloud_inference_allowed"].default is False


def test_a_configuration_file_cannot_smuggle_in_the_hosted_embedder(local_settings, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"name": "hosted", "embedding": {"provider": "openai"}}), encoding="utf-8"
    )
    gateway = bootstrap.build_model_gateway(str(path), embeddings_client_factory=ClientFactory())

    with pytest.raises(CloudInferenceNotAllowed):
        bootstrap.build_embedder(gateway=gateway)


def test_the_environment_setting_cannot_select_it_either(local_settings, monkeypatch):
    monkeypatch.setattr(bootstrap.settings, "embedding_provider", "openai")

    with pytest.raises(CloudInferenceNotAllowed):
        bootstrap.build_embedder()


async def test_with_cloud_inference_allowed_the_hosted_embedder_is_selected(
    local_settings, monkeypatch
):
    monkeypatch.setattr(bootstrap.settings, "cloud_inference_allowed", True)
    factory = ClientFactory(width=EMBEDDING_DIMENSION)
    gateway = bootstrap.build_model_gateway(embeddings_client_factory=factory)

    embedder = bootstrap.build_embedder("openai", gateway=gateway)
    vector = await embedder.embed("Python on Postgres")

    assert embedder.tier is PrivacyTier.CLOUD
    assert (embedder.model, embedder.version) == ("text-embedding-3-small", "latest")
    assert factory.specs[0].api_key_env == "OPENAI_API_KEY"
    assert len(vector) == EMBEDDING_DIMENSION


def test_reembed_refuses_the_hosted_embedder_while_cloud_inference_is_off(
    local_settings, monkeypatch
):
    monkeypatch.setattr(cli_module, "init_db", lambda: None)

    result = CliRunner().invoke(cli, ["reembed", "--provider", "openai"])

    assert result.exit_code != 0
    assert "CLOUD_INFERENCE_ALLOWED" in result.output

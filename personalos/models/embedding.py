"""Embedders: text to vectors, local by default and hosted only by opt-in.

Every embedder here satisfies `personalos.retrieval.artifact_prep.Embedder`:
a `model` and `version` that are written to every vector row, and an async
`embed`. They also have `embed_many`, which the re-embedding backfill
(`personalos.retrieval.reembedding`) uses to send a batch in one request.

- `HashingEmbedder` (`hashing`): a feature-hashing bag of words. It is local and
  deterministic and needs no download, but it knows nothing about meaning. It
  is the offline fallback, and what the quick start and the tests run on.
- `ModelEmbedder` over Ollama (`ollama`): the default. A local embedding model
  such as `nomic-embed-text`, served by Ollama on this machine. Nothing leaves
  the host.
- `ModelEmbedder` over the OpenAI embeddings API (`openai`): hosted. Resume and
  email text is sent to a third party, so `ModelGateway.embedder` refuses it
  unless `CLOUD_INFERENCE_ALLOWED` is set (see `PrivacyTier`).

Which one runs is an `EmbeddingSpec` that `ModelGateway.embedder` resolves.
Nodes and graphs hold only the `Embedder` port, so swapping embedders is a
configuration change. Vectors from different embedders are not comparable,
so a swap also means re-embedding what is stored (`cli reembed`).

Every table stores `DEFAULT_EMBEDDING_DIMENSION`-wide vectors. A narrower
model's output is zero-padded to that width. Padding with zeros changes
neither dot products nor norms, so cosine similarity is exactly what the
model produced. A model wider than the column is refused, because truncating
its vectors would change their meaning.

Provider packages (`langchain-ollama`, `langchain-openai`, from the `local`
extra) are imported only when a client is built, as for the chat models.
"""

import hashlib
import math
import os
import re
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

#: The width of every stored vector: `personalos.persistence.models.EMBEDDING_DIMENSION`,
#: restated because this layer may not import persistence (a test keeps the two equal).
DEFAULT_EMBEDDING_DIMENSION = 1536

EmbeddingProvider = Literal["hashing", "ollama", "openai"]


class PrivacyTier(str, Enum):
    """Where the text an embedder is given ends up."""

    #: Embedded on this machine; the text never leaves it.
    LOCAL = "local"
    #: Sent to a hosted API. Only allowed when `CLOUD_INFERENCE_ALLOWED` is true.
    CLOUD = "cloud"


#: The privacy tier of each embedding provider.
EMBEDDING_PROVIDER_TIERS: dict[str, PrivacyTier] = {
    "hashing": PrivacyTier.LOCAL,
    "ollama": PrivacyTier.LOCAL,
    "openai": PrivacyTier.CLOUD,
}

#: `EMBEDDING_PROVIDER` values this build has an embedder for.
EMBEDDING_PROVIDERS: tuple[str, ...] = tuple(EMBEDDING_PROVIDER_TIERS)

#: The model each provider uses when the spec names none.
DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
}


class CloudInferenceNotAllowed(PermissionError):
    """A hosted embedder was selected while `CLOUD_INFERENCE_ALLOWED` is false."""


class EmbeddingSpec(BaseModel):
    """Which embedder to use, and how to reach it."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    provider: EmbeddingProvider = "ollama"
    #: The provider's model name. Defaults per provider (`DEFAULT_EMBEDDING_MODELS`).
    #: Ignored by `hashing`.
    model: str | None = None
    #: Stored as `embedding_version` on every row. When unset, it is the model's
    #: tag (`nomic-embed-text:v1.5` gives `v1.5`, an untagged name gives `latest`).
    #: Pin a tag, or set this, so that re-pulling the model under the same name
    #: is a new version rather than a silent change.
    version: str | None = None
    #: Server URL. For `ollama` the default is the client's (localhost:11434).
    base_url: str | None = None
    #: Environment variable holding the API key for `openai`. The key itself is
    #: never stored in a configuration file.
    api_key_env: str | None = None
    #: Passed to the client's constructor as keyword arguments.
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def tier(self) -> PrivacyTier:
        """Where text embedded under this spec goes."""
        return EMBEDDING_PROVIDER_TIERS[self.provider]

    def model_name(self) -> str:
        """The model name rows are keyed by, after defaults."""
        if self.provider == "hashing":
            return HASHING_EMBEDDER_MODEL
        return self.model or DEFAULT_EMBEDDING_MODELS[self.provider]

    def model_version(self) -> str:
        """The version rows are stored with, after defaults."""
        if self.version:
            return self.version
        if self.provider == "hashing":
            return HASHING_EMBEDDER_VERSION
        name = self.model_name()
        # `registry/namespace/model:tag`: the tag follows the last colon after the last slash.
        tail = name.rsplit("/", 1)[-1]
        return tail.rsplit(":", 1)[1] if ":" in tail else "latest"

    def label(self) -> str:
        """`provider:model@version`."""
        return f"{self.provider}:{self.model_name()}@{self.model_version()}"


def fit_dimension(vector: Sequence[float], dimension: int, *, model: str) -> list[float]:
    """`vector`, zero-padded to `dimension`. Raises `ValueError` if it is wider.

    Padding keeps every cosine similarity unchanged. Truncating would not, so
    a wider vector is refused rather than cut.
    """
    values = [float(value) for value in vector]
    if not values:
        raise ValueError(f"embedder {model} returned an empty vector")
    if len(values) > dimension:
        raise ValueError(
            f"embedder {model} returns {len(values)} dimensions; the vector columns hold "
            f"{dimension}. Pick a model of at most {dimension} dimensions, or widen the "
            f"columns with a migration"
        )
    return values + [0.0] * (dimension - len(values))


# --- Hashing --------------------------------------------------------------------------

#: `Embedder.model` for the hashing embedder, as stored on every vector row.
HASHING_EMBEDDER_MODEL = "personalos-hashing"

#: Bump when the tokenization or hashing changes: old vectors stop being comparable.
HASHING_EMBEDDER_VERSION = "1"

#: Weight of an adjacent-word pair relative to a single word.
_BIGRAM_WEIGHT = 0.5

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.]*[a-z0-9+#]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens, keeping `c++`, `c#`, `node.js` and `3.11` whole."""
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    """Deterministic, local text embedding by feature hashing.

    Each word, and each pair of adjacent words, is hashed to one of `dimension`
    buckets with a sign taken from the hash, and the counts are L2-normalized.
    Texts that share words get similar vectors, so retrieval ranks by lexical
    overlap. "Postgres" and "PostgreSQL" are unrelated to it.
    """

    model = HASHING_EMBEDDER_MODEL
    version = HASHING_EMBEDDER_VERSION
    dimension = DEFAULT_EMBEDDING_DIMENSION
    tier = PrivacyTier.LOCAL

    def vector(self, text: str) -> list[float]:
        """The embedding of `text`; raises `ValueError` if it has no words."""
        tokens = tokenize(text)
        if not tokens:
            raise ValueError("nothing to embed: the text has no words")
        features = [(token, 1.0) for token in tokens]
        features += [
            (f"{left} {right}", _BIGRAM_WEIGHT)
            for left, right in zip(tokens, tokens[1:], strict=False)
        ]

        values = [0.0] * self.dimension
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            values[bucket] += sign * weight
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            # Every feature cancelled out against another. This is very rare,
            # and a zero vector has no cosine similarity to anything.
            values[0], norm = 1.0, 1.0
        return [value / norm for value in values]

    async def embed(self, text: str) -> list[float]:
        """`Embedder`: the embedding of `text`."""
        return self.vector(text)

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """The embeddings of `texts`, in order."""
        return [self.vector(text) for text in texts]


# --- Model-backed ---------------------------------------------------------------------


class EmbeddingsClient(Protocol):
    """The part of a LangChain `Embeddings` that this module uses.

    `OllamaEmbeddings` and `OpenAIEmbeddings` satisfy it, and so does a test
    double.
    """

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in order."""
        ...


#: Builds the client for a spec whose provider is a model (not `hashing`).
EmbeddingsClientFactory = Callable[[EmbeddingSpec], EmbeddingsClient]


class ModelEmbedder:
    """An embedding model behind an `EmbeddingsClient`, padded to the column width."""

    def __init__(
        self,
        client: EmbeddingsClient | Callable[[], EmbeddingsClient],
        *,
        model: str,
        version: str,
        tier: PrivacyTier,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ):
        """Bind the client, and the name and version the vectors are stored under.

        `client` may be a zero-argument callable that builds it. It is then
        called on the first embedding, so a process that never embeds (a
        worker with nothing to ingest) does not need the provider package.
        """
        if client is None:
            raise ValueError("ModelEmbedder requires an embeddings client")
        if not model or not version:
            raise ValueError("ModelEmbedder requires a model name and version")
        self._client = client if hasattr(client, "aembed_documents") else None
        self._build_client = None if self._client is not None else client
        self.model = model
        self.version = version
        self.tier = tier
        self.dimension = dimension

    @property
    def client(self) -> EmbeddingsClient:
        """The embeddings client, built on first use."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """The embeddings of `texts`, in order, one request for the lot."""
        texts = list(texts)
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("nothing to embed: a text is empty")
        vectors = await self.client.aembed_documents(texts)
        if len(vectors) != len(texts):
            raise ValueError(
                f"embedder {self.model} returned {len(vectors)} vectors for {len(texts)} texts"
            )
        return [fit_dimension(vector, self.dimension, model=self.model) for vector in vectors]

    async def embed(self, text: str) -> list[float]:
        """`Embedder`: the embedding of `text`."""
        return (await self.embed_many([text]))[0]


def _missing(provider: str, package: str) -> ImportError:
    return ImportError(
        f"the '{provider}' embedding provider requires {package} (the 'local' extra): "
        f"pip install 'personalos[local]'"
    )


def default_embeddings_client_factory(spec: EmbeddingSpec) -> EmbeddingsClient:
    """Build the LangChain embeddings client a spec names."""
    params = dict(spec.params)
    model = spec.model_name()
    if spec.base_url:
        params["base_url"] = spec.base_url
    if spec.provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise _missing(spec.provider, "langchain-ollama") from exc
        return OllamaEmbeddings(model=model, **params)

    if spec.provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise _missing(spec.provider, "langchain-openai") from exc
        key_env = spec.api_key_env or "OPENAI_API_KEY"
        if not os.environ.get(key_env):
            raise ValueError(f"the 'openai' embedding provider needs an API key in ${key_env}")
        params["api_key"] = os.environ[key_env]
        # The v3 models can return vectors of any width up to their native one.
        if model.startswith("text-embedding-3") and "dimensions" not in params:
            params["dimensions"] = DEFAULT_EMBEDDING_DIMENSION
        return OpenAIEmbeddings(model=model, **params)

    raise ValueError(f"embedding provider '{spec.provider}' is not backed by a model")


__all__ = [
    "DEFAULT_EMBEDDING_DIMENSION",
    "EmbeddingProvider",
    "PrivacyTier",
    "EMBEDDING_PROVIDER_TIERS",
    "EMBEDDING_PROVIDERS",
    "DEFAULT_EMBEDDING_MODELS",
    "CloudInferenceNotAllowed",
    "EmbeddingSpec",
    "fit_dimension",
    "HASHING_EMBEDDER_MODEL",
    "HASHING_EMBEDDER_VERSION",
    "tokenize",
    "HashingEmbedder",
    "EmbeddingsClient",
    "EmbeddingsClientFactory",
    "ModelEmbedder",
    "default_embeddings_client_factory",
]

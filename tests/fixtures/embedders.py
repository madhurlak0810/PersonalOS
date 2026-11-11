"""Stand-ins for an embedding server, and an embedder that dies mid-run.

`FakeEmbeddingsClient` answers `aembed_documents` the way `OllamaEmbeddings` or
`OpenAIEmbeddings` would, with no network: deterministic feature-hashing
vectors of a chosen width (768 by default, like `nomic-embed-text`), so the
zero-padding to the column width is exercised. `ClientFactory` is the
`embeddings_client_factory` seam of `ModelGateway`, and records which specs it
was asked to build and when.
"""

from collections.abc import Sequence

from personalos.models.embedding import EmbeddingSpec, HashingEmbedder


class FakeEmbeddingsClient:
    """An embeddings server in a box: hashing vectors `width` wide."""

    def __init__(self, width: int = 768):
        self._hashing = HashingEmbedder()
        self._hashing.dimension = width
        self.requests: list[list[str]] = []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(list(texts))
        return [self._hashing.vector(text) for text in texts]


class ClientFactory:
    """`EmbeddingsClientFactory` that builds `FakeEmbeddingsClient`s and records the specs."""

    def __init__(self, width: int = 768):
        self.width = width
        self.specs: list[EmbeddingSpec] = []
        self.clients: list[FakeEmbeddingsClient] = []

    def __call__(self, spec: EmbeddingSpec) -> FakeEmbeddingsClient:
        self.specs.append(spec)
        client = FakeEmbeddingsClient(self.width)
        self.clients.append(client)
        return client

    @property
    def texts(self) -> list[str]:
        """Every text any client built here was asked to embed."""
        return [text for client in self.clients for request in client.requests for text in request]


class EmbedderCrashed(RuntimeError):
    """What `CrashingEmbedder` raises: the process dying mid-backfill."""


class CrashingEmbedder:
    """Wraps an embedder and raises on its `crash_on`-th batch, as a killed process would."""

    def __init__(self, inner, *, crash_on: int):
        self.inner = inner
        self.model = inner.model
        self.version = inner.version
        self.crash_on = crash_on
        self.batches = 0

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_many([text]))[0]

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches += 1
        if self.batches == self.crash_on:
            raise EmbedderCrashed(f"killed during batch {self.batches}")
        return await self.inner.embed_many(texts)

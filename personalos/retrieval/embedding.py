"""The embedder this build ships with until real ones arrive (Phase J6).

`HashingEmbedder` is a feature-hashing bag of words: each word and each pair
of adjacent words is hashed to one of `EMBEDDING_DIMENSION` buckets with a
hash-derived sign, and the counts are L2-normalized. Texts that share words
get similar vectors, so nearest-evidence retrieval ranks by lexical overlap.
It knows nothing about meaning -- "Postgres" and "PostgreSQL" are unrelated to
it -- but it is local, deterministic, needs no model download and no network,
and its vectors fit the `evidence_chunks.embedding` column as they are.

It satisfies `personalos.retrieval.artifact_prep.Embedder`, the Protocol J6's
local and hosted embedders implement too. Vectors from different embedders
are not comparable, so every row records `model` and `version`, and swapping
embedders means re-ingesting: `ingest-resume` does that for an unchanged file
when the embedder on record differs from the configured one.
"""

import hashlib
import math
import re

from personalos.persistence.models import EMBEDDING_DIMENSION

#: `Embedder.model` for this embedder, as stored on every vector row.
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
    """Deterministic, local text embedding by feature hashing."""

    model = HASHING_EMBEDDER_MODEL
    version = HASHING_EMBEDDER_VERSION
    dimension = EMBEDDING_DIMENSION

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
            # Every feature cancelled out against another; vanishingly rare,
            # and a zero vector has no cosine similarity to anything.
            values[0], norm = 1.0, 1.0
        return [value / norm for value in values]

    async def embed(self, text: str) -> list[float]:
        """`Embedder`: the embedding of `text`."""
        return self.vector(text)


__all__ = [
    "HASHING_EMBEDDER_MODEL",
    "HASHING_EMBEDDER_VERSION",
    "tokenize",
    "HashingEmbedder",
]

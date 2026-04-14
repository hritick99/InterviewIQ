"""Sentence-level embeddings using sentence-transformers all-MiniLM-L6-v2.

Uses hf-mirror.com as a fallback if huggingface.co is unreachable.
"""

from __future__ import annotations

import os

# Use HuggingFace mirror if set (handles network-restricted environments)
if not os.environ.get("HF_ENDPOINT"):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors, one per text."""
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]

"""ChromaDB vector store with strict tenant isolation.

Tenant isolation is enforced at the query level via metadata filters.
A query for tenant_id="A" CANNOT return documents from tenant_id="B".
"""

from __future__ import annotations

from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings as app_settings

COLLECTION_NAME = "interview_segments"

_client: Optional[chromadb.ClientAPI] = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=app_settings.CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_segment(
    embedding_id: str,
    embedding: list[float],
    document: str,
    metadata: dict,
) -> None:
    """Store a segment embedding. metadata MUST include tenant_id."""
    col = _get_collection()
    col.add(
        ids=[embedding_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata],
    )


def vector_search(
    query_embedding: list[float],
    tenant_id: str,
    top_k: int = 10,
) -> list[dict]:
    """Return top_k segments for this tenant only."""
    col = _get_collection()
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"tenant_id": tenant_id},
        include=["documents", "metadatas", "distances"],
    )
    if not results["ids"] or not results["ids"][0]:
        return []

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        hits.append({
            "id": doc_id,
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1.0 - results["distances"][0][i],  # cosine similarity
        })
    return hits


def get_all_documents_for_tenant(tenant_id: str) -> list[dict]:
    """Fetch all raw documents for a tenant (used for BM25 index building)."""
    col = _get_collection()
    results = col.get(
        where={"tenant_id": tenant_id},
        include=["documents", "metadatas"],
    )
    if not results["ids"]:
        return []
    return [
        {"id": doc_id, "document": doc, "metadata": meta}
        for doc_id, doc, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        )
    ]


def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    tenant_id: str,
    top_k: int = 5,
) -> list[dict]:
    """Reciprocal Rank Fusion of vector search + BM25 keyword search."""
    from rank_bm25 import BM25Okapi
    from app.observability.logger import record_cache_hit

    # Vector search
    vector_hits = vector_search(query_embedding, tenant_id, top_k=top_k * 2)

    # BM25 over all tenant documents
    all_docs = get_all_documents_for_tenant(tenant_id)
    record_cache_hit(bool(all_docs))  # approximate cache metric

    if not all_docs:
        return vector_hits[:top_k]

    corpus = [d["document"].split() for d in all_docs]
    bm25 = BM25Okapi(corpus)
    query_tokens = query_text.split()
    bm25_scores = bm25.get_scores(query_tokens)

    # Map doc_id -> bm25 rank
    bm25_ranked = sorted(
        enumerate(bm25_scores), key=lambda x: x[1], reverse=True
    )
    bm25_rank_map = {all_docs[idx]["id"]: rank + 1 for rank, (idx, _) in enumerate(bm25_ranked)}

    # RRF fusion
    k = 60  # standard RRF constant
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {h["id"]: h for h in vector_hits}
    for doc in all_docs:
        doc_map.setdefault(doc["id"], doc)

    for rank, hit in enumerate(vector_hits):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

    for doc_id, bm25_rank in bm25_rank_map.items():
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + bm25_rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [doc_map[doc_id] for doc_id, _ in fused if doc_id in doc_map]

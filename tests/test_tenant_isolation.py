"""Test: Tenant A query must NEVER return Tenant B segments.

This test fails if tenant isolation is broken.
"""

import pytest
import uuid

from app.db.vector_store import add_segment, vector_search
from app.pipeline.embedding import embed_one


def test_tenant_isolation():
    tenant_a = f"tenant_a_{uuid.uuid4().hex[:6]}"
    tenant_b = f"tenant_b_{uuid.uuid4().hex[:6]}"

    text_a = "Tenant A candidate discussed advanced Kubernetes cluster management and node autoscaling."
    text_b = "Tenant B candidate talked about React frontend development and UI performance."

    emb_a = embed_one(text_a)
    emb_b = embed_one(text_b)

    id_a = f"seg_a_{uuid.uuid4().hex}"
    id_b = f"seg_b_{uuid.uuid4().hex}"

    add_segment(id_a, emb_a, text_a, {"tenant_id": tenant_a, "candidate_id": "cand_a", "segment_index": 0, "topic": "kubernetes"})
    add_segment(id_b, emb_b, text_b, {"tenant_id": tenant_b, "candidate_id": "cand_b", "segment_index": 0, "topic": "frontend"})

    # Query as Tenant A using Tenant B's content topic — must return zero Tenant B results
    query_emb = embed_one("React frontend development")
    results_a = vector_search(query_emb, tenant_a, top_k=10)

    for hit in results_a:
        assert hit["metadata"]["tenant_id"] == tenant_a, (
            f"ISOLATION BREACH: Tenant A query returned segment from tenant "
            f"{hit['metadata']['tenant_id']!r}"
        )

    # Verify Tenant B can find its own data
    results_b = vector_search(query_emb, tenant_b, top_k=10)
    assert any(h["id"] == id_b for h in results_b), "Tenant B should be able to find its own segment"

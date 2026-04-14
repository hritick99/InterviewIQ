"""GET /query/stream — genuine token-by-token SSE streaming with source citations."""

import json
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from app.auth.jwt_handler import extract_tenant_from_header
from app.db.vector_store import hybrid_search
from app.llm.base import get_llm_client
from app.pipeline.embedding import embed_one

router = APIRouter()

SYSTEM_PROMPT = """You are an expert recruitment analyst.
Answer the recruiter's question using ONLY the provided interview transcript segments.
Do not hallucinate candidate details. If the answer is not in the provided segments, say so.
Be concise and cite candidate IDs when making specific claims."""


@router.get("/query/stream")
async def query_stream(
    question: str = Query(..., description="Recruiter's natural-language question"),
    authorization: Optional[str] = Header(default=None),
):
    """SSE endpoint — streams LLM answer token-by-token with source citations."""
    # Validate JWT on SSE handshake
    token_data = extract_tenant_from_header(authorization)
    tenant_id = token_data["tenant_id"]

    # Embed query
    query_embedding = embed_one(question)

    # Hybrid retrieval (vector + BM25, tenant-isolated)
    hits = hybrid_search(
        query_text=question,
        query_embedding=query_embedding,
        tenant_id=tenant_id,
        top_k=5,
    )

    if not hits:
        async def empty_stream():
            yield {"data": json.dumps({"token": "No relevant interview segments found for this query.", "citations": []})}
        return EventSourceResponse(empty_stream())

    # Build context with citations
    citations = []
    context_parts = []
    for hit in hits:
        meta = hit.get("metadata", {})
        candidate_id = meta.get("candidate_id", "unknown")
        segment_index = meta.get("segment_index", 0)
        citations.append({"candidate_id": candidate_id, "segment_index": segment_index})
        context_parts.append(
            f"[Candidate: {candidate_id}, Segment: {segment_index}]\n{hit['document']}"
        )

    context = "\n\n---\n\n".join(context_parts)
    prompt = f"Interview segments:\n\n{context}\n\nRecruiter question: {question}"

    llm = get_llm_client()

    async def event_generator():
        # Send citations as first event
        yield {"data": json.dumps({"citations": citations, "token": ""})}

        # Stream answer tokens
        async for token in llm.stream(prompt=prompt, system=SYSTEM_PROMPT, task_type="retrieval_rerank"):
            yield {"data": json.dumps({"token": token, "citations": []})}

        yield {"data": json.dumps({"token": "", "done": True, "citations": citations})}

    return EventSourceResponse(event_generator())

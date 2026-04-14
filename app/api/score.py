"""POST /score/{candidate_id} — run LangGraph scoring workflow."""

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth.jwt_handler import get_current_tenant
from app.db.database import get_db
from app.db.models import Candidate, InterviewSegment, Tenant

router = APIRouter()


@router.post("/score/{candidate_id}")
async def score_candidate(
    candidate_id: str,
    db: Session = Depends(get_db),
    token_data: dict = Depends(get_current_tenant),
):
    tenant_id = token_data["tenant_id"]

    candidate = (
        db.query(Candidate)
        .filter(Candidate.id == candidate_id, Candidate.tenant_id == tenant_id)
        .first()
    )
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    segments = (
        db.query(InterviewSegment)
        .filter(InterviewSegment.candidate_id == candidate_id)
        .order_by(InterviewSegment.segment_index)
        .all()
    )

    if not segments:
        raise HTTPException(status_code=422, detail="No segments found for this candidate")

    # Load tenant rubric
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    rubric = json.loads(tenant.rubric_json) if tenant and tenant.rubric_json else {}

    segment_dicts = [
        {
            "id": s.id,
            "segment_index": s.segment_index,
            "topic": s.topic,
            "skills_mentioned": json.loads(s.skills_mentioned or "[]"),
            "sentiment": s.sentiment,
            "confidence_in_answer": s.confidence_in_answer,
            "raw_text": s.raw_text,
        }
        for s in segments
    ]

    from app.scoring.agent import run_scoring_agent

    state = await run_scoring_agent(
        candidate_id=candidate_id,
        tenant_id=tenant_id,
        segments=segment_dicts,
        rubric=rubric,
    )

    return state

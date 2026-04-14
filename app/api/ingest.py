"""POST /ingest/audio — upload audio, run pipeline in background, return job_id.
GET  /ingest/status/{job_id} — check job status.
POST /auth/token — issue JWT for a tenant (POC convenience endpoint).
"""

import json
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.auth.jwt_handler import create_token, get_current_tenant
from app.db.database import get_db
from app.db.models import AudioIngestionLog, Candidate, InterviewSegment, Job, Tenant
from app.dedup.fuzzy_match import check_duplicate
from app.llm.base import get_llm_client
from app.observability.logger import log_audio_ingestion, set_queue_depth

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
UPLOAD_DIR = Path("tmp_audio")
UPLOAD_DIR.mkdir(exist_ok=True)

_queue_depth = 0


@router.post("/auth/token")
def issue_token(tenant_id: str = Form(...), user_id: str = Form(...), db: Session = Depends(get_db)):
    """POC: issue a JWT. In production this would require real auth."""
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id!r} not found")
    return {"access_token": create_token(tenant_id, user_id), "token_type": "bearer"}


@router.post("/ingest/audio")
async def ingest_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    candidate_name: str = Form(...),
    interview_date: str = Form(default=""),
    db: Session = Depends(get_db),
    token_data: dict = Depends(get_current_tenant),
):
    tenant_id = token_data["tenant_id"]
    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type {ext}")

    # Check for duplicate candidate before creating job
    dedup_result = check_duplicate(
        name=candidate_name,
        interview_date=interview_date,
        tenant_id=tenant_id,
        db_session=db,
    )
    if dedup_result["is_duplicate"]:
        return {
            "warning": "possible_duplicate",
            "message": f"Candidate may already exist as '{dedup_result['matched_name']}' "
                       f"(confidence {dedup_result['confidence_score']}%). "
                       "Use force=true to ingest anyway (not implemented in POC).",
            "matched_candidate_id": dedup_result["matched_candidate_id"],
            "confidence_score": dedup_result["confidence_score"],
        }

    # Save temp file
    job_id = str(uuid.uuid4())
    audio_path = UPLOAD_DIR / f"{job_id}{ext}"
    content = await file.read()
    audio_path.write_bytes(content)

    # Create job record
    job = Job(id=job_id, tenant_id=tenant_id, status="pending", audio_filename=file.filename, candidate_name=candidate_name)
    db.add(job)
    db.commit()

    global _queue_depth
    _queue_depth += 1
    set_queue_depth(_queue_depth)

    background_tasks.add_task(
        _run_pipeline,
        job_id=job_id,
        audio_path=str(audio_path),
        candidate_name=candidate_name,
        interview_date=interview_date,
        tenant_id=tenant_id,
        file_size=len(content),
    )

    return {"job_id": job_id, "status": "pending"}


@router.get("/ingest/status/{job_id}")
def job_status(job_id: str, db: Session = Depends(get_db), token_data: dict = Depends(get_current_tenant)):
    tenant_id = token_data["tenant_id"]
    job = db.query(Job).filter(Job.id == job_id, Job.tenant_id == tenant_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result = {"job_id": job_id, "status": job.status}
    if job.candidate:
        result["candidate_id"] = job.candidate.id
    if job.error_message:
        result["error"] = job.error_message
    return result


async def _run_pipeline(
    job_id: str,
    audio_path: str,
    candidate_name: str,
    interview_date: str,
    tenant_id: str,
    file_size: int,
) -> None:
    """Full ingestion pipeline: transcribe → segment → extract → embed → store."""
    from app.db.database import SessionLocal
    from app.db.vector_store import add_segment
    from app.pipeline.embedding import embed
    from app.pipeline.extraction import extract_segment_metadata, strip_pii_for_vector
    from app.pipeline.segmentation import segment
    from app.pipeline.transcription import transcribe

    db = SessionLocal()
    global _queue_depth

    try:
        # Update job status
        job = db.query(Job).filter(Job.id == job_id).first()
        job.status = "processing"
        db.commit()

        # 1. Transcribe
        t_start = time.perf_counter()
        transcription, transcription_latency = transcribe(audio_path)
        t_latency = time.perf_counter() - t_start

        # 2. Segment
        segments = segment(transcription.text)

        # 3. Extract metadata per segment
        llm = get_llm_client()
        texts = [s.text for s in segments]

        # 4. Embed all at once
        e_start = time.perf_counter()
        embeddings = embed(texts)
        embedding_latency = time.perf_counter() - e_start

        # 5. Create candidate record
        candidate = Candidate(
            tenant_id=tenant_id,
            name=candidate_name,
            interview_date=interview_date or None,
            job_id=job_id,
        )
        db.add(candidate)
        db.flush()

        # 6. Extract metadata and store per segment
        for i, (seg, embedding) in enumerate(zip(segments, embeddings)):
            meta, pii_flagged = await extract_segment_metadata(seg.text, llm)
            vector_text = strip_pii_for_vector(seg.text) if pii_flagged else seg.text
            embedding_id = f"{candidate.id}_{i}"

            add_segment(
                embedding_id=embedding_id,
                embedding=embedding,
                document=vector_text,
                metadata={
                    "tenant_id": tenant_id,
                    "candidate_id": candidate.id,
                    "segment_index": i,
                    "topic": meta.topic,
                },
            )

            db_seg = InterviewSegment(
                candidate_id=candidate.id,
                tenant_id=tenant_id,
                segment_index=i,
                topic=meta.topic,
                skills_mentioned=json.dumps(meta.skills_mentioned),
                sentiment=meta.sentiment,
                confidence_in_answer=meta.confidence_in_answer,
                raw_text=seg.text,
                pii_flagged=pii_flagged,
                embedding_id=embedding_id,
            )
            db.add(db_seg)

        job.status = "done"
        db.commit()

        # Log ingestion
        log_audio_ingestion(
            job_id=job_id,
            file_size=file_size,
            duration_seconds=transcription.duration_seconds,
            transcription_engine="whisper",
            transcription_latency=transcription_latency * 1000,
            segment_count=len(segments),
            embedding_latency=embedding_latency * 1000,
            tenant_id=tenant_id,
        )

    except Exception as exc:
        db.rollback()
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(exc)
            db.commit()
    finally:
        # Delete raw audio file after ingestion
        try:
            os.remove(audio_path)
        except OSError:
            pass
        _queue_depth = max(0, _queue_depth - 1)
        set_queue_depth(_queue_depth)
        db.close()

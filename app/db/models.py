import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False, unique=True)
    # JSON string: configurable rubric per tenant
    rubric_json = Column(Text, default="{}")
    dedup_threshold = Column(Float, default=85.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    candidates = relationship("Candidate", back_populates="tenant")


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(String, primary_key=True, default=_uuid)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    name = Column(String, nullable=False)
    interview_date = Column(String, nullable=True)  # ISO date string
    job_id = Column(String, ForeignKey("jobs.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    tenant = relationship("Tenant", back_populates="candidates")
    segments = relationship("InterviewSegment", back_populates="candidate")
    job = relationship("Job", back_populates="candidate")


class InterviewSegment(Base):
    __tablename__ = "interview_segments"

    id = Column(String, primary_key=True, default=_uuid)
    candidate_id = Column(String, ForeignKey("candidates.id"), nullable=False)
    tenant_id = Column(String, nullable=False)
    segment_index = Column(Integer, nullable=False)
    topic = Column(String, default="unknown")
    # JSON array stored as string
    skills_mentioned = Column(Text, default="[]")
    sentiment = Column(String, default="neutral")
    confidence_in_answer = Column(String, default="medium")
    raw_text = Column(Text, nullable=False)
    pii_flagged = Column(Boolean, default=False)
    # ChromaDB document ID for cross-reference
    embedding_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="segments")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=_uuid)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    status = Column(String, default="pending")  # pending|processing|done|failed
    error_message = Column(Text, nullable=True)
    audio_filename = Column(String, nullable=True)
    candidate_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="job", uselist=False)


class LLMCallLog(Base):
    __tablename__ = "llm_call_logs"

    id = Column(String, primary_key=True, default=_uuid)
    model = Column(String, nullable=False)
    task_type = Column(String, nullable=False)  # transcription_analysis|retrieval_rerank|node_scoring|synthesis
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    latency_ms = Column(Float, default=0.0)
    request_id = Column(String, nullable=True)
    retries = Column(Integer, default=0)
    success = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AudioIngestionLog(Base):
    __tablename__ = "audio_ingestion_logs"

    id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, nullable=False)
    file_size = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)
    transcription_engine = Column(String, default="whisper")
    transcription_latency = Column(Float, default=0.0)
    segment_count = Column(Integer, default=0)
    embedding_latency = Column(Float, default=0.0)
    tenant_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

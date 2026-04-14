"""Seed script: creates 2 tenants and ingests 8 synthetic interview audio clips.

Usage:
    python seed/seed.py

Generates audio via gTTS, runs the full ingestion pipeline for each clip,
and prints JWT tokens for both tenants.
"""

import asyncio
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


SYNTHETIC_INTERVIEWS = [
    {
        "tenant": "alpha_corp",
        "candidate": "Alice Johnson",
        "date": "2026-01-15",
        "transcript": (
            "I have seven years of software engineering experience specializing in distributed systems. "
            "In my last role at a fintech startup, I designed a real-time payment processing system using Kafka and Kubernetes. "
            "The main challenge was achieving exactly-once delivery semantics at high throughput. "
            "I chose Kafka because its log-based architecture allows replay and audit trails. "
            "Tell me about a time you led a team. I led a team of six engineers to migrate our monolith to microservices. "
            "We used the strangler fig pattern and completed the migration in eight months with zero downtime."
        ),
    },
    {
        "tenant": "alpha_corp",
        "candidate": "Bob Chen",
        "date": "2026-01-16",
        "transcript": (
            "I am a backend engineer with four years of experience in Python and Go. "
            "I have built REST and GraphQL APIs and have experience with PostgreSQL and Redis. "
            "Describe your experience with cloud platforms. I have used AWS extensively, particularly EC2, S3, Lambda, and RDS. "
            "I recently obtained my AWS Solutions Architect Associate certification. "
            "Tell me about a challenging bug you fixed. We had a memory leak in production. "
            "I used heap profiling to identify an object cache that was never evicted. "
            "Fixing it reduced memory usage by forty percent."
        ),
    },
    {
        "tenant": "alpha_corp",
        "candidate": "Carol Davis",
        "date": "2026-01-17",
        "transcript": (
            "I am a machine learning engineer with three years of experience. "
            "I have built recommendation systems using collaborative filtering and neural networks. "
            "I am proficient in Python, TensorFlow, and PyTorch. "
            "How do you approach model deployment? I package models with Docker, serve them via FastAPI, "
            "and use Kubernetes for horizontal scaling. I monitor model drift using custom metrics dashboards. "
            "Tell me about a project you are proud of. I built a real-time fraud detection model that reduced "
            "false positives by thirty percent while maintaining ninety-eight percent recall."
        ),
    },
    {
        "tenant": "alpha_corp",
        "candidate": "David Park",
        "date": "2026-01-18",
        "transcript": (
            "I have two years of experience as a junior developer. I know Python and JavaScript. "
            "I built a todo application in React and a simple API with Flask. "
            "How would you approach a performance problem? I would use browser developer tools to profile. "
            "I am not very experienced with backend performance yet but I am learning. "
            "I am eager to grow and take on more challenging projects."
        ),
    },
    {
        "tenant": "beta_ventures",
        "candidate": "Emma Wilson",
        "date": "2026-02-01",
        "transcript": (
            "I am a principal engineer with twelve years of experience in large-scale systems. "
            "I have designed systems serving fifty million users using microservices, event sourcing, and CQRS. "
            "Walk me through your most complex architecture. I designed a multi-region active-active database setup "
            "using CockroachDB that survived a full region failure with under five seconds of disruption. "
            "The key trade-off was accepting higher write latency for stronger consistency guarantees. "
            "I led architecture reviews across four engineering teams and introduced RFC processes."
        ),
    },
    {
        "tenant": "beta_ventures",
        "candidate": "Frank Liu",
        "date": "2026-02-02",
        "transcript": (
            "I am a DevOps engineer with six years of experience. "
            "I manage Kubernetes clusters on GCP and have built CI/CD pipelines using GitHub Actions and ArgoCD. "
            "Describe your approach to observability. I implement the three pillars: metrics with Prometheus, "
            "logs with structured JSON shipped to Loki, and traces with OpenTelemetry. "
            "I set up SLOs for all critical services and run monthly reliability reviews. "
            "Tell me about an incident you handled. We had a cascading failure caused by a misconfigured HPA. "
            "I diagnosed it through metrics, rolled back, and implemented circuit breakers to prevent recurrence."
        ),
    },
    {
        "tenant": "beta_ventures",
        "candidate": "Grace Kim",
        "date": "2026-02-03",
        "transcript": (
            "I am a full-stack engineer specialising in TypeScript, React, and Node.js. "
            "I have built complex SPAs with real-time features using WebSockets. "
            "How do you ensure frontend performance? I use lighthouse audits, lazy loading, "
            "code splitting, and image optimization. I reduced our largest contentful paint from four seconds to one second. "
            "Describe a time you worked with a difficult stakeholder. A product manager kept changing requirements mid-sprint. "
            "I introduced bi-weekly scope freeze agreements and improved our sprint completion rate from sixty to ninety percent."
        ),
    },
    {
        "tenant": "beta_ventures",
        "candidate": "Henry Brown",
        "date": "2026-02-04",
        "transcript": (
            "I am a security engineer with eight years of experience. "
            "I specialize in application security, penetration testing, and secure SDLC implementation. "
            "What security practices do you implement? I run SAST and DAST in CI pipelines, "
            "conduct threat modeling for new features, and perform quarterly penetration tests. "
            "I implemented OAuth two and OpenID Connect for our authentication system. "
            "Tell me about a security vulnerability you found and fixed. "
            "I discovered an IDOR vulnerability that exposed other users data. "
            "I implemented proper authorization checks and rewrote our permission model."
        ),
    },
]


async def ingest_transcript(
    transcript: str,
    candidate_name: str,
    interview_date: str,
    tenant_id: str,
    audio_path: str,
) -> str:
    """Run the full ingestion pipeline on a synthetic audio file."""
    from app.db.database import SessionLocal
    from app.db.models import Candidate, InterviewSegment, Job
    from app.db.vector_store import add_segment
    from app.pipeline.embedding import embed
    from app.pipeline.extraction import extract_segment_metadata, strip_pii_for_vector
    from app.pipeline.segmentation import segment
    from app.llm.base import get_llm_client
    import json as _json
    import time

    db = SessionLocal()
    try:
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, tenant_id=tenant_id, status="processing", candidate_name=candidate_name)
        db.add(job)
        db.commit()

        segments = segment(transcript)
        texts = [s.text for s in segments]
        embeddings = embed(texts)

        candidate = Candidate(
            tenant_id=tenant_id,
            name=candidate_name,
            interview_date=interview_date,
            job_id=job_id,
        )
        db.add(candidate)
        db.flush()

        llm = get_llm_client()
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
                skills_mentioned=_json.dumps(meta.skills_mentioned),
                sentiment=meta.sentiment,
                confidence_in_answer=meta.confidence_in_answer,
                raw_text=seg.text,
                pii_flagged=pii_flagged,
                embedding_id=embedding_id,
            )
            db.add(db_seg)

        job.status = "done"
        db.commit()
        return candidate.id
    finally:
        db.close()


async def main():
    from gtts import gTTS
    from pathlib import Path

    from app.auth.jwt_handler import create_token
    from app.db.database import create_tables, SessionLocal
    from app.db.models import Tenant

    print("Creating database tables...")
    create_tables()

    db = SessionLocal()
    tenants: dict[str, str] = {}

    try:
        for interview in SYNTHETIC_INTERVIEWS:
            tenant_name = interview["tenant"]
            if tenant_name not in tenants:
                existing = db.query(Tenant).filter(Tenant.name == tenant_name).first()
                if existing:
                    tenants[tenant_name] = existing.id
                else:
                    tenant = Tenant(
                        id=str(uuid.uuid4()),
                        name=tenant_name,
                        rubric_json=json.dumps({
                            "technical_topics": ["distributed systems", "python", "cloud"],
                            "behavioural_topics": ["leadership", "collaboration", "problem solving"],
                            "weights": {"technical": 0.6, "behavioural": 0.4},
                        }),
                    )
                    db.add(tenant)
                    db.commit()
                    tenants[tenant_name] = tenant.id
                    print(f"Created tenant: {tenant_name} ({tenants[tenant_name]})")
    finally:
        db.close()

    # Generate audio and ingest
    audio_dir = Path("tmp_seed_audio")
    audio_dir.mkdir(exist_ok=True)

    for i, interview in enumerate(SYNTHETIC_INTERVIEWS):
        tenant_name = interview["tenant"]
        tenant_id = tenants[tenant_name]
        candidate = interview["candidate"]
        print(f"\n[{i+1}/{len(SYNTHETIC_INTERVIEWS)}] Generating audio for {candidate} ({tenant_name})...")

        audio_path = str(audio_dir / f"interview_{i}.mp3")
        tts = gTTS(text=interview["transcript"], lang="en", slow=False)
        tts.save(audio_path)

        print(f"  Ingesting...")
        candidate_id = await ingest_transcript(
            transcript=interview["transcript"],
            candidate_name=candidate,
            interview_date=interview["date"],
            tenant_id=tenant_id,
            audio_path=audio_path,
        )
        print(f"  Done. Candidate ID: {candidate_id}")

    print("\n" + "=" * 60)
    print("SEED COMPLETE")
    print("=" * 60)
    for tenant_name, tenant_id in tenants.items():
        token = create_token(tenant_id=tenant_id, user_id=f"admin_{tenant_name}")
        print(f"\nTenant: {tenant_name}")
        print(f"  ID   : {tenant_id}")
        print(f"  JWT  : {token}")

    print("\nUse the JWT as: Authorization: Bearer <token>")


if __name__ == "__main__":
    asyncio.run(main())

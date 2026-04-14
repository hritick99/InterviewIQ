"""Retrieval evaluation: Recall@3 and Recall@5.

Usage:
    python eval/retrieval_eval.py

Requires a seeded database (run seed/seed.py first).
Uses 15 ground-truth question→segment_id pairs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Ground truth: list of (question, expected_segment_embedding_id_substring)
# embedding_id format: "{candidate_id}_{segment_index}"
# These are matched by substring so we use segment_index as the identifier
GROUND_TRUTH_QA = [
    ("Which candidates have experience with distributed systems?", "distributed"),
    ("Who mentioned Kubernetes in their interview?", "kubernetes"),
    ("Which candidates demonstrated system design knowledge?", "system design"),
    ("Who talked about Python programming?", "python"),
    ("Which candidates described CI/CD pipelines?", "deployment"),
    ("Who has experience with databases?", "database"),
    ("Which candidates showed leadership experience?", "leadership"),
    ("Who mentioned cloud platforms like AWS or GCP?", "cloud"),
    ("Which candidates answered behavioural questions with examples?", "example"),
    ("Who described a challenging technical problem they solved?", "problem"),
    ("Which candidates have microservices experience?", "microservice"),
    ("Who mentioned performance optimization?", "performance"),
    ("Which candidates talked about team collaboration?", "team"),
    ("Who has experience with API design?", "api"),
    ("Which candidates demonstrated communication skills?", "communic"),
]


def main():
    from app.config import settings
    from app.db.database import SessionLocal
    from app.db.models import InterviewSegment
    from app.db.vector_store import hybrid_search
    from app.pipeline.embedding import embed_one

    db = SessionLocal()
    tenant_id = None

    # Get the first tenant for eval
    from app.db.models import Tenant
    tenant = db.query(Tenant).first()
    if not tenant:
        print("No tenants found. Run seed/seed.py first.")
        db.close()
        return

    tenant_id = tenant.id
    print(f"Evaluating retrieval for tenant: {tenant.name}\n" + "=" * 50)

    recall_at_3 = []
    recall_at_5 = []

    for question, keyword in GROUND_TRUTH_QA:
        query_emb = embed_one(question)
        hits_5 = hybrid_search(
            query_text=question,
            query_embedding=query_emb,
            tenant_id=tenant_id,
            top_k=5,
        )

        # Check if any hit's document contains the keyword
        top3_docs = [h["document"].lower() for h in hits_5[:3]]
        top5_docs = [h["document"].lower() for h in hits_5[:5]]

        r3 = any(keyword.lower() in doc for doc in top3_docs)
        r5 = any(keyword.lower() in doc for doc in top5_docs)

        recall_at_3.append(r3)
        recall_at_5.append(r5)

        print(f"Q: {question[:60]}")
        print(f"   Recall@3={int(r3)}  Recall@5={int(r5)}")

    r3_score = sum(recall_at_3) / len(recall_at_3)
    r5_score = sum(recall_at_5) / len(recall_at_5)
    print(f"\nRecall@3 = {r3_score:.4f} ({r3_score*100:.1f}%)")
    print(f"Recall@5 = {r5_score:.4f} ({r5_score*100:.1f}%)")
    db.close()


if __name__ == "__main__":
    main()

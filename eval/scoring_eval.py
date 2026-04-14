"""Scoring agent evaluation: agreement rate on 5 synthetic transcripts.

Usage:
    python eval/scoring_eval.py

Runs the scoring agent on 5 pre-defined synthetic transcripts and compares
against manually-assigned ground truth hire recommendations.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Ground truth: (transcript_segments, manual_hire_recommendation)
EVAL_CASES = [
    {
        "name": "Strong Yes — Senior Engineer",
        "manual_recommendation": "strong_yes",
        "manual_score": 9.0,
        "segments": [
            {"segment_index": 0, "topic": "system design", "skills_mentioned": ["distributed systems", "kafka", "kubernetes"], "sentiment": "positive", "confidence_in_answer": "high", "raw_text": "I designed a distributed event-driven system using Kafka and Kubernetes that processed 10 million events per day. The key design challenge was ensuring exactly-once delivery semantics across 50 microservices. I chose Kafka because its offset-based replay gives us audit trails and replay capability which relational queues cannot match."},
            {"segment_index": 1, "topic": "leadership", "skills_mentioned": ["mentoring", "code review"], "sentiment": "positive", "confidence_in_answer": "high", "raw_text": "I led a team of 8 engineers for the platform migration. When two senior engineers disagreed on database sharding strategy, I ran a structured ADR process, had both write up trade-offs, and we reached consensus within a week. The result was a 40% latency improvement."},
        ],
    },
    {
        "name": "Yes — Mid-level",
        "manual_recommendation": "yes",
        "manual_score": 7.0,
        "segments": [
            {"segment_index": 0, "topic": "python", "skills_mentioned": ["python", "fastapi", "postgres"], "sentiment": "positive", "confidence_in_answer": "medium", "raw_text": "I have three years of Python experience and have built REST APIs with FastAPI. I understand async programming and have used PostgreSQL with SQLAlchemy for most of my projects. I am still learning distributed systems concepts."},
            {"segment_index": 1, "topic": "problem solving", "skills_mentioned": ["debugging"], "sentiment": "neutral", "confidence_in_answer": "medium", "raw_text": "I tackled a performance issue where our API was slow. I used profiling tools and found an N+1 query issue. I fixed it with eager loading and performance improved by 60%."},
        ],
    },
    {
        "name": "Maybe — Junior with potential",
        "manual_recommendation": "maybe",
        "manual_score": 5.0,
        "segments": [
            {"segment_index": 0, "topic": "basics", "skills_mentioned": ["python", "sql"], "sentiment": "neutral", "confidence_in_answer": "low", "raw_text": "I know Python and SQL from my university projects. I have not worked in production yet but I am eager to learn. I built a simple CRUD application for my thesis."},
            {"segment_index": 1, "topic": "learning", "skills_mentioned": [], "sentiment": "positive", "confidence_in_answer": "low", "raw_text": "I learn quickly and I have been doing online courses on AWS and Docker. I think I can pick up the required skills fast."},
        ],
    },
    {
        "name": "No — Contradictions and surface answers",
        "manual_recommendation": "no",
        "manual_score": 2.0,
        "segments": [
            {"segment_index": 0, "topic": "kubernetes", "skills_mentioned": ["kubernetes"], "sentiment": "positive", "confidence_in_answer": "high", "raw_text": "I have been using Kubernetes in production for five years and I consider myself an expert. I handle all our cluster operations."},
            {"segment_index": 1, "topic": "kubernetes", "skills_mentioned": ["kubernetes"], "sentiment": "neutral", "confidence_in_answer": "low", "raw_text": "I am not sure how pods are scheduled actually. I mostly just run kubectl commands that my team wrote. I have not set up a cluster from scratch myself."},
        ],
    },
    {
        "name": "Strong Yes — Principal Engineer",
        "manual_recommendation": "strong_yes",
        "manual_score": 9.5,
        "segments": [
            {"segment_index": 0, "topic": "architecture", "skills_mentioned": ["microservices", "event sourcing", "cqrs"], "sentiment": "positive", "confidence_in_answer": "high", "raw_text": "I architected a CQRS and event sourcing system for a financial platform processing 500 transactions per second. The trade-off between eventual consistency and strong consistency was the core challenge. I chose eventual consistency for the read models but enforced strong consistency for the write side using a saga pattern for distributed transactions."},
            {"segment_index": 1, "topic": "mentoring", "skills_mentioned": ["technical leadership", "architecture review"], "sentiment": "positive", "confidence_in_answer": "high", "raw_text": "As principal engineer I run weekly architecture review sessions and mentor 15 engineers across three teams. I introduced RFC processes that reduced rework by 35% in the first quarter."},
        ],
    },
]

# Map recommendation to ordinal for score comparison
REC_ORDER = {"strong_yes": 4, "yes": 3, "maybe": 2, "no": 1}


async def run_case(case: dict) -> dict:
    from app.scoring.agent import run_scoring_agent

    state = await run_scoring_agent(
        candidate_id=f"eval_{case['name'][:10]}",
        tenant_id="eval_tenant",
        segments=case["segments"],
        rubric={},
    )

    agent_rec = state["final_verdict"].get("hire_recommendation", "maybe")
    manual_rec = case["manual_recommendation"]

    # Agreement: exact match OR adjacent bucket
    agent_ord = REC_ORDER.get(agent_rec, 2)
    manual_ord = REC_ORDER.get(manual_rec, 2)
    exact_match = agent_rec == manual_rec
    adjacent_match = abs(agent_ord - manual_ord) <= 1

    return {
        "name": case["name"],
        "manual_recommendation": manual_rec,
        "agent_recommendation": agent_rec,
        "manual_score": case["manual_score"],
        "agent_score": state["final_verdict"].get("overall_score", 0),
        "exact_match": exact_match,
        "adjacent_match": adjacent_match,
        "fallback_used": any(e["fallback_used"] for e in state["audit_log"]),
    }


async def main():
    print("Scoring Agent Evaluation\n" + "=" * 50)
    results = []
    for case in EVAL_CASES:
        print(f"\nRunning: {case['name']}...")
        r = await run_case(case)
        results.append(r)
        print(f"  Manual: {r['manual_recommendation']} ({r['manual_score']:.1f})")
        print(f"  Agent:  {r['agent_recommendation']} ({r['agent_score']:.1f})")
        print(f"  Exact match: {r['exact_match']}  Adjacent: {r['adjacent_match']}")
        if r["fallback_used"]:
            print("  ⚠ Fallback was used in at least one node")

    exact_rate = sum(r["exact_match"] for r in results) / len(results)
    adjacent_rate = sum(r["adjacent_match"] for r in results) / len(results)
    print(f"\nExact agreement rate : {exact_rate:.2f} ({exact_rate*100:.0f}%)")
    print(f"Adjacent agreement rate: {adjacent_rate:.2f} ({adjacent_rate*100:.0f}%)")
    print("\nResults ready for SUBMISSION.md")


if __name__ == "__main__":
    asyncio.run(main())

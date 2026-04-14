"""Fuzzy candidate deduplication using rapidfuzz.

Uses token_sort_ratio on (name + interview_date) combined string.
Reports confidence score to caller — never silently merges.
"""

from __future__ import annotations

from typing import Optional

from rapidfuzz import fuzz


def _key(name: str, date: Optional[str]) -> str:
    return f"{name.lower().strip()} {(date or '').strip()}"


def check_duplicate(
    name: str,
    interview_date: Optional[str],
    tenant_id: str,
    threshold: float = 85.0,
    db_session=None,
) -> dict:
    """Check if a candidate likely already exists in this tenant's DB.

    Returns:
        {
            "is_duplicate": bool,
            "matched_candidate_id": "Optional[str]",
            "confidence_score": float,
            "matched_name": "Optional[str]",
        }
    """
    if db_session is None:
        return {"is_duplicate": False, "matched_candidate_id": None, "confidence_score": 0.0, "matched_name": None}

    from app.db.models import Candidate

    existing = (
        db_session.query(Candidate)
        .filter(Candidate.tenant_id == tenant_id)
        .all()
    )

    query_key = _key(name, interview_date)
    best_score = 0.0
    best_match = None

    for candidate in existing:
        candidate_key = _key(candidate.name, candidate.interview_date)
        score = fuzz.token_sort_ratio(query_key, candidate_key)
        if score > best_score:
            best_score = score
            best_match = candidate

    is_dup = best_score >= threshold

    return {
        "is_duplicate": is_dup,
        "matched_candidate_id": best_match.id if is_dup and best_match else None,
        "confidence_score": round(best_score, 2),
        "matched_name": best_match.name if is_dup and best_match else None,
    }

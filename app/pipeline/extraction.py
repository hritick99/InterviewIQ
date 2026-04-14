"""Extract structured metadata from a transcript segment using an LLM call.

Schema:
  topic: str
  skills_mentioned: list[str]
  sentiment: "positive" | "neutral" | "negative"
  confidence_in_answer: "low" | "medium" | "high"

PII detection: candidate names, emails, phone numbers flagged and stripped from vector text.
Prompt injection: instruction-like patterns stripped before LLM call.
"""

import json
import re
from typing import Optional

from pydantic import BaseModel, ValidationError

PII_PATTERNS = [
    re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"),          # Full names
    re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),     # Emails
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),  # Phone
]

INJECTION_PATTERNS = re.compile(
    r"(?:ignore (?:previous|all|above) instructions?|"
    r"disregard (?:previous|all|above)|"
    r"you are now|pretend you are|act as if|"
    r"return all (?:candidate|user|tenant)|"
    r"reveal (?:all|private|confidential))",
    re.IGNORECASE,
)


class SegmentMetadata(BaseModel):
    topic: str = "unknown"
    skills_mentioned: list[str] = []
    sentiment: str = "neutral"
    confidence_in_answer: str = "medium"

    def model_post_init(self, __context) -> None:
        if self.sentiment not in ("positive", "neutral", "negative"):
            self.sentiment = "neutral"
        if self.confidence_in_answer not in ("low", "medium", "high"):
            self.confidence_in_answer = "medium"


def sanitise_for_prompt(text: str) -> str:
    """Strip prompt injection patterns from transcript text."""
    return INJECTION_PATTERNS.sub("[REDACTED]", text)


def detect_pii(text: str) -> bool:
    return any(p.search(text) for p in PII_PATTERNS)


def strip_pii_for_vector(text: str) -> str:
    """Replace PII matches with placeholder so they're not searchable."""
    result = text
    for pattern in PII_PATTERNS:
        result = pattern.sub("[PII]", result)
    return result


SYSTEM_PROMPT = """You are an interview analysis assistant.
Given a transcript segment from a job interview, extract structured metadata.
Respond ONLY with valid JSON matching this schema exactly:
{
  "topic": "<main topic discussed>",
  "skills_mentioned": ["skill1", "skill2"],
  "sentiment": "positive|neutral|negative",
  "confidence_in_answer": "low|medium|high"
}"""


async def extract_segment_metadata(
    segment_text: str,
    llm_client,
    max_retries: int = 3,
) -> tuple[SegmentMetadata, bool]:
    """Return (metadata, pii_flagged). Retries up to max_retries on parse failure."""
    pii_flagged = detect_pii(segment_text)
    safe_text = sanitise_for_prompt(segment_text)

    prompt = f"Interview segment:\n\"\"\"\n{safe_text}\n\"\"\"\n\nExtract the metadata as JSON."

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            raw = await llm_client.complete(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                json_mode=True,
                task_type="transcription_analysis",
            )
            data = json.loads(raw)
            meta = SegmentMetadata(**data)
            return meta, pii_flagged
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            last_error = e

    # Fallback default after all retries exhausted
    return SegmentMetadata(), pii_flagged

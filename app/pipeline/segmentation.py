"""Segment a transcript into answerable chunks.

Strategy (in priority order):
1. Question-word boundary patterns (e.g. "Tell me about", "Describe", "How would you")
2. Sentence boundaries after a minimum token threshold
3. Fallback: fixed 300-token windows

Failure modes documented in SUBMISSION.md:
- When two speakers' speech is merged, the segmenter may treat a very long answer as one segment.
- Short question patterns may over-segment if a candidate repeats phrasing.
"""

import re
from dataclasses import dataclass, field

QUESTION_PATTERNS = re.compile(
    r"(?:^|\.\s+|[?!]\s+)"
    r"(?:tell me|can you|could you|describe|explain|walk me through|"
    r"how (?:would|do|did|have)|what (?:is|are|was|were|do|did)|"
    r"why did|give me an example|have you ever|talk about)",
    re.IGNORECASE,
)

MIN_SEGMENT_TOKENS = 50
MAX_SEGMENT_TOKENS = 300


@dataclass
class Segment:
    text: str
    index: int
    speaker_hint: str = "unknown"
    word_count: int = 0


def _split_sentences(text: str) -> list[str]:
    # Naive sentence splitter — good enough for POC
    return re.split(r"(?<=[.!?])\s+", text.strip())


def segment(transcript_text: str) -> list[Segment]:
    """Split transcript into segments suitable for independent retrieval."""
    if not transcript_text.strip():
        return []

    sentences = _split_sentences(transcript_text)
    segments: list[Segment] = []
    current_sentences: list[str] = []
    current_tokens = 0

    def flush(hint: str = "unknown") -> None:
        nonlocal current_sentences, current_tokens
        if current_sentences:
            text = " ".join(current_sentences).strip()
            segments.append(
                Segment(
                    text=text,
                    index=len(segments),
                    speaker_hint=hint,
                    word_count=len(text.split()),
                )
            )
            current_sentences = []
            current_tokens = 0

    for sentence in sentences:
        tokens = len(sentence.split())
        is_question_boundary = bool(QUESTION_PATTERNS.search(sentence))

        if is_question_boundary and current_tokens >= MIN_SEGMENT_TOKENS:
            flush()

        if current_tokens + tokens > MAX_SEGMENT_TOKENS and current_tokens >= MIN_SEGMENT_TOKENS:
            flush()

        current_sentences.append(sentence)
        current_tokens += tokens

    flush()

    # Fallback: if no segments (very short transcript), treat whole text as one
    if not segments:
        segments.append(
            Segment(
                text=transcript_text.strip(),
                index=0,
                word_count=len(transcript_text.split()),
            )
        )

    return segments

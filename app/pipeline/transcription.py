import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

_whisper_model = None


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        from app.config import settings
        _whisper_model = whisper.load_model(settings.WHISPER_MODEL)
    return _whisper_model


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    text: str
    duration_seconds: float
    word_timestamps: list[WordTimestamp] = field(default_factory=list)
    engine: str = "whisper"


def transcribe(audio_path: Union[str, Path]) -> tuple[TranscriptionResult, float]:
    """Transcribe audio file. Returns (result, latency_seconds)."""
    model = _get_model()
    start = time.perf_counter()

    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        fp16=False,
    )

    latency = time.perf_counter() - start

    word_ts = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            word_ts.append(WordTimestamp(word=w["word"], start=w["start"], end=w["end"]))

    duration = result["segments"][-1]["end"] if result.get("segments") else 0.0

    return TranscriptionResult(
        text=result["text"].strip(),
        duration_seconds=duration,
        word_timestamps=word_ts,
    ), latency

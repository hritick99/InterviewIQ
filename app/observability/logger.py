import json
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_llm_log_file = LOG_DIR / "llm_calls.jsonl"
_audio_log_file = LOG_DIR / "audio_ingestion.jsonl"

# In-memory metrics store (POC: not thread-safe for high concurrency)
_metrics: dict[str, Any] = {
    "endpoint_latencies": defaultdict(list),  # endpoint -> [latency_ms, ...]
    "llm_errors": 0,
    "llm_calls": 0,
    "ingestion_queue_depth": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "recent_llm_logs": deque(maxlen=500),
}


def _write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def log_llm_call(
    model: str,
    task_type: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
    request_id: "Optional[str]" = None,
    retries: int = 0,
    success: bool = True,
) -> None:
    record = {
        "ts": datetime.utcnow().isoformat(),
        "model": model,
        "task_type": task_type,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": round(latency_ms, 2),
        "request_id": request_id or str(uuid.uuid4()),
        "retries": retries,
        "success": success,
    }
    _write_jsonl(_llm_log_file, record)
    _metrics["llm_calls"] += 1
    if not success:
        _metrics["llm_errors"] += 1
    _metrics["recent_llm_logs"].append(record)


def log_audio_ingestion(
    job_id: str,
    file_size: int,
    duration_seconds: float,
    transcription_engine: str,
    transcription_latency: float,
    segment_count: int,
    embedding_latency: float,
    tenant_id: str,
) -> None:
    record = {
        "ts": datetime.utcnow().isoformat(),
        "job_id": job_id,
        "file_size": file_size,
        "duration_seconds": round(duration_seconds, 2),
        "transcription_engine": transcription_engine,
        "transcription_latency": round(transcription_latency, 2),
        "segment_count": segment_count,
        "embedding_latency": round(embedding_latency, 2),
        "tenant_id": tenant_id,
    }
    _write_jsonl(_audio_log_file, record)


def record_endpoint_latency(endpoint: str, latency_ms: float) -> None:
    _metrics["endpoint_latencies"][endpoint].append(latency_ms)
    # Keep only last 1000 per endpoint
    if len(_metrics["endpoint_latencies"][endpoint]) > 1000:
        _metrics["endpoint_latencies"][endpoint] = _metrics["endpoint_latencies"][endpoint][-1000:]


def record_cache_hit(hit: bool) -> None:
    if hit:
        _metrics["cache_hits"] += 1
    else:
        _metrics["cache_misses"] += 1


def set_queue_depth(depth: int) -> None:
    _metrics["ingestion_queue_depth"] = depth


def get_metrics() -> dict:
    import numpy as np

    def percentiles(values: list) -> dict:
        if not values:
            return {"p50": 0, "p95": 0}
        arr = np.array(values)
        return {"p50": round(float(np.percentile(arr, 50)), 2), "p95": round(float(np.percentile(arr, 95)), 2)}

    total = _metrics["llm_calls"]
    errors = _metrics["llm_errors"]
    hits = _metrics["cache_hits"]
    misses = _metrics["cache_misses"]

    return {
        "endpoint_latency_ms": {
            ep: percentiles(lats)
            for ep, lats in _metrics["endpoint_latencies"].items()
        },
        "llm_error_rate": round(errors / total, 4) if total else 0.0,
        "llm_total_calls": total,
        "audio_ingestion_queue_depth": _metrics["ingestion_queue_depth"],
        "retrieval_cache_hit_rate": round(hits / (hits + misses), 4) if (hits + misses) else 0.0,
    }

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.database import create_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield


app = FastAPI(
    title="InterviewIQ POC",
    description="Voice-Driven Candidate Intelligence Platform",
    version="0.1.0",
    lifespan=lifespan,
)

from app.api.ingest import router as ingest_router
from app.api.query import router as query_router
from app.api.score import router as score_router
from app.api.metrics import router as metrics_router

app.include_router(ingest_router, tags=["ingestion"])
app.include_router(query_router, tags=["query"])
app.include_router(score_router, tags=["scoring"])
app.include_router(metrics_router, tags=["observability"])


@app.get("/health")
def health():
    return {"status": "ok"}

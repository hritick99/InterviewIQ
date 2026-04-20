from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

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


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Add HTTPBearer security scheme so Swagger shows a simple token input
    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    # Apply to all endpoints
    for path in schema.get("paths", {}).values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi

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

"""GET /metrics — operational metrics."""

from fastapi import APIRouter

from app.observability.logger import get_metrics

router = APIRouter()


@router.get("/metrics")
def metrics():
    return get_metrics()

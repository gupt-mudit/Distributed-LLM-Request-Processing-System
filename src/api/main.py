from __future__ import annotations

import logging

from fastapi import FastAPI

from .dependencies.services import get_embedding_service
from .routers import health, metrics, process

logger = logging.getLogger(__name__)

app = FastAPI(title="Prompt Processing System")

app.include_router(process.router)
app.include_router(health.router)
app.include_router(metrics.router)


@app.on_event("startup")
async def preload_embedding_model() -> None:
    """Pre-load the embedding model on startup to avoid first-request delay."""
    logger.info("Pre-loading embedding model...")
    try:
        embedding_service = get_embedding_service()
        # Trigger model loading with a dummy string
        embedding_service.embed("preload")
        logger.info("Embedding model loaded successfully")
    except Exception as exc:
        logger.warning(f"Failed to pre-load embedding model: {exc}")

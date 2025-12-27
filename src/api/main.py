from __future__ import annotations

import logging

from fastapi import FastAPI

from .routers import health, metrics, process

logger = logging.getLogger(__name__)

app = FastAPI(title="Prompt Processing System")

app.include_router(process.router)
app.include_router(health.router)
app.include_router(metrics.router)

from __future__ import annotations

import logging
import os

from celery import Celery
from kombu import Queue


def _configure_logging() -> None:
    level_name = os.getenv("SERVICE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


_configure_logging()


def _get_broker_url() -> str:
    return os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")


def _get_result_backend() -> str:
    return os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")


celery_app = Celery(
    "prompt_processing",
    broker=_get_broker_url(),
    backend=_get_result_backend(),
    include=["src.worker.tasks"],
)

celery_app.conf.update(
    task_default_queue="prompt_processing",
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_default_retry_delay=5,
    task_default_rate_limit=None,
)
celery_app.conf.task_queues = (
    Queue("prompt_high"),
    Queue("prompt_normal"),
    Queue("prompt_low"),
)
celery_app.conf.task_default_queue = "prompt_normal"
celery_app.conf.task_default_exchange_type = "direct"
# Note: Task routing is handled dynamically in PromptTaskClient.enqueue()
# based on priority, so we don't hardcode routes here


@celery_app.task(name="health.check")
def health_check() -> str:
    return "ok"

from __future__ import annotations

import logging
from typing import Optional

from celery.result import AsyncResult

# MongoDB models are used in routes, not here
from src.worker import celery_app

logger = logging.getLogger(__name__)


class PromptTaskClient:
    """Wrapper around Celery for prompt processing tasks."""

    PRIORITY_TO_QUEUE = {
        "high": "prompt_high",
        "normal": "prompt_normal",
        "low": "prompt_low",
    }

    def enqueue(
        self,
        *,
        user_id: str,
        prompt_id: str,
        text: str,
        priority: str,
    ) -> AsyncResult:
        queue = self.PRIORITY_TO_QUEUE.get(priority.lower(), "prompt_normal")
        signature = celery_app.signature(
            "prompt.process",
            args=(
                {
                    "user_id": user_id,
                    "prompt_id": prompt_id,
                    "text": text,
                    "priority": priority,
                },
            ),
            queue=queue,
            routing_key=queue,
        )
        result = signature.apply_async(priority=self._priority_to_routing(priority))
        logger.info(
            "Enqueued prompt task",
            extra={
                "user_id": user_id,
                "prompt_id": prompt_id,
                "task_id": result.id,
            },
        )
        return result

    def poll_result(self, task_id: str) -> Optional[dict[str, object]]:
        result = AsyncResult(task_id, app=celery_app)
        if not result.ready():
            return None
        if result.failed():
            raise result.result
        payload = result.result
        assert isinstance(payload, dict)
        return payload

    def _priority_to_routing(self, priority: str) -> int | None:
        mapping = {
            "high": 0,
            "normal": 5,
            "low": 9,
        }
        return mapping.get(priority.lower())


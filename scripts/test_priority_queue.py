from __future__ import annotations

import time
import os
import sys
from dataclasses import dataclass
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import click

from src.api.services import PromptTaskClient
from src.worker.app import celery_app


@dataclass
class TaskHandle:
    priority: str
    prompt_id: str
    queue: str
    enqueued_at: float
    async_result: object
    completed_at: float | None = None
    response_payload: dict | None = None

    @property
    def duration(self) -> float | None:
        if self.completed_at is None:
            return None
        return self.completed_at - self.enqueued_at


def enqueue_task(priority: str, index: int) -> TaskHandle:
    prompt_id = f"{priority}-{index}"
    payload = {
        "user_id": "priority-test-user",
        "prompt_id": prompt_id,
        "text": f"Priority queue demo #{priority}-{index}",
        "priority": priority,
    }
    queue = PromptTaskClient.PRIORITY_TO_QUEUE.get(priority, "prompt_normal")
    signature = celery_app.signature(
        "prompt.process",
        args=(payload,),
        queue=queue,
        routing_key=queue,
    )
    result = signature.apply_async()
    return TaskHandle(priority, prompt_id, queue, time.perf_counter(), async_result=result)


@click.command()
@click.option("--low-count", default=6, show_default=True)
@click.option("--timeout", default=60, show_default=True, help="Seconds to wait for each task to finish.")
def main(low_count: int, timeout: float) -> None:
    """Queue a batch of low-priority tasks followed by a high-priority task and show completion order."""
    handles: List[TaskHandle] = []
    for i in range(low_count):
        handles.append(enqueue_task("low", i))

    time.sleep(0.5)
    handles.append(enqueue_task("high", 0))

    for handle in handles:
        try:
            payload = handle.async_result.get(timeout=timeout)
        except Exception as exc:  # pragma: no cover - unexpected failures
            payload = {"status": "error", "error": str(exc)}
        handle.response_payload = payload
        handle.completed_at = time.perf_counter()

    handles.sort(key=lambda h: h.completed_at or float("inf"))

    click.echo("Completion order:")
    for idx, handle in enumerate(handles, start=1):
        click.echo(
            f"{idx}. queue={handle.queue} priority={handle.priority.upper()} "
            f"id={handle.prompt_id} duration={handle.duration:.2f}s "
            f"status={handle.response_payload.get('status')}"
        )

    first_high_index = next((i for i, h in enumerate(handles) if h.priority == "high"), None)
    first_low_after = next(
        (i for i, h in enumerate(handles) if h.priority == "low" and first_high_index is not None and i > first_high_index),
        None,
    )

    if first_high_index is not None and first_low_after is not None:
        click.echo("High priority task completed before at least one low priority task ✅")
    else:
        click.echo("High priority task did not finish ahead of low priority backlog. Adjust counts/timeout.", err=True)


if __name__ == "__main__":
    main()


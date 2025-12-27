from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import click
import requests

DEFAULT_PRIORITIES = ("high", "normal", "low")


@dataclass
class MetricsSnapshot:
    cache_hits: float
    cache_misses: float
    llm_calls: float
    llm_errors: float


def fetch_metrics(base_url: str) -> MetricsSnapshot:
    response = requests.get(f"{base_url}/metrics", timeout=5)
    response.raise_for_status()
    hits = misses = llm_calls = llm_errors = 0.0
    for line in response.text.splitlines():
        if line.startswith("semantic_cache_hits_total "):
            hits = float(line.split(" ", 1)[1])
        elif line.startswith("semantic_cache_misses_total "):
            misses = float(line.split(" ", 1)[1])
        elif line.startswith("mock_llm_calls_total "):
            llm_calls = float(line.split(" ", 1)[1])
        elif line.startswith("mock_llm_errors_total "):
            llm_errors = float(line.split(" ", 1)[1])
    return MetricsSnapshot(hits, misses, llm_calls, llm_errors)


def iter_prompts(seed: int = 42) -> Iterable[Tuple[str, str, str]]:
    random.seed(seed)
    counter = 0
    prompts = [
        "Explain how Celery ensures durable execution.",
        "Give five bullet points about Redis rate limiting.",
        "Describe semantic caching in three sentences.",
        "What is idempotency and why is it important?",
        "Outline failure handling in distributed systems.",
    ]
    users = ["load-user-1", "load-user-2", "load-user-3"]
    while True:
        user_id = random.choice(users)
        prompt_id = f"lt-{user_id}-{counter}"
        text = random.choice(prompts)
        counter += 1
        yield user_id, prompt_id, text


@click.command()
@click.option("--duration", default=60, show_default=True, help="Test duration in seconds.")
@click.option("--concurrency", default=5, show_default=True, help="Number of concurrent workers.")
@click.option("--base-url", default="http://localhost:8000", show_default=True, help="API base URL.")
@click.option(
    "--priorities",
    default=",".join(DEFAULT_PRIORITIES),
    show_default=True,
    help="Comma-separated list of priorities to cycle through.",
)
def main(duration: int, concurrency: int, base_url: str, priorities: str) -> None:
    priority_choices: List[str] = [p.strip() for p in priorities.split(",") if p.strip()]
    if not priority_choices:
        raise click.ClickException("At least one priority must be specified.")

    prompts = iter_prompts()
    start_metrics = fetch_metrics(base_url)

    click.echo(f"Starting load test for {duration}s with concurrency={concurrency}")
    click.echo(f"Priorities: {priority_choices}")

    responses = []
    start_time = time.perf_counter()
    end_time = start_time + duration
    request_count = 0

    while time.perf_counter() < end_time:
        user_id, prompt_id, text = next(prompts)
        payload = {
            "user_id": user_id,
            "prompt_id": prompt_id,
            "text": text,
            "priority": random.choice(priority_choices),
        }
        try:
            response = requests.post(
                f"{base_url}/process",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30,
            )
            request_count += 1
            if response.status_code == 200:
                responses.append(response.json())
            else:
                click.echo(
                    f"Request failed (status={response.status_code}): {response.text}",
                    err=True,
                )
        except requests.RequestException as exc:
            click.echo(f"Request exception: {exc}", err=True)

    duration_actual = time.perf_counter() - start_time
    total_processing_ms = sum(r.get("processing_time_ms") or 0 for r in responses)
    completed = sum(1 for r in responses if r.get("status") == "completed")
    cached_hits = sum(1 for r in responses if r.get("cached"))

    end_metrics = fetch_metrics(base_url)

    click.echo("\n=== Load Test Summary ===")
    click.echo(f"Requests sent: {request_count}")
    click.echo(f"Completed responses: {completed}")
    click.echo(f"Duration (s): {duration_actual:.2f}")
    click.echo(f"Throughput (req/s): {request_count / duration_actual:.2f}")
    click.echo(f"Average processing_time_ms: {total_processing_ms / max(completed, 1):.2f}")
    click.echo(f"Cache hits (response payload): {cached_hits}")

    click.echo("\nMetric deltas (start -> end):")
    click.echo(
        f"semantic_cache_hits_total: "
        f"{start_metrics.cache_hits} -> {end_metrics.cache_hits} "
        f"(+{end_metrics.cache_hits - start_metrics.cache_hits})"
    )
    click.echo(
        f"semantic_cache_misses_total: "
        f"{start_metrics.cache_misses} -> {end_metrics.cache_misses} "
        f"(+{end_metrics.cache_misses - start_metrics.cache_misses})"
    )
    click.echo(
        f"mock_llm_calls_total: "
        f"{start_metrics.llm_calls} -> {end_metrics.llm_calls} "
        f"(+{end_metrics.llm_calls - start_metrics.llm_calls})"
    )
    click.echo(
        f"mock_llm_errors_total: "
        f"{start_metrics.llm_errors} -> {end_metrics.llm_errors} "
        f"(+{end_metrics.llm_errors - start_metrics.llm_errors})"
    )


if __name__ == "__main__":
    main()


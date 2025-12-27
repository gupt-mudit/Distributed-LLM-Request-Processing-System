# Design Overview

## Architecture Summary

- **FastAPI** provides the REST layer. `POST /process` validates the request, runs a semantic cache lookup, and, on a miss, enqueues a Celery task while blocking until the result arrives (up to the configured timeout). `GET /process/{user}/{prompt}` surfaces status for dashboards or tooling, and `GET /metrics` exposes Prometheus metrics.
- **Celery** is the durable execution engine. I use Redis for both broker and result backend, `task_acks_late=True`, `task_reject_on_worker_lost=True`, and `worker_prefetch_multiplier=1` so a task that crashes midway is redelivered to another worker.
- **Priority queues**: I configured Celery with three queues (`prompt_high`, `prompt_normal`, `prompt_low`). The API routes each request based on the declared priority, so high-priority prompts jump the backlog. A single worker listens to all three queues in priority order; additional workers can be added later (see Trade-offs).
- **PostgreSQL + pgvector** store prompt metadata and cached responses. `ivfflat` indexes over pgvector (cosine distance) let me do fast similarity lookups, and the `(user_id, prompt_id)` unique constraint makes the API idempotent.
- **Redis** also backs the rate limiter—a Lua-script token bucket capped at 300 calls per minute plus a concurrency guard (defaults to 5 in-flight LLM calls).
- **Services layer** hides implementation details:
  - `EmbeddingService` loads `sentence-transformers` (or a deterministic fallback in tests).
  - `SemanticCacheService` handles lookups, writes, and hit-count increments.
  - `RedisRateLimiter` enforces the global provider quota.
  - `MockLLM` simulates latency, random provider failures, and integrates with the limiter for realistic behaviour.

### Why these choices?
- **FastAPI** keeps request validation simple, supports async handlers, and plays nicely with Pydantic & Prometheus middleware. Alternatives such as Flask or Django REST were heavier for the scope.
- **Celery + Redis** were picked over a Temporal/Prefect deployment because they’re easy to containerise, widely understood, and robust enough for retries and crash recovery. Temporal would add strong guarantees but needs a larger footprint.
- **PostgreSQL** handles the write-heavy workload, transactionally updates status/retry counts, and with `pgvector` I avoid running a separate vector database.
- **Python 3.11** for structural pattern matching, `taskgroups`, and performance improvements. Go was considered but Python’s ecosystem (sentence-transformers, Celery) offered more out of the box for LLM workloads.

## Prompt Processing Flow

```
        +---------+         +---------+         +-----------+
        | Client  | ----->  |  API    | ----->  |  Celery   |
        +---------+         +---------+         +-----------+
                                 |                    |
                                 v                    v
                           +-----------+        +------------+
                           | Postgres  | <----> |  Worker    |
                           +-----------+        +------------+
                                 ^                    |
                                 |                    v
                           +-----------+        +------------+
                           | Semantic  |        |  Redis     |
                           |  Cache    |        | (Limiter)  |
                           +-----------+        +------------+
                                                     |
                                                     v
                                               +------------+
                                               | Mock LLM   |
                                               +------------+

       1. Client calls `POST /process`.
       2. API locks/upserts `PromptRequest` in Postgres.
       3. API checks semantic cache; if hit, return response immediately.
       4. On miss, API enqueues `prompt.process` task on priority queue.
       5. Worker pulls task, rechecks cache, and acquires Redis token.
       6. Worker calls Mock LLM, stores response + embedding in Postgres.
       7. Worker updates request status; Celery returns result to API.
       8. API replies to client with completed payload.
```

1. Client issues `POST /process`.
2. The API upserts the `PromptRequest` row (locking via `SELECT … FOR UPDATE`), short-circuits to a cached response when similarity ≥ 0.9, otherwise persists the request as `queued`.
3. The request is routed to the Celery queue that matches the priority and `apply_async` returns an AsyncResult.
4. The API blocks on `result.get` (120s by default). If another worker already finished the prompt, we return the finished payload immediately; if the deadline passes or the task fails after max retries, the error is surfaced in the HTTP response.
5. The worker re-checks the cache (to avoid duplicate LLM calls), invokes the rate-limited `MockLLM`, and stores the result (embedding, response, timestamps) back into the cache and request tables.

## Data Model

- `PromptRequest`: user/prompt identifiers, text, priority, status, retry count, processing time, cache link, last error. Unique per `(user_id, prompt_id)` and updated atomically inside transactions.
- `PromptCacheEntry`: prompt text, embedding vector, response, `hit_count`, `created_at`, `last_hit_at`. `ivfflat` index on the embedding column for quick similarity lookup.

## Rate Limiting & Backpressure

- **Token bucket** the rate limiter keys off `rate:mock_llm:bucket` with minute granularity. When the bucket is empty the Celery task raises `RateLimitExceeded`, triggering built-in retries with exponential backoff (capped at 60s).
- **Concurrency guard** maintains `max_concurrent` (default 5) so I never stampede the provider even if multiple workers are idle.
- **API level**: because `/process` blocks until completion, incoming HTTP requests naturally backpressure the client if workers are saturated.

## Resilience Measures

- Celery with `acks_late` + Redis broker ensures tasks aren’t lost. If the worker dies mid-flight, the task is requeued once the visibility timeout expires.
- Worker code reselects the `PromptRequest` row using `SELECT … FOR UPDATE`; if a duplicate insert occurs (common after crash recovery) I reload the existing row instead of failing.
- `scripts/test_resilience.sh` proves it: it posts a prompt, kills the worker, restarts it, and waits until the job returns `completed`.
- Error handling bubbles provider failures back into `PromptRequest.error` and the HTTP payload (`"error": "...", "retry_count": n`).

## Priority Handling

- Requests are mapped to `prompt_high`, `prompt_normal`, or `prompt_low` queues. The worker command `-Q prompt_high,prompt_normal,prompt_low` makes Celery consume the high queue first.
- The demo script `scripts/test_priority_queue.py` enqueues a stack of low-priority tasks followed by a high-priority one and prints the completion order.
- **Trade-off**: a single worker still processes work sequentially; a long-running low-priority task can monopolise the worker. If the workload demands stricter SLAs, I can add dedicated high-priority workers (e.g., a separate container listening only to `prompt_high`) or separate horizontal pods per queue.

## Observability & Tooling

- **Metrics**: `/metrics` exposes Prometheus counters (`semantic_cache_hits_total`, `semantic_cache_misses_total`, `mock_llm_calls_total`, `mock_llm_errors_total`), process stats, etc. Compose bundles Prometheus and Grafana so dashboards can be built without extra setup.
- **Scripts**:
  - `scripts/load_test.py` generates traffic for a set duration and prints throughput/cache ratios plus metric deltas.
  - `scripts/verify_rate_limit.py` shows the limiter denying calls after the budget.
  - `scripts/test_resilience.sh` proves crash recovery.
  - `scripts/test_priority_queue.py` demonstrates priority routing using the Celery API.
- **Logging**: `structlog` emits structured JSON logs with latency, cache hit/miss, retry counts, task IDs.
- **Health**: `/health` ping backs DB, Redis, and Celery, returning `503` if any component is degraded.

## Testing Strategy

- **Unit tests** cover embeddings behaviour, rate limiter semantics, semantic cache logic, and mock LLM failure/rate-limit handling.
- **Integration tests** exercise `/process` for cache hits/misses, `/metrics`, `/process/{user}/{prompt}` status endpoint, and the priority-to-queue mapping.
- **Manual/regression scripts** (load, resilience, rate limit, priority) complement automated coverage and are run against the Docker stack.

## Alternatives & Trade-offs

- **Workflow orchestration**: I evaluated Temporal/Prefect. Temporal provides temporal workflows and guarantees but requires separate infrastructure and steeper learning curve. Celery/Redis gave me exactly-once semantics good enough for this exercise with minimal overhead.
- **Database choice**: MongoDB with Atlas Vector Search was considered. Postgres + pgvector won because I needed transactional updates, strong consistency for retry bookkeeping, and I could avoid running an additional vector store.
- **Rate limiting location**: Putting the limiter in the API would have reduced worker complexity but would have tied up the HTTP request thread during wait time. Keeping it in the worker ensures compliance even if multiple gateway instances exist.
- **Priority implementation**: The current multi-queue approach is simple and preserves FIFO order within each priority. For stricter isolation I could:
  - Run dedicated workers per priority queue.
  - Use Celery’s rate limits per queue.
  - Or move to a multi-queue broker that supports priority ordering within a single queue (RabbitMQ). We chose the multi-queue variant to keep Redis as the single broker.
- **Synchronous `/process`**: Blocking until completion simplifies clients but ties up the HTTP connection. If I expect very long-running jobs, I may expose a fully async mode (return 202 + encourage polling `/process/{user}/{prompt}`) or add WebSocket/server-sent events for status updates.

## Future Enhancements

- **Dedicated high-priority worker** to guarantee latency under extreme load.
- **Adaptive rate limit** (per-user quotas or dynamic budgets) if provider usage gets more complex.
- **UI layer** to trigger requests, view metrics, and run scripts directly from a browser.
- **Distributed tracing** with OpenTelemetry if we integrate real providers and need per-span insight.

This design balances durability, simplicity, and observability while keeping the deployment to a single Docker Compose stack that reviewers can run with `docker compose up --build`.

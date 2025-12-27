# Design Overview

## Architecture Summary

- **FastAPI** handles REST endpoints (`/process`, `/health`). The API validates
  requests, performs an immediate semantic cache lookup, and enqueues Celery
  tasks for cache misses. It waits synchronously (with timeout) for the worker
  result; otherwise responds with `status="processing"`.
- **Celery workers** execute the prompt pipeline. Configuration uses Redis as
  broker/result backend, `task_acks_late=True`, and `worker_prefetch_multiplier=1`
  to guarantee redelivery if a worker crashes mid-task.
- **PostgreSQL + pgvector** persist prompt requests and cache entries. Vector
  similarity search is performed using `ivfflat` index (cosine distance). A
  unique constraint on `(user_id, prompt_id)` ensures idempotency.
- **Redis** powers Celery and the global rate limiter (token bucket + optional
  concurrency guard). Rate limit is 300 LLM calls/minute across all workers.
- **Services layer** encapsulates shared logic:
  - `EmbeddingService`: sentence-transformers (or deterministic fallback) for
    embeddings.
  - `SemanticCacheService`: lookup/store cache entries, record hit counts.
  - `RedisRateLimiter`: Redis Lua scripts for quota tracking.
  - `MockLLM`: Simulates provider latency + failure rate while delegating to
    the rate limiter.

## Prompt Processing Flow

1. Client calls `POST /process`.
2. API locks/reads `PromptRequest`. If already completed with cache entry, returns cached response.
3. API computes embedding and queries semantic cache. If similarity ≥ 0.9, records hit and returns cached response.
4. On cache miss, API persists/updates `PromptRequest` (status `queued`) and enqueues Celery task (`prompt.process`).
5. Worker pipeline (executed under database session):
   - Re-fetches `PromptRequest` with `FOR UPDATE` to enforce idempotency.
   - Re-checks cache (race-safe).
   - Calls `MockLLM.complete()` on miss. Rate limit or provider failure triggers Celery retry with exponential backoff.
   - Stores new cache entry, updates request status, retry count, and timing.
6. API polls task result for up to 30 seconds. If completed, returns final response; otherwise indicates `processing`.

## Data Model Highlights

- `PromptRequest`: captures prompt text, status, priority, retry count,
  processing time, cache linkage, and timestamps. Unique per `(user_id, prompt_id)`.
- `PromptCacheEntry`: stores normalized embedding vector, response text,
  hit counters, and timestamps. `ivfflat` index optimised for cosine distance.

## Rate Limiting Strategy

- Redis token bucket with key `rate:mock_llm:bucket` counts calls per 60-second
  window (configurable). If quota exhausted, `RateLimitExceeded` prompts Celery
  retry.
- Optional concurrency guard (default max 5 simultaneous LLM calls) prevents
  bursts exceeding provider concurrency caps.

## Resilience & Error Handling

- Celery tasks use late acknowledgements; if a worker dies mid-processing,
  tasks are re-queued automatically.
- Rate-limit/provider failures trigger exponential backoff (`2^retries`, capped
  at 60s) with max 5 retries.
- Fatal exceptions mark `PromptRequest` as `failed` with `error_message`.
- `scripts/test_resilience.sh` automates a workflow: start prompt, kill worker,
  restart worker, poll until completion.

## Observability

- Structured logging (JSON-ready) captures user/prompt IDs, cache hit/miss,
  latency, retry counts.
- `/health` checks database, Redis, and Celery worker availability; returns 503
  if any component is degraded.
- Future enhancements (stretch goals) could expose `/metrics` for Prometheus or
  add structured eventing for cache hit rate tracking.

## Testing Strategy

- **Unit tests** cover embeddings fallback, rate limiter semantics, mock LLM
  behavior, and semantic cache helpers (with mocked sessions).
- **Integration tests** use FastAPI dependency overrides to simulate API flows:
  cached response path and enqueue/worker result path.
- Manual smoke tests use curl and the resilience script once Docker stack is running.

## Alternatives Considered

- **Workflow engine**: Temporal vs Celery. Celery chosen for simplicity and
  lower operational overhead for this assignment.
- **Database**: MongoDB with Atlas Vector Search vs PostgreSQL with `pgvector`.
  Postgres selected for mature vector support, transactional semantics, and
  easier migrations.
- **Rate limiting**: API-level synchronous limiter vs worker-level token bucket.
  Worker-level approach keeps rate enforcement close to LLM calls and avoids
  bottlenecking the API.



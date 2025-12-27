# Prompt Processing System

FastAPI service that queues prompt-processing jobs, checks a semantic cache
(`pgvector` in PostgreSQL), enforces a global LLM rate limit through Redis, and
executes work durably with Celery workers.

`POST /process` is synchronous: the request blocks until the prompt is processed
(or a cache hit is returned). This includes retrying work transparently if a
worker crashes mid-flight.

## Repository Layout

```
├── docker-compose.yml         # Application + infrastructure stack
├── Dockerfile.api / worker    # Runtime images for API and Celery worker
├── src/
│   ├── api/                   # FastAPI app, routers, dependency wiring
│   ├── services/              # Embeddings, semantic cache, rate limiter, mock LLM
│   ├── worker/                # Celery app and task implementation
│   └── models/                # SQLAlchemy models + Alembic migrations
├── tests/                     # Unit & integration tests
└── scripts/
    └── test_resilience.sh     # Worker crash-recovery smoke test
```

## Getting Started

1. **Prerequisites**
   - Docker & Docker Compose
   - Make sure ports `8000`, `5432`, `6379` are available

2. **Configuration**
   - Copy `config/env.example` to `.env` (or edit the example file) if you need to
     override defaults such as database credentials or embedding model name.

3. **Boot the stack (migrations run automatically)**
   ```bash
   docker compose up --build
   ```
   This starts PostgreSQL (with `pgvector`), Redis, the FastAPI app, Celery worker,
   and Celery beat scheduler. The API container’s entrypoint applies Alembic
   migrations before Uvicorn launches, so the schema is always up to date.

4. **Verify**
   - Health check: `curl http://localhost:8000/health`
   - Submit prompt:
     ```bash
     curl -X POST http://localhost:8000/process \
       -H "Content-Type: application/json" \
       -d '{"user_id": "demo", "prompt_id": "p1", "text": "Explain quantum computing", "priority": "normal"}'
     ```

## API Endpoints

- `POST /process` – submit a prompt. Returns the completed result (or cached response) before the HTTP call finishes.
- `GET /process/{user_id}/{prompt_id}` – read the latest status/result without re-processing, useful for polling or dashboards.
- `GET /health` – component health snapshot for load balancers/monitors.

## Testing

- Unit & integration tests:
  ```bash
  docker compose exec api poetry run pytest
  ```
  (Adjust command if you prefer running locally with a configured `DATABASE_URL`/Redis.)

- Resilience script (kills the worker mid-flight and verifies recovery):
  ```bash
  ./scripts/test_resilience.sh
  ```
  Override defaults with `API_URL`, `USER_ID`, `PRIORITY`, etc. if needed.

- Rate limit demo (shows limiter denying calls after configured quota):
  ```bash
  docker compose exec api python scripts/verify_rate_limit.py
  ```
  You can adjust the test limit by editing the script or instantiating `RedisRateLimiter`
  with different parameters.

- Load test (fires prompts for a fixed duration, summarises throughput & cache stats):
  ```bash
  docker compose exec api python scripts/load_test.py --duration 30 --concurrency 3
  ```
  Metrics deltas come from the Prometheus counters exposed at `/metrics`.

- Priority queue demo (fires a burst of low-priority requests followed by a high-priority one to show queue ordering):
  ```bash
  docker compose exec api python scripts/test_priority_queue.py --low-count 6
  ```
  Increase `--low-count` if you want to create more backlog and emphasise the difference.

- Metrics & dashboards:
  - Prometheus UI: http://localhost:9090 (scrapes `api:8000/metrics`)
  - Grafana UI: http://localhost:3000 (default creds `admin` / `admin`). Add Prometheus (`http://prometheus:9090`) as a data source to build dashboards.

## Key Features

- **Semantic caching**: embeddings via `sentence-transformers` (or deterministic mock)
  stored in PostgreSQL with `pgvector`, similarity threshold 0.9.
- **Rate limiting**: Redis-backed token bucket + concurrency guard to maintain 300
  LLM calls per minute across all workers.
- **Durable execution**: Celery configured with late acknowledgements, retry/backoff,
  and idempotent database updates to survive worker crashes.
- **Mock LLM**: simulates provider latency, random failures, and respects the rate
  limiter—no external API keys required.

See `DESIGN.md` for architectural decisions, trade-offs, and future improvements.


# Distributed-LLM-Request-Processing-System
# Distributed-LLM-Request-Processing-System

#!/bin/bash
set -euo pipefail

echo "[entrypoint] Applying database migrations..."

# Ensure src package can be discovered when running Alembic from the container
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="/app"
elif [[ ":${PYTHONPATH}:" != *":/app:"* ]]; then
  export PYTHONPATH="/app:${PYTHONPATH}"
fi

attempt=0
until poetry run alembic upgrade head; do
  attempt=$((attempt + 1))
  if [[ "${attempt}" -ge 10 ]]; then
    echo "[entrypoint] Migration failed after ${attempt} attempts; exiting."
    exit 1
  fi
  echo "[entrypoint] Migration failed, retrying in 5s (attempt ${attempt})..."
  sleep 5
done

echo "[entrypoint] Starting FastAPI server..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload


#!/bin/bash
set -euo pipefail

echo "[entrypoint] Starting FastAPI server..."
echo "[entrypoint] MongoDB indexes will be initialized on startup via main.py"

# Ensure src package can be discovered
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="/app"
elif [[ ":${PYTHONPATH}:" != *":/app:"* ]]; then
  export PYTHONPATH="/app:${PYTHONPATH}"
fi

exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload


#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_URL="${API_URL:-http://localhost:8000}"
USER_ID="${USER_ID:-resilience-user}"
PROMPT_ID="resilience-$(date +%s)"
PROMPT_TEXT="Resilience probe $(uuidgen)"
PRIORITY="${PRIORITY:-high}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-15}"
SLEEP_SECONDS="${SLEEP_SECONDS:-4}"

if docker compose version >/dev/null 2>&1; then
  DC="docker compose"
elif docker-compose version >/dev/null 2>&1; then
  DC="docker-compose"
else
  echo "docker compose is required but was not found." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but was not found." >&2
  exit 1
fi

TMP_RESPONSE="$(mktemp)"

echo "Submitting resilience prompt (user=${USER_ID}, prompt=${PROMPT_ID})..."
curl -sS -X POST "${API_URL}/process" \
  -H "Content-Type: application/json" \
  -d "$(cat <<JSON
{
  "user_id": "${USER_ID}",
  "prompt_id": "${PROMPT_ID}",
  "text": "${PROMPT_TEXT}",
  "priority": "${PRIORITY}"
}
JSON
)" > "${TMP_RESPONSE}" &
CURL_PID=$!

# Give the worker a moment to pick up the task before we kill it.
sleep 1

echo "Simulating worker crash..."
eval "${DC} kill worker" >/dev/null
sleep 3
eval "${DC} up -d worker" >/dev/null
echo "Worker restarted."

# Wait for the initial POST to complete now that the worker is back.
if ! wait "${CURL_PID}"; then
  echo "Initial request failed unexpectedly." >&2
  cat "${TMP_RESPONSE}"
  rm -f "${TMP_RESPONSE}"
  exit 1
fi

INITIAL_RESPONSE="$(cat "${TMP_RESPONSE}")"
rm -f "${TMP_RESPONSE}"

echo "Initial response: ${INITIAL_RESPONSE}"

echo "Polling for completion..."
attempt=1
while [[ "${attempt}" -le "${MAX_ATTEMPTS}" ]]; do
  RESPONSE="$(curl -sS "${API_URL}/process/${USER_ID}/${PROMPT_ID}")"

  STATUS="$(RESILIENCE_RESPONSE="${RESPONSE}" python - <<'PY'
import json
import os

response = os.environ.get("RESILIENCE_RESPONSE", "")
try:
    payload = json.loads(response)
except json.JSONDecodeError:
    payload = {}

print(payload.get("status", "unknown"))
PY
)"

  echo "Attempt ${attempt}: status=${STATUS}"

  if [[ "${STATUS}" == "completed" ]]; then
    echo "Resilience test succeeded. Prompt completed after worker restart."
    exit 0
  fi

  sleep "${SLEEP_SECONDS}"
  attempt=$((attempt + 1))
done

echo "Resilience test failed: prompt did not complete within allotted attempts." >&2
exit 1



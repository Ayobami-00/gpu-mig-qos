#!/usr/bin/env bash
set -euo pipefail

PORTS="${PORTS:-8000,8001,8002,8003,8004,8005,8006}"
TIMEOUT_SECS="${TIMEOUT_SECS:-300}"
START_TIME="$(date +%s)"

IFS=',' read -r -a PORT_ARRAY <<<"${PORTS}"

for port in "${PORT_ARRAY[@]}"; do
  echo "Waiting for vLLM on port ${port}..."
  until curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 || \
        curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do
    NOW="$(date +%s)"
    if (( NOW - START_TIME > TIMEOUT_SECS )); then
      echo "Timed out waiting for port ${port} after ${TIMEOUT_SECS}s." >&2
      exit 1
    fi
    sleep 2
  done
done

echo "All requested vLLM endpoints are ready."

#!/usr/bin/env bash
set -euo pipefail
url="http://127.0.0.1:8000/v1"
for _ in $(seq 1 90); do
    if curl -fsS --max-time 3 "$url/models" >/dev/null 2>&1; then
        break
    fi
    sleep 10
done
curl -fsS --max-time 120 "$url/models" >/dev/null
curl -fsS --max-time 180 \
    -H 'Content-Type: application/json' \
    -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Reply with exactly: READY"}],"temperature":0,"max_tokens":8}' \
    "$url/chat/completions" >/dev/null

#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "=== Health Check ==="
curl -s "$BASE_URL/health" | python3 -m json.tool

echo ""
echo "=== Waiting for engine to be ready ==="
for i in $(seq 1 60); do
    if curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/ready" | grep -q "200"; then
        echo "Engine is ready!"
        break
    fi
    echo "  Attempt $i/60 - not ready yet, waiting 5s..."
    sleep 5
done

echo ""
echo "=== Chat Completion ==="
curl -s -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "temperature": 0,
        "max_tokens": 256
    }' | python3 -m json.tool

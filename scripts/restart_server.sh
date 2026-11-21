#!/usr/bin/env bash
# Restart the server and wait until THIS run has warmed its prompt caches.
set -euo pipefail
cd "$(dirname "$0")/.."
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
docker compose restart server >/dev/null 2>&1
for _ in $(seq 1 180); do
  if docker compose logs --since "$since" server 2>&1 | grep -qE "caches warm|warmup failed"; then
    docker compose logs --no-log-prefix --since "$since" server | grep -E "caches warm|warmup failed"
    exit 0
  fi
  sleep 2
done
echo "server did not warm up in time" >&2; exit 1

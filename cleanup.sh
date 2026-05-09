#!/bin/bash
# Emergency cleanup: kill ALL processes on ports 7342 and 8765

set -e

echo "=== EMERGENCY CLEANUP ==="

for port in 7342 8765; do
  echo "Checking port $port..."
  pids=$(lsof -ti :$port 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "Killing processes on port $port: $pids"
    kill -9 $pids 2>/dev/null || true
  fi
done

sleep 1

echo "=== Verification ==="
for port in 7342 8765; do
  if lsof -i :$port 2>/dev/null | grep -q .; then
    echo "❌ Port $port still in use!"
    exit 1
  else
    echo "✅ Port $port free"
  fi
done

echo "=== System ready ==="

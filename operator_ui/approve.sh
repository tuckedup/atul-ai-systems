#!/bin/sh
# approve.sh — run the approval round-trip via the API (no browser needed)
# Usage: cd operator_ui && sh approve.sh
set -e
cd "$(dirname "$0")"
echo "==> Seeding fixtures"
python operator_api/seed.py
echo "==> Starting API server on :8777"
uvicorn operator_api.app:APP --host 127.0.0.1 --port 8777 &
API_PID=$!
sleep 2
echo "==> Running approval round-trip"
python operator_api/approve_roundtrip.py
echo "==> Stopping API server"
kill $API_PID 2>/dev/null || true
echo "==> Done"

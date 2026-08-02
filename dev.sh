#!/usr/bin/env bash
# The Honest Backtester — one command to run everything.
# Backend (FastAPI) on :8000, frontend (Next.js) on :3000.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating Python venv (.venv)…"
  python3.11 -m venv .venv
  .venv/bin/pip install --quiet -r backend/requirements.txt
fi
if [ ! -d frontend/node_modules ]; then
  echo "Installing frontend dependencies…"
  (cd frontend && npm install)
fi

mkdir -p data

# Pick the backend port: HONEST_BACKEND_PORT if set, else 8000, else the next
# free port in 8010/8020/8030 (dev machines often have something on 8000).
pick_port() {
  for p in "$@"; do
    if ! lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "$p"; return
    fi
  done
  echo "no free port among: $*" >&2; exit 1
}
BACK_PORT="${HONEST_BACKEND_PORT:-$(pick_port 8000 8010 8020 8030)}"

echo "Starting backend on http://localhost:${BACK_PORT} (docs at /docs)…"
.venv/bin/python -m uvicorn main:app --app-dir backend --port "$BACK_PORT" --reload &
BACK_PID=$!

echo "Starting frontend on http://localhost:3000…"
(cd frontend && NEXT_PUBLIC_API_BASE="http://localhost:${BACK_PORT}" npm run dev) &
FRONT_PID=$!

trap 'echo; echo "Shutting down…"; kill $BACK_PID $FRONT_PID 2>/dev/null; wait' INT TERM
wait

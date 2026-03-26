#!/bin/bash
set -euo pipefail

# Start JupyterLab in the background (same image/deps as dev).
# Open http://localhost:8888 — token disabled for local dev only.

docker compose up -d jupyter

sleep 2

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:8888" || true
elif command -v open >/dev/null 2>&1; then
  open "http://localhost:8888" || true
else
  echo "Open http://localhost:8888 in your browser"
fi

echo "To stop: docker compose stop jupyter"

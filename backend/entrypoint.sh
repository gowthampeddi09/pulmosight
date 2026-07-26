#!/bin/bash
set -e

echo "Running database migrations..."
alembic upgrade head || echo "Alembic migration warning, continuing..."

PORT="${PORT:-8000}"
echo "Starting PulmoSight backend on port $PORT..."
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"

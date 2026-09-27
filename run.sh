#!/usr/bin/env bash
# One-shot dev launcher for Sentinel.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}

if [ ! -d .venv ]; then
  echo "▶ creating virtualenv (.venv)"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "▶ installing dependencies"
pip install -q -r requirements-dev.txt

if [ ! -f sentinel/artifacts/model.joblib ]; then
  echo "▶ training model (first run only, ~10s)"
  python -m sentinel.train
  echo "▶ evaluating on a fresh held-out world"
  python -m sentinel.eval || true
  echo "▶ latency + fairness audit"
  python -m sentinel.audit || true
fi

echo "▶ starting API + dashboard on http://127.0.0.1:8000"
exec uvicorn sentinel.main:app --host 127.0.0.1 --port 8000

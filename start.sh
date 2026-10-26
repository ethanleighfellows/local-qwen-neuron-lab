#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="$PWD/data/hf"
exec .venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8000

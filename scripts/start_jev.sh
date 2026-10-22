#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
JEV_PYTHON="${JEV_PYTHON:-$PROJECT_DIR/.venv-jev/bin/python}"
if ! command -v "$JEV_PYTHON" >/dev/null 2>&1; then
  echo "JEV environment missing: $JEV_PYTHON. Create .venv-jev or set JEV_PYTHON." >&2
  exit 1
fi
exec "$JEV_PYTHON" -m voice_agent.jev "$@"

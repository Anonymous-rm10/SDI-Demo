#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  DEFAULT_GATEWAY_PYTHON="$PROJECT_DIR/.venv/bin/python"
else
  DEFAULT_GATEWAY_PYTHON=python3
fi
GATEWAY_PYTHON="${GATEWAY_PYTHON:-$DEFAULT_GATEWAY_PYTHON}"
ASR_PYTHON="${ASR_PYTHON:-$PROJECT_DIR/.venv-asr/bin/python}"
TTS_PYTHON="${TTS_PYTHON:-$PROJECT_DIR/.venv-tts/bin/python}"
if [[ "${1:-}" == "--mock" ]]; then
  exec "$GATEWAY_PYTHON" -m voice_agent gateway --mock
fi
if [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/start.sh [--mock]" >&2
  exit 2
fi
for interpreter in "$ASR_PYTHON" "$TTS_PYTHON"; do
  if ! command -v "$interpreter" >/dev/null 2>&1; then
    echo "Python environment missing: $interpreter. See README or set ASR_PYTHON/TTS_PYTHON." >&2
    exit 1
  fi
done
pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
"$ASR_PYTHON" -m voice_agent asr &
pids+=("$!")
"$TTS_PYTHON" -m voice_agent tts &
pids+=("$!")
"$GATEWAY_PYTHON" -m voice_agent gateway &
pids+=("$!")
echo "Gateway: http://127.0.0.1:8090 (or GATEWAY_PORT in .env). JEV is started separately."
# Stop all children if any service exits. Never leave orphaned GPU workers.
wait -n "${pids[@]}"

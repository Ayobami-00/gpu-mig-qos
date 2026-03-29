#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v uv >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install --user uv
  UV_BIN="$("$PYTHON_BIN" -m site --user-base)/bin/uv"
else
  UV_BIN="uv"
fi

"$UV_BIN" venv --python "$PYTHON_BIN" --seed .venv
source .venv/bin/activate

# vLLM recommends using uv and selecting the torch backend automatically.
"$UV_BIN" pip install vllm --torch-backend=auto
"$UV_BIN" pip install -r requirements.txt

echo
echo "Bootstrap complete."
echo "Activate the environment with:"
echo "  source .venv/bin/activate"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:?usage: run_experiment.sh <shared|mig>}"
SCENARIO="${2:-configs/scenarios/${MODE}.yaml}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${REPO_ROOT}/experiments/${MODE}/${TIMESTAMP}"
LATEST_DIR="${REPO_ROOT}/experiments/${MODE}/latest"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"

export MODEL_ID

mkdir -p "${OUT_DIR}"

echo "Running mode: ${MODE}"
echo "Scenario: ${SCENARIO}"
echo "Model: ${MODEL_ID}"

./scripts/capture_state.sh "${OUT_DIR}/state-before"
python apps/loadgen/run.py --scenario "${SCENARIO}" --output-dir "${OUT_DIR}"
python charts/plot_results.py \
  --requests-csv "${OUT_DIR}/requests.csv" \
  --summary-csv "${OUT_DIR}/summary.csv" \
  --output-dir "${OUT_DIR}/charts" \
  --label "${MODE}"
./scripts/capture_state.sh "${OUT_DIR}/state-after"

rm -rf "${LATEST_DIR}"
mkdir -p "$(dirname "${LATEST_DIR}")"
cp -R "${OUT_DIR}" "${LATEST_DIR}"

echo
echo "Experiment complete."
echo "Results: ${OUT_DIR}"

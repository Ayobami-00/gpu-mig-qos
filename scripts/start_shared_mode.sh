#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
VLLM_API_KEY="${VLLM_API_KEY:-token-abc123}"
BASE_GPU="${BASE_GPU:-0}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$REPO_ROOT/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/shared}"

mkdir -p "$DOWNLOAD_DIR" "$LOG_DIR"

pkill -f "vllm serve" >/dev/null 2>&1 || true

echo "Using model: ${MODEL_ID}"

if nvidia-smi -L 2>/dev/null | grep -q 'MIG '; then
  echo "MIG devices detected. Disable MIG on GPU ${BASE_GPU} before starting shared mode." >&2
  exit 1
fi

echo "Starting tenant A on GPU ${BASE_GPU}, port 8000..."
CUDA_VISIBLE_DEVICES="${BASE_GPU}" \
vllm serve "${MODEL_ID}" \
  --config configs/vllm/shared/tenant-a.yaml \
  --generation-config vllm \
  --download-dir "${DOWNLOAD_DIR}" \
  --api-key "${VLLM_API_KEY}" \
  >"${LOG_DIR}/tenant-a.log" 2>&1 &

TENANT_A_PID=$!

echo "Waiting for tenant A to stabilize (45 seconds)..."
sleep 45

echo "Starting tenant B on GPU ${BASE_GPU}, port 8001..."
CUDA_VISIBLE_DEVICES="${BASE_GPU}" \
vllm serve "${MODEL_ID}" \
  --config configs/vllm/shared/tenant-b.yaml \
  --generation-config vllm \
  --download-dir "${DOWNLOAD_DIR}" \
  --api-key "${VLLM_API_KEY}" \
  >"${LOG_DIR}/tenant-b.log" 2>&1 &
echo "Waiting 50s for tenant B to profile memory before starting C..."
sleep 50

echo "Starting tenant C on GPU ${BASE_GPU}, port 8002..."
CUDA_VISIBLE_DEVICES="${BASE_GPU}" \
vllm serve "${MODEL_ID}" \
  --config configs/vllm/shared/tenant-c.yaml \
  --generation-config vllm \
  --download-dir "${DOWNLOAD_DIR}" \
  --api-key "${VLLM_API_KEY}" \
  >"${LOG_DIR}/tenant-c.log" 2>&1 &
echo "Waiting 50s for tenant C to profile memory before starting D..."
sleep 50

echo "Starting tenant D on GPU ${BASE_GPU}, port 8003..."
CUDA_VISIBLE_DEVICES="${BASE_GPU}" \
vllm serve "${MODEL_ID}" \
  --config configs/vllm/shared/tenant-d.yaml \
  --generation-config vllm \
  --download-dir "${DOWNLOAD_DIR}" \
  --api-key "${VLLM_API_KEY}" \
  >"${LOG_DIR}/tenant-d.log" 2>&1 &
echo "Waiting 50s for tenant D to profile memory before starting E..."
sleep 50

echo "Starting tenant E on GPU ${BASE_GPU}, port 8004..."
CUDA_VISIBLE_DEVICES="${BASE_GPU}" \
vllm serve "${MODEL_ID}" \
  --config configs/vllm/shared/tenant-e.yaml \
  --generation-config vllm \
  --download-dir "${DOWNLOAD_DIR}" \
  --api-key "${VLLM_API_KEY}" \
  >"${LOG_DIR}/tenant-e.log" 2>&1 &
echo "Waiting 50s for tenant E to profile memory before starting F..."
sleep 50

echo "Starting tenant F on GPU ${BASE_GPU}, port 8005..."
CUDA_VISIBLE_DEVICES="${BASE_GPU}" \
vllm serve "${MODEL_ID}" \
  --config configs/vllm/shared/tenant-f.yaml \
  --generation-config vllm \
  --download-dir "${DOWNLOAD_DIR}" \
  --api-key "${VLLM_API_KEY}" \
  >"${LOG_DIR}/tenant-f.log" 2>&1 &
echo "Waiting 50s for tenant F to profile memory before starting G..."
sleep 50

echo "Starting tenant G on GPU ${BASE_GPU}, port 8006..."
CUDA_VISIBLE_DEVICES="${BASE_GPU}" \
vllm serve "${MODEL_ID}" \
  --config configs/vllm/shared/tenant-g.yaml \
  --generation-config vllm \
  --download-dir "${DOWNLOAD_DIR}" \
  --api-key "${VLLM_API_KEY}" \
  >"${LOG_DIR}/tenant-g.log" 2>&1 &

echo
echo "Shared mode servers are launching (7 tenants on GPU ${BASE_GPU})."
echo "Logs:"
echo "  ${LOG_DIR}/tenant-a.log"
echo "  ${LOG_DIR}/tenant-b.log"
echo "  ${LOG_DIR}/tenant-c.log  tenant-d.log  tenant-e.log  tenant-f.log  tenant-g.log"

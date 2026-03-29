#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
VLLM_API_KEY="${VLLM_API_KEY:-token-abc123}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$REPO_ROOT/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/mig}"

mkdir -p "$DOWNLOAD_DIR" "$LOG_DIR"

echo "Using model: ${MODEL_ID}"

detect_mig_uuid() {
  local index="$1"
  nvidia-smi -L | grep 'MIG 1g.5gb' | sed -E 's/.*UUID: (MIG-[^)]+)\).*/\1/' | sed -n "${index}p"
}

TENANT_A_MIG_UUID="${TENANT_A_MIG_UUID:-$(detect_mig_uuid 1)}"
TENANT_B_MIG_UUID="${TENANT_B_MIG_UUID:-$(detect_mig_uuid 2)}"
TENANT_C_MIG_UUID="${TENANT_C_MIG_UUID:-$(detect_mig_uuid 3)}"
TENANT_D_MIG_UUID="${TENANT_D_MIG_UUID:-$(detect_mig_uuid 4)}"
TENANT_E_MIG_UUID="${TENANT_E_MIG_UUID:-$(detect_mig_uuid 5)}"
TENANT_F_MIG_UUID="${TENANT_F_MIG_UUID:-$(detect_mig_uuid 6)}"
TENANT_G_MIG_UUID="${TENANT_G_MIG_UUID:-$(detect_mig_uuid 7)}"

if [[ -z "${TENANT_A_MIG_UUID}" || -z "${TENANT_B_MIG_UUID}" || -z "${TENANT_C_MIG_UUID}" || \
      -z "${TENANT_D_MIG_UUID}" || -z "${TENANT_E_MIG_UUID}" || -z "${TENANT_F_MIG_UUID}" || \
      -z "${TENANT_G_MIG_UUID}" ]]; then
  echo "Could not detect 7 MIG UUIDs. Ensure 7x 1g.5gb instances are created." >&2
  nvidia-smi -L >&2
  exit 1
fi

pkill -f "vllm serve" >/dev/null 2>&1 || true

echo "Starting tenant A on ${TENANT_A_MIG_UUID}, port 8000..."
(
  export CUDA_VISIBLE_DEVICES="${TENANT_A_MIG_UUID}"
  vllm serve "${MODEL_ID}" \
    --config configs/vllm/mig/tenant-a.yaml \
    --generation-config vllm \
    --download-dir "${DOWNLOAD_DIR}" \
    --api-key "${VLLM_API_KEY}" \
    >"${LOG_DIR}/tenant-a.log" 2>&1
) &

echo "Waiting for tenant A to stabilize (20 seconds)..."
sleep 20

declare -A TENANT_PORTS=([b]=8001 [c]=8002 [d]=8003 [e]=8004 [f]=8005 [g]=8006)

for TENANT in b c d e f g; do
  UUID_VAR="TENANT_$(echo "${TENANT}" | tr '[:lower:]' '[:upper:]')_MIG_UUID"
  PORT="${TENANT_PORTS[${TENANT}]}"
  echo "Starting tenant ${TENANT} on ${!UUID_VAR}, port ${PORT}..."
  (
    export CUDA_VISIBLE_DEVICES="${!UUID_VAR}"
    vllm serve "${MODEL_ID}" \
      --config "configs/vllm/mig/tenant-${TENANT}.yaml" \
      --generation-config vllm \
      --download-dir "${DOWNLOAD_DIR}" \
      --api-key "${VLLM_API_KEY}" \
      >"${LOG_DIR}/tenant-${TENANT}.log" 2>&1
  ) &
done

echo
echo "MIG mode servers are launching (7 tenants, one per 1g.5gb slice)."
echo "Model: ${MODEL_ID}"
echo "Tenant A MIG UUID: ${TENANT_A_MIG_UUID}"
echo "Tenant B MIG UUID: ${TENANT_B_MIG_UUID}"
echo "Tenant C MIG UUID: ${TENANT_C_MIG_UUID}"
echo "Tenant D MIG UUID: ${TENANT_D_MIG_UUID}"
echo "Tenant E MIG UUID: ${TENANT_E_MIG_UUID}"
echo "Tenant F MIG UUID: ${TENANT_F_MIG_UUID}"
echo "Tenant G MIG UUID: ${TENANT_G_MIG_UUID}"

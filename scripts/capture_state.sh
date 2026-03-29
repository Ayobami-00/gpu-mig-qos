#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:?usage: capture_state.sh <output-dir>}"
mkdir -p "${OUT_DIR}"
ERROR_FILE="${OUT_DIR}/capture-errors.txt"

can_sudo_without_password() {
  command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1
}

capture_optional() {
  local output_path="$1"
  shift

  if "$@" >"${output_path}" 2>&1; then
    return 0
  fi

  if can_sudo_without_password && sudo -n "$@" >"${output_path}" 2>&1; then
    return 0
  fi

  printf 'Capture failed: %s\n' "$*" | tee -a "${ERROR_FILE}" >&2
  return 0
}

nvidia-smi -L >"${OUT_DIR}/nvidia-smi-L.txt"
nvidia-smi >"${OUT_DIR}/nvidia-smi.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,utilization.memory --format=csv,noheader,nounits \
  >"${OUT_DIR}/gpu-query.csv" || true

capture_optional "${OUT_DIR}/mig-profiles.txt" nvidia-smi mig -lgip
capture_optional "${OUT_DIR}/mig-gpu-instances.txt" nvidia-smi mig -lgi
capture_optional "${OUT_DIR}/mig-compute-instances.txt" nvidia-smi mig -lci

if command -v dcgmi >/dev/null 2>&1; then
  capture_optional "${OUT_DIR}/dcgm-discovery.txt" dcgmi discovery -l
fi

if [[ ! -s "${ERROR_FILE}" ]]; then
  rm -f "${ERROR_FILE}"
fi

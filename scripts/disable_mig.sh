#!/usr/bin/env bash
set -euo pipefail

TARGET_GPU="${TARGET_GPU:-0}"

if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl stop dcgm >/dev/null 2>&1 || true
fi

sudo nvidia-smi mig -i "${TARGET_GPU}" -dci >/dev/null 2>&1 || true
sudo nvidia-smi mig -i "${TARGET_GPU}" -dgi >/dev/null 2>&1 || true
sudo nvidia-smi -i "${TARGET_GPU}" -mig 0

if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl start dcgm >/dev/null 2>&1 || true
fi

echo
echo "MIG disabled on GPU ${TARGET_GPU}."
nvidia-smi -L

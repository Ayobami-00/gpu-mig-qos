#!/usr/bin/env bash
set -euo pipefail

TARGET_GPU="${TARGET_GPU:-0}"
MIG_PROFILE="${MIG_PROFILE:-1g.5gb}"
MIG_COUNT="${MIG_COUNT:-7}"

if (( MIG_COUNT < 2 )); then
  echo "MIG_COUNT must be at least 2 for the tenant isolation experiment." >&2
  exit 1
fi

if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl stop dcgm >/dev/null 2>&1 || true
fi

echo "Enabling MIG on GPU ${TARGET_GPU}..."
sudo nvidia-smi -i "${TARGET_GPU}" -mig 1

echo "Clearing any existing compute instances on GPU ${TARGET_GPU}..."
sudo nvidia-smi mig -i "${TARGET_GPU}" -dci >/dev/null 2>&1 || true
sudo nvidia-smi mig -i "${TARGET_GPU}" -dgi >/dev/null 2>&1 || true

profiles=""
for _ in $(seq 1 "${MIG_COUNT}"); do
  if [[ -z "${profiles}" ]]; then
    profiles="${MIG_PROFILE}"
  else
    profiles="${profiles},${MIG_PROFILE}"
  fi
done

echo "Creating MIG geometry ${profiles} on GPU ${TARGET_GPU}..."
sudo nvidia-smi mig -i "${TARGET_GPU}" -cgi "${profiles}" -C

if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl start dcgm >/dev/null 2>&1 || true
fi

echo
echo "Current device inventory:"
nvidia-smi -L

echo
echo "Available GPU instance profiles:"
nvidia-smi mig -i "${TARGET_GPU}" -lgip || true

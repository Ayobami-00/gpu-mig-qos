#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script to install vLLM with MIG support fix (PR #35526)
# This script clones vLLM from the PR branch and builds it from source
# to enable proper MIG UUID handling in CUDA_VISIBLE_DEVICES

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_BUILD_DIR="${VLLM_BUILD_DIR:-/tmp/vllm-mig-fix}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=========================================="
echo "vLLM MIG Fix Bootstrap"
echo "=========================================="
echo "Repository root: $REPO_ROOT"
echo "Build directory: $VLLM_BUILD_DIR"
echo "Python binary: $PYTHON_BIN"
echo ""

# Ensure we have a virtual environment
if [ ! -d "$REPO_ROOT/.venv" ]; then
    echo "Error: Virtual environment not found at $REPO_ROOT/.venv"
    echo "Please run bootstrap_lambda_host.sh first"
    exit 1
fi

source "$REPO_ROOT/.venv/bin/activate"

# Check if uv is available
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    "$PYTHON_BIN" -m pip install uv
    UV_BIN="uv"
else
    UV_BIN="uv"
fi

echo "Step 1: Cloning vLLM repository with MIG fix branch..."
if [ -d "$VLLM_BUILD_DIR" ]; then
    echo "  Removing existing build directory..."
    rm -rf "$VLLM_BUILD_DIR"
fi

git clone https://github.com/vllm-project/vllm.git "$VLLM_BUILD_DIR"
cd "$VLLM_BUILD_DIR"

echo "Step 2: Checking out PR #35526 branch (fix-mig-comprehensive)..."
git fetch origin pull/35526/head:fix-mig-comprehensive
git checkout fix-mig-comprehensive

echo "Step 3: Installing PyTorch (if needed)..."
"$UV_BIN" pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 2>&1 | grep -v "already satisfied" || true

echo "Step 4: Installing build dependencies..."
grep -v '^torch==' requirements/build.txt | "$UV_BIN" pip install -r - 2>&1 | grep -v "already satisfied" || true

echo "Step 5: Building and installing vLLM from source..."
"$UV_BIN" pip install -e . --no-build-isolation

echo ""
echo "=========================================="
echo "✓ vLLM with MIG fix installed successfully!"
echo "=========================================="
echo ""
echo "Installed from: $VLLM_BUILD_DIR"
echo "Branch: fix-mig-comprehensive (PR #35526)"
echo ""
echo "Key fixes included:"
echo "  - MIG UUID handling in device_id_to_physical_device_id()"
echo "  - NVML helper functions for MIG device lookups"
echo "  - Proper parent device handle retrieval for MIG instances"
echo ""
echo "You can now run MIG mode with:"
echo "  ./scripts/start_mig_mode.sh"
echo "  ./scripts/wait_for_vllm.sh"
echo ""

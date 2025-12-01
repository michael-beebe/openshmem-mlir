#!/bin/bash

set -euo pipefail

# Setup conda environment for OpenSHMEM MLIR shmem4py frontend
#
# This script creates a conda environment with:
#   - Python 3.10+
#   - NumPy for array operations
#   - Development tools
#   - Optional: shmem4py when available on PyPI
#
# Usage:
#   source scripts/setup_shmem4py_env.sh [env-name] [python-version]
#
# Default:
#   scripts/setup_shmem4py_env.sh
#   Creates/activates environment: openshmem-mlir
#   Python version: 3.11

ENV_NAME="${1:-openshmem-mlir}"
PYTHON_VERSION="${2:-3.11}"

echo "Setting up conda environment: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"

# Check if conda is available
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda." >&2
    return 1
fi

# Create environment if it doesn't exist
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[*] Environment '${ENV_NAME}' already exists."
    read -p "    Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove --name "$ENV_NAME" --yes
    else
        echo "[*] Skipping creation. Activating existing environment..."
        conda activate "$ENV_NAME"
        return 0
    fi
fi

# Create the environment
echo "[*] Creating conda environment..."
conda create \
    --name "$ENV_NAME" \
    --yes \
    "python=${PYTHON_VERSION}" \
    numpy \
    pip

# Activate the environment
echo "[*] Activating environment..."
conda activate "$ENV_NAME"

# Install pip dependencies from requirements.txt
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PIP_REQUIREMENTS="${ROOT_DIR}/python/requirements.txt"

if [[ -f "$PIP_REQUIREMENTS" ]]; then
    echo "[*] Installing Python dependencies..."
    pip install -r "$PIP_REQUIREMENTS" || true
    # Allow failures for optional deps like shmem4py which may not be on PyPI yet
else
    echo "WARNING: Could not find requirements.txt at $PIP_REQUIREMENTS" >&2
fi

cat <<EOF

==> Environment setup complete

To activate the environment:
    conda activate $ENV_NAME

To deactivate:
    conda deactivate

Environment details:
    Name: $ENV_NAME
    Python: $PYTHON_VERSION
    Location: $(conda run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)")

Next steps:
    1. Build OpenSHMEM MLIR with Python bindings enabled:
       ./scripts/build_openshmem_mlir.sh

    2. Add the build output to Python path:
       export PYTHONPATH=\$PYTHONPATH:\$(pwd)/build-incubator/python_packages

    3. Test the shmem4py frontend (once development begins):
       cd python
       pytest tests/

EOF

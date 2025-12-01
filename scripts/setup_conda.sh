#!/bin/bash

if [ ! -x "${CONDA_EXE:-}" ]; then
    echo "==> Conda is not installed or not on PATH."
    echo "    Install Miniconda/Anaconda and re-run this script."
    exit 1
fi

if conda env list | grep -q "^openshmem-mlir "; then
    echo "==> Conda environment 'openshmem-mlir' already exists."
else
    echo "==> Conda environment 'openshmem-mlir' does not exist."
    echo "==> Creating conda environment 'openshmem-mlir'..."
    conda create -n openshmem-mlir -y
fi

echo "==> Activating conda environment 'openshmem-mlir'..."
eval "$(conda shell.bash hook)"
conda activate openshmem-mlir

# Install Python dependencies if a requirements file is present.
REQ_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/python/requirements.txt"
if [ -f "${REQ_FILE}" ]; then
    echo "==> Installing Python requirements from ${REQ_FILE}..."
    pip install -r "${REQ_FILE}"
else
    echo "==> No python/requirements.txt found; skipping pip install."
fi

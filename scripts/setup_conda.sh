#!/bin/bash

if [ ! -x "${CONDA_EXE:-}" ]; then
    echo "==> Conda is not installed."
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


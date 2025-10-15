#!/bin/bash
set -euo pipefail
# Build OpenSHMEM MLIR project against a ClangIR (incubator) build.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd -P)"
CLANGIR_SRC_DIR="${CLANGIR_SRC_DIR:-${ROOT_DIR}/clangir}"
CLANGIR_BUILD_DIR="${CLANGIR_BUILD_DIR:-${CLANGIR_SRC_DIR}/build-main}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-clangir}" # separate build tree
GENERATOR="${GENERATOR:-Ninja}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
TOTAL_CORES="$(nproc)"
DEFAULT_CORES=$(( TOTAL_CORES / 2 ))
if (( DEFAULT_CORES < 1 )); then DEFAULT_CORES=1; fi
CORES="${CORES:-${DEFAULT_CORES}}"
# Sanity bounds
if ! [[ "${CORES}" =~ ^[0-9]+$ ]]; then CORES=${DEFAULT_CORES}; fi
if (( CORES < 1 )); then CORES=1; fi
if (( CORES > TOTAL_CORES )); then CORES=${TOTAL_CORES}; fi

MLIR_DIR="${MLIR_DIR:-${CLANGIR_BUILD_DIR}/lib/cmake/mlir}"
LLVM_DIR="${LLVM_DIR:-${CLANGIR_BUILD_DIR}/lib/cmake/llvm}"

echo "==> Project root : ${ROOT_DIR}"
echo "==> ClangIR src  : ${CLANGIR_SRC_DIR}"
echo "==> ClangIR build: ${CLANGIR_BUILD_DIR}"
echo "==> Build dir    : ${BUILD_DIR}"
echo "==> MLIR_DIR     : ${MLIR_DIR}"
echo "==> LLVM_DIR     : ${LLVM_DIR}"
echo "==> Parallel jobs: ${CORES}"

if [[ ! -d "${CLANGIR_BUILD_DIR}" ]]; then
  echo "ERROR: ClangIR build not found at ${CLANGIR_BUILD_DIR}" >&2
  echo "Run ./scripts/build_clangir.sh first." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"

echo "==> Configuring (clangir toolchain)..."
cmake -G "${GENERATOR}" \
  -S "${ROOT_DIR}" \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DCLANGIR_SRC_DIR="${CLANGIR_SRC_DIR}" \
  -DCLANGIR_BUILD_DIR="${CLANGIR_BUILD_DIR}"

echo "==> Building..."
cmake --build "${BUILD_DIR}" -- -j"${CORES}"

echo "==> Done. Build tree: ${BUILD_DIR} (clangir toolchain)"

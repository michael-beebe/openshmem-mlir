#!/bin/bash

set -euo pipefail

# Build this out-of-tree OpenSHMEM MLIR project against an existing llvm-project build/install.
# Minimal behavior: detect MLIR/LLVM CMake package dirs under ./llvm-project and configure+build.
# No environment modifications; no installs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd -P)"
LLVM_SRC_DIR="${LLVM_SRC_DIR:-${ROOT_DIR}/llvm-project}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
GENERATOR="${GENERATOR:-Ninja}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

echo "==> Project root : ${ROOT_DIR}"
echo "==> LLVM src dir : ${LLVM_SRC_DIR}"
echo "==> Build dir    : ${BUILD_DIR}"
echo "==> Generator    : ${GENERATOR}"
echo "==> Build type   : ${CMAKE_BUILD_TYPE}"

# If MLIR_DIR/LLVM_DIR are not provided, try to auto-detect from llvm-project build/install trees.
detect_pkg_dir() {
  local pattern
  for pattern in "$@"; do
    # shellcheck disable=SC2206
    local matches=( ${pattern} )
    for d in "${matches[@]:-}"; do
      if [[ -d "${d}" ]]; then
        echo "${d}"
        return 0
      fi
    done
  done
  return 1
}

MLIR_DIR="${MLIR_DIR:-}"
LLVM_DIR="${LLVM_DIR:-}"

if [[ -z "${MLIR_DIR}" ]]; then
  MLIR_DIR=$(detect_pkg_dir \
    "${LLVM_SRC_DIR}/install-*/lib/cmake/mlir" \
    "${LLVM_SRC_DIR}/build-*/lib/cmake/mlir") || {
      echo "ERROR: Could not locate MLIR CMake package (MLIR_DIR)." >&2
      echo "Set MLIR_DIR to <llvm-build-or-install>/lib/cmake/mlir and retry." >&2
      exit 1
    }
fi

if [[ -z "${LLVM_DIR}" ]]; then
  LLVM_DIR=$(detect_pkg_dir \
    "${LLVM_SRC_DIR}/install-*/lib/cmake/llvm" \
    "${LLVM_SRC_DIR}/build-*/lib/cmake/llvm") || {
      echo "ERROR: Could not locate LLVM CMake package (LLVM_DIR)." >&2
      echo "Set LLVM_DIR to <llvm-build-or-install>/lib/cmake/llvm and retry." >&2
      exit 1
    }
fi

echo "==> Using MLIR_DIR: ${MLIR_DIR}"
echo "==> Using LLVM_DIR: ${LLVM_DIR}"

mkdir -p "${BUILD_DIR}"

echo "==> Configuring CMake..."
cmake -G "${GENERATOR}" \
  -S "${ROOT_DIR}" \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}"

echo "==> Building..."
cmake --build "${BUILD_DIR}" -- -j"$(nproc)"

echo "==> Done. Build tree: ${BUILD_DIR}"


#!/bin/bash

set -euo pipefail

# Build llvm-project with Clang, MLIR, and CIR enabled.
# llvm-project will be cloned to <repo_root>/llvm-project (ignored by git).
# Build and install directories are under that clone.

# Configuration (override via env vars):
#   LLVM_REPO        - git URL for llvm-project (default: https://github.com/llvm/llvm-project.git)
#   LLVM_BRANCH      - branch or tag to checkout (default: release/21.x)
#   LLVM_SRC_DIR     - where to clone llvm-project (default: <repo_root>/llvm-project)
#   BUILD_DIR        - cmake build dir (ignored; always under llvm-project)
#   INSTALL_DIR      - cmake install prefix (ignored; always under llvm-project)
#   GENERATOR        - cmake generator (default: Ninja)
#   CMAKE_BUILD_TYPE - Release/Debug/RelWithDebInfo/etc (default: Release)
#   LLVM_TARGETS_TO_BUILD - targets to build (default: host)
#   BUILD_TARGETS    - optional space-separated cmake targets to build after configuration

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd -P)"

# Use upstream llvm-project repo
LLVM_REPO="${LLVM_REPO:-https://github.com/llvm/llvm-project.git}"
LLVM_BRANCH="${LLVM_BRANCH:-release/21.x}"

# Sanitize branch for directory names
BRANCH_SLUG="${LLVM_BRANCH//\//-}"

LLVM_SRC_DIR="${LLVM_SRC_DIR:-${ROOT_DIR}/llvm-project}"
BUILD_DIR="${LLVM_SRC_DIR}/build-${BRANCH_SLUG}"
INSTALL_DIR="${LLVM_SRC_DIR}/install-${BRANCH_SLUG}"

GENERATOR="${GENERATOR:-Ninja}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
LLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-host}"

# Parallelism (default to half the cores to avoid OOM)
TOTAL_CORES="$(nproc)"
DEFAULT_CORES=$(( TOTAL_CORES / 2 ))
if (( DEFAULT_CORES < 1 )); then DEFAULT_CORES=1; fi
unset CORES
CORES="${CORES:-${DEFAULT_CORES}}"

echo "==> Repository root: ${ROOT_DIR}"
echo "==> LLVM source dir: ${LLVM_SRC_DIR}"
echo "==> LLVM branch/tag: ${LLVM_BRANCH}"
echo "==> Build directory: ${BUILD_DIR}"
echo "==> Install prefix : ${INSTALL_DIR}"
echo "==> Generator      : ${GENERATOR}"
echo "==> Build type     : ${CMAKE_BUILD_TYPE}"
echo "==> Parallel jobs  : ${CORES}"

# Build only upstream llvm-project (no external projects). If a previous cache
# had external settings, clear them using -U during configure.
echo "==> Building upstream llvm-project only (no external project)."
CLEAR_EXTERNAL_CMAKE_ARGS="-U LLVM_EXTERNAL_PROJECTS -U LLVM_EXTERNAL_OPENSHMEM_MLIR_SOURCE_DIR"

# Ensure generator tools exist
if [[ "${GENERATOR}" == "Ninja" ]]; then
  if ! command -v ninja >/dev/null 2>&1; then
    echo "ERROR: Ninja is not installed or not on PATH." >&2
    echo "Please install ninja-build and retry, or set GENERATOR=Unix Makefiles." >&2
    exit 1
  fi
fi

# Clone or update llvm-project
if [[ ! -d "${LLVM_SRC_DIR}/.git" ]]; then
  echo "==> Cloning llvm-project (${LLVM_BRANCH})..."
  git clone --depth 1 --branch "${LLVM_BRANCH}" "${LLVM_REPO}" "${LLVM_SRC_DIR}"
else
  echo "==> Updating existing llvm-project checkout (shallow fetch of ${LLVM_BRANCH})..."
  set +e
  git -C "${LLVM_SRC_DIR}" fetch --depth 1 origin "${LLVM_BRANCH}"
  FETCH_STATUS=$?
  set -e
  if [[ ${FETCH_STATUS} -ne 0 ]]; then
    echo "==> Shallow fetch failed; re-cloning a fresh shallow copy..."
    rm -rf "${LLVM_SRC_DIR}"
    git clone --depth 1 --branch "${LLVM_BRANCH}" "${LLVM_REPO}" "${LLVM_SRC_DIR}"
  else
    git -C "${LLVM_SRC_DIR}" checkout --detach FETCH_HEAD
  fi
fi

mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

echo "==> Configuring CMake..."
cmake -G "${GENERATOR}" \
  -S "${LLVM_SRC_DIR}/llvm" \
  -B "${BUILD_DIR}" \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD}" \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_PLUGINS=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DCLANG_ENABLE_CIR=ON \
  ${CLEAR_EXTERNAL_CMAKE_ARGS} \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

echo "==> Building..."
if [[ -n "${BUILD_TARGETS:-}" ]]; then
  cmake --build "${BUILD_DIR}" --target ${BUILD_TARGETS} -- -j"${CORES}"
else
  cmake --build "${BUILD_DIR}" -- -j"${CORES}"
fi

cat <<EOF
==> Done.
Build tree: ${BUILD_DIR}
Install prefix (not installed yet): ${INSTALL_DIR}

To install (optional):
  cmake --build "${BUILD_DIR}" --target install -- -j"${CORES}"

After installing, prepend to your PATH for tools like mlir-opt and cir-opt:
  export PATH="${INSTALL_DIR}/bin:\$PATH"

You can override defaults via env vars, for example:
  LLVM_BRANCH=release/21.x GENERATOR=Ninja CMAKE_BUILD_TYPE=RelWithDebInfo \
  ${BASH_SOURCE[0]}
EOF

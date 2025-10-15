#!/bin/bash
set -euo pipefail
# Build the ClangIR incubator repository (https://github.com/llvm/clangir) as a parallel toolchain.
# Clone into ./clangir (default) and build under ./clangir/build-main.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd -P)"
CLANGIR_REPO="${CLANGIR_REPO:-https://github.com/llvm/clangir.git}"
CLANGIR_BRANCH="${CLANGIR_BRANCH:-main}"
CLANGIR_SRC_DIR="${CLANGIR_SRC_DIR:-${ROOT_DIR}/clangir}"
CLANGIR_BUILD_DIR="${CLANGIR_BUILD_DIR:-${CLANGIR_SRC_DIR}/build-main}"
GENERATOR="${GENERATOR:-Ninja}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
TOTAL_CORES="$(nproc)"
DEFAULT_CORES=$(( TOTAL_CORES / 2 ))
if (( DEFAULT_CORES < 1 )); then DEFAULT_CORES=1; fi
# Allow override via env; default to half the cores
CORES="${CORES:-${DEFAULT_CORES}}"
# Sanity bounds
if ! [[ "$CORES" =~ ^[0-9]+$ ]]; then CORES=${DEFAULT_CORES}; fi
if (( CORES < 1 )); then CORES=1; fi
if (( CORES > TOTAL_CORES )); then CORES=${TOTAL_CORES}; fi

echo "==> ClangIR source : ${CLANGIR_SRC_DIR}"
echo "==> Branch        : ${CLANGIR_BRANCH}"
echo "==> Build dir     : ${CLANGIR_BUILD_DIR}"
echo "==> Parallel jobs : ${CORES}"

if [[ ! -d "${CLANGIR_SRC_DIR}/.git" ]]; then
  echo "==> Cloning ClangIR..."
  git clone --depth 1 --branch "${CLANGIR_BRANCH}" "${CLANGIR_REPO}" "${CLANGIR_SRC_DIR}"
else
  echo "==> Updating ClangIR..."
  set +e
  git -C "${CLANGIR_SRC_DIR}" fetch --depth 1 origin "${CLANGIR_BRANCH}" && \
  git -C "${CLANGIR_SRC_DIR}" checkout --detach FETCH_HEAD
  set -e
fi

mkdir -p "${CLANGIR_BUILD_DIR}"

echo "==> Configuring..."
cmake -G "${GENERATOR}" \
  -S "${CLANGIR_SRC_DIR}/llvm" \
  -B "${CLANGIR_BUILD_DIR}" \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DCLANG_ENABLE_CIR=ON \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_PLUGINS=ON \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON

echo "==> Building..."
cmake --build "${CLANGIR_BUILD_DIR}" -- -j"${CORES}"

cat <<EOF
==> ClangIR build complete
Toolchain root: ${CLANGIR_BUILD_DIR}
Add to PATH:
  export PATH="${CLANGIR_BUILD_DIR}/bin:\$PATH"

Use with OpenSHMEM build:
  LLVM_SRC_DIR="${CLANGIR_SRC_DIR}" MLIR_DIR="${CLANGIR_BUILD_DIR}/lib/cmake/mlir" \\
  LLVM_DIR="${CLANGIR_BUILD_DIR}/lib/cmake/llvm" ./scripts/build_openshmem_mlir_clangir.sh

Step 2/3 plugin example:
  export EXTRA_STEP2_FLAGS="--load-dialect-plugin ${CLANGIR_BUILD_DIR}/lib/libCIRDialect.so"
  export EXTRA_STEP3_FLAGS="--load-dialect-plugin ${CLANGIR_BUILD_DIR}/lib/libCIRDialect.so --load-pass-plugin ${CLANGIR_BUILD_DIR}/lib/libCIRTransforms.so"
EOF

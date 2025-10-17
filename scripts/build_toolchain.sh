#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/scripts/lib/toolchain.sh"

print_usage() {
  cat <<EOF
Usage: $0 [options]

Build an LLVM/Clang toolchain used by the OpenSHMEM MLIR project.

Options:
  -t, --toolchain <id>   Toolchain to build (default: incubator)
  -j, --jobs <N>         Override parallel build jobs (defaults to nproc/2)
  -h, --help             Show this help message

Environment overrides:
  LLVM_REPO, LLVM_BRANCH, LLVM_SRC_DIR, LLVM_BUILD_DIR, LLVM_INSTALL_DIR
  CLANGIR_REPO, CLANGIR_BRANCH, CLANGIR_SRC_DIR, CLANGIR_BUILD_DIR, CLANGIR_INSTALL_DIR
  GENERATOR, CMAKE_BUILD_TYPE, LLVM_TARGETS_TO_BUILD, BUILD_TARGETS
  CORES (parallelism), any other CMake cache variables via CMAKE_ARGS

$(toolchain_help_text)
EOF
}

TOOLCHAIN="incubator"
CLI_JOBS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--toolchain)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --toolchain requires an argument" >&2
        exit 1
      fi
      TOOLCHAIN="$2"
      shift 2
      ;;
    -j|--jobs)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --jobs requires an argument" >&2
        exit 1
      fi
      CLI_JOBS="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option '$1'" >&2
      echo "Run with --help for usage." >&2
      exit 1
      ;;
  esac
done

toolchain_resolve "${TOOLCHAIN}"

CORES="${CORES:-}"
REQUESTED_JOBS="${CLI_JOBS:-${CORES}}"
JOBS=$(toolchain_resolve_jobs "${REQUESTED_JOBS}")
GENERATOR="${GENERATOR:-Ninja}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

if [[ "${GENERATOR}" == "Ninja" ]] && ! command -v ninja >/dev/null 2>&1; then
  echo "ERROR: Ninja is not installed. Install ninja-build or set GENERATOR=Unix Makefiles." >&2
  exit 1
fi

echo "==> Building toolchain"
toolchain_print_summary
cat <<EOF
  Generator     : ${GENERATOR}
  Build type    : ${CMAKE_BUILD_TYPE}
  Parallel jobs : ${JOBS}
EOF

build_upstream() {
  local repo="${LLVM_REPO:-https://github.com/llvm/llvm-project.git}"
  local branch="${LLVM_BRANCH:-${TC_BRANCH}}"
  local build_targets="${BUILD_TARGETS:-}"

  printf "\n==> Preparing llvm-project (%s)\n" "${branch}"
  if [[ ! -d "${TC_SRC_DIR}/.git" ]]; then
    echo "==> Cloning repository..."
    git clone --depth 1 --branch "${branch}" "${repo}" "${TC_SRC_DIR}"
  else
    echo "==> Updating existing checkout..."
    set +e
    git -C "${TC_SRC_DIR}" fetch --depth 1 origin "${branch}"
    local fetch_status=$?
    set -e
    if [[ ${fetch_status} -ne 0 ]]; then
      echo "==> Fetch failed, re-cloning fresh copy..."
      rm -rf "${TC_SRC_DIR}"
      git clone --depth 1 --branch "${branch}" "${repo}" "${TC_SRC_DIR}"
    else
      git -C "${TC_SRC_DIR}" checkout --detach FETCH_HEAD
    fi
  fi

  mkdir -p "${TC_BUILD_DIR}" "${TC_INSTALL_DIR}"

  printf "\n==> Configuring CMake...\n"
  local clear_external="-U LLVM_EXTERNAL_PROJECTS -U LLVM_EXTERNAL_OPENSHMEM_MLIR_SOURCE_DIR"
  cmake -G "${GENERATOR}" \
    -S "${TC_SRC_DIR}/llvm" \
    -B "${TC_BUILD_DIR}" \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-host}" \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_PLUGINS=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DCLANG_ENABLE_CIR=ON \
    ${clear_external} \
    -DCMAKE_INSTALL_PREFIX="${TC_INSTALL_DIR}" \
    ${CMAKE_ARGS:-}

  printf "\n==> Building...\n"
  if [[ -n "${build_targets}" ]]; then
    cmake --build "${TC_BUILD_DIR}" --target ${build_targets} -- -j"${JOBS}"
  else
    cmake --build "${TC_BUILD_DIR}" -- -j"${JOBS}"
  fi

  cat <<EOF

==> llvm-project build complete
Build tree : ${TC_BUILD_DIR}
Install dir: ${TC_INSTALL_DIR} (run 'cmake --build "${TC_BUILD_DIR}" --target install' to populate)

Add toolchain to PATH:
  export PATH="${TC_BIN_DIR}:\$PATH"
EOF
}

build_incubator() {
  local repo="${CLANGIR_REPO:-https://github.com/llvm/clangir.git}"
  local branch="${CLANGIR_BRANCH:-${TC_BRANCH}}"

  printf "\n==> Preparing clangir incubator (%s)\n" "${branch}"
  if [[ ! -d "${TC_SRC_DIR}/.git" ]]; then
    echo "==> Cloning repository..."
    git clone --depth 1 --branch "${branch}" "${repo}" "${TC_SRC_DIR}"
  else
    echo "==> Updating existing checkout..."
    set +e
    git -C "${TC_SRC_DIR}" fetch --depth 1 origin "${branch}" && \
      git -C "${TC_SRC_DIR}" checkout --detach FETCH_HEAD
    set -e
  fi

  mkdir -p "${TC_BUILD_DIR}" "${TC_INSTALL_DIR}"

  printf "\n==> Configuring CMake...\n"
  cmake -G "${GENERATOR}" \
    -S "${TC_SRC_DIR}/llvm" \
    -B "${TC_BUILD_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${TC_INSTALL_DIR}" \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DCLANG_ENABLE_CIR=ON \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_PLUGINS=ON \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    ${CMAKE_ARGS:-}

  printf "\n==> Building...\n"
  cmake --build "${TC_BUILD_DIR}" -- -j"${JOBS}"

  cat <<EOF

==> ClangIR incubator build complete
Toolchain root: ${TC_BUILD_DIR}
Add to PATH:
  export PATH="${TC_BIN_DIR}:\$PATH"

Use with OpenSHMEM build:
  ./scripts/build_openshmem_mlir.sh --toolchain incubator
EOF
}

case "${TC_ID}" in
  upstream)
    build_upstream
    ;;
  incubator)
    build_incubator
    ;;
  *)
    echo "ERROR: Unsupported toolchain '${TC_ID}'" >&2
    exit 1
    ;;
esac

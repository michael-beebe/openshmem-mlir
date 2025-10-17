#!/bin/bash
# shellcheck shell=bash

# Shared helpers for managing LLVM/Clang toolchains used by the OpenSHMEM MLIR
# project. Sourcing this file defines utility functions and populates the
# following global variables after calling toolchain_resolve <id>:
#   TC_ID, TC_DESC, TC_BRANCH, TC_BRANCH_SLUG
#   TC_SRC_DIR, TC_BUILD_DIR, TC_INSTALL_DIR
#   TC_LLVM_PROJECT_SOURCE_DIR, TC_LLVM_PROJECT_BUILD_DIR
#   TC_CLANGIR_SRC_DIR, TC_CLANGIR_BUILD_DIR (incubator only)
#   TC_LLVM_DIR, TC_MLIR_DIR, TC_BIN_DIR, TC_CIR_OPT, TC_CLANG
#   TC_PROJECT_BUILD_DIR_DEFAULT

if [[ -n "${OPENSHMEM_TOOLCHAIN_LIB_SOURCED:-}" ]]; then
  return 0
fi
OPENSHMEM_TOOLCHAIN_LIB_SOURCED=1

# Compute repository root: <repo>/scripts/lib/toolchain.sh -> go up two levels.
_TOOLCHAIN_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${ROOT_DIR:-$(cd "${_TOOLCHAIN_LIB_DIR}/../.." && pwd -P)}"

#------------------------------------------------------------------------------
# Public helper functions
#------------------------------------------------------------------------------

toolchain_available() {
  echo "upstream incubator"
}

toolchain_require() {
  local id="$1"
  case "${id}" in
    upstream|incubator) ;;
    *)
      echo "ERROR: Unknown toolchain '${id}'. Valid choices: $(toolchain_available)" >&2
      exit 1
      ;;
  esac
}

toolchain_help_text() {
  cat <<EOF
Valid toolchains:
  upstream   - llvm-project (release/21.x by default) with CLANG_ENABLE_CIR=ON
  incubator  - clangir incubator repository (main branch)
EOF
}

toolchain_default_jobs() {
  local total
  total=$(nproc)
  local half=$(( total / 2 ))
  if (( half < 1 )); then
    half=1
  fi
  echo "${half}"
}

toolchain_resolve_jobs() {
  local requested="$1"
  local total
  total=$(nproc)
  local jobs

  if [[ -z "${requested}" ]]; then
    jobs=$(toolchain_default_jobs)
  else
    jobs="${requested}"
  fi

  if ! [[ "${jobs}" =~ ^[0-9]+$ ]]; then
    jobs=$(toolchain_default_jobs)
  fi

  if (( jobs < 1 )); then
    jobs=1
  fi

  if (( jobs > total )); then
    jobs=${total}
  fi

  echo "${jobs}"
}

# Populate TC_* globals describing the selected toolchain.
toolchain_resolve() {
  local id="$1"
  toolchain_require "${id}"

  # Clear previously-set globals to avoid leaking stale values when resolving
  # multiple toolchains in the same shell.
  for var in TC_ID TC_DESC TC_BRANCH TC_BRANCH_SLUG TC_SRC_DIR TC_BUILD_DIR \
             TC_INSTALL_DIR TC_LLVM_PROJECT_SOURCE_DIR \
             TC_LLVM_PROJECT_BUILD_DIR TC_CLANGIR_SRC_DIR TC_CLANGIR_BUILD_DIR \
             TC_LLVM_DIR TC_MLIR_DIR TC_BIN_DIR TC_CIR_OPT TC_CLANG \
             TC_PROJECT_BUILD_DIR_DEFAULT; do
    unset "${var}"
  done

  case "${id}" in
    upstream)
      local branch="${LLVM_BRANCH:-release/21.x}"
      local branch_slug="${branch//\//-}"
      local src="${LLVM_SRC_DIR:-${ROOT_DIR}/llvm-project}"
      local build="${LLVM_BUILD_DIR:-${LLVM_PROJECT_BUILD_DIR:-${src}/build-${branch_slug}}}"
      local install="${LLVM_INSTALL_DIR:-${LLVM_PROJECT_INSTALL_DIR:-${src}/install-${branch_slug}}}"
      local bin_dir="${build}/bin"

      local default_llvm_build="${build}/lib/cmake/llvm"
      local default_llvm_install="${install}/lib/cmake/llvm"
      local default_mlir_build="${build}/lib/cmake/mlir"
      local default_mlir_install="${install}/lib/cmake/mlir"

      local llvm_dir="${LLVM_DIR:-}"
      if [[ -z "${llvm_dir}" ]]; then
        if [[ -d "${default_llvm_build}" ]]; then
          llvm_dir="${default_llvm_build}"
        else
          llvm_dir="${default_llvm_install}"
        fi
      fi
      if [[ -z "${llvm_dir}" ]]; then
        llvm_dir="${default_llvm_build}"
      fi

      local mlir_dir="${MLIR_DIR:-}"
      if [[ -z "${mlir_dir}" ]]; then
        if [[ -d "${default_mlir_build}" ]]; then
          mlir_dir="${default_mlir_build}"
        else
          mlir_dir="${default_mlir_install}"
        fi
      fi
      if [[ -z "${mlir_dir}" ]]; then
        mlir_dir="${default_mlir_build}"
      fi

      declare -g TC_ID="${id}"
      declare -g TC_DESC="Upstream llvm-project"
      declare -g TC_BRANCH="${branch}"
      declare -g TC_BRANCH_SLUG="${branch_slug}"
      declare -g TC_SRC_DIR="${src}"
      declare -g TC_BUILD_DIR="${build}"
      declare -g TC_INSTALL_DIR="${install}"
      declare -g TC_LLVM_PROJECT_SOURCE_DIR="${src}"
      declare -g TC_LLVM_PROJECT_BUILD_DIR="${build}"
      declare -g TC_CLANGIR_SRC_DIR=""
      declare -g TC_CLANGIR_BUILD_DIR=""
      declare -g TC_LLVM_DIR="${llvm_dir}"
      declare -g TC_MLIR_DIR="${mlir_dir}"
      declare -g TC_BIN_DIR="${bin_dir}"
      declare -g TC_CIR_OPT="${bin_dir}/cir-opt"
      declare -g TC_CLANG="${bin_dir}/clang"
      declare -g TC_PROJECT_BUILD_DIR_DEFAULT="${ROOT_DIR}/build-upstream"
      ;;
    incubator)
      local branch="${CLANGIR_BRANCH:-main}"
      local src="${CLANGIR_SRC_DIR:-${ROOT_DIR}/clangir}"
      local build="${CLANGIR_BUILD_DIR:-${src}/build-main}"
      local install="${CLANGIR_INSTALL_DIR:-${src}/install}"
      local bin_dir="${build}/bin"

      local default_llvm_build="${build}/lib/cmake/llvm"
      local default_llvm_install="${install}/lib/cmake/llvm"
      local default_mlir_build="${build}/lib/cmake/mlir"
      local default_mlir_install="${install}/lib/cmake/mlir"

      local llvm_dir="${LLVM_DIR:-}"
      if [[ -z "${llvm_dir}" ]]; then
        if [[ -d "${default_llvm_build}" ]]; then
          llvm_dir="${default_llvm_build}"
        else
          llvm_dir="${default_llvm_install}"
        fi
      fi
      if [[ -z "${llvm_dir}" ]]; then
        llvm_dir="${default_llvm_build}"
      fi

      local mlir_dir="${MLIR_DIR:-}"
      if [[ -z "${mlir_dir}" ]]; then
        if [[ -d "${default_mlir_build}" ]]; then
          mlir_dir="${default_mlir_build}"
        else
          mlir_dir="${default_mlir_install}"
        fi
      fi
      if [[ -z "${mlir_dir}" ]]; then
        mlir_dir="${default_mlir_build}"
      fi

      declare -g TC_ID="${id}"
      declare -g TC_DESC="ClangIR incubator"
      declare -g TC_BRANCH="${branch}"
      declare -g TC_BRANCH_SLUG="${branch//\//-}"
      declare -g TC_SRC_DIR="${src}"
      declare -g TC_BUILD_DIR="${build}"
      declare -g TC_INSTALL_DIR="${install}"
      declare -g TC_LLVM_PROJECT_SOURCE_DIR="${src}"
      declare -g TC_LLVM_PROJECT_BUILD_DIR="${build}"
      declare -g TC_CLANGIR_SRC_DIR="${src}"
      declare -g TC_CLANGIR_BUILD_DIR="${build}"
      declare -g TC_LLVM_DIR="${llvm_dir}"
      declare -g TC_MLIR_DIR="${mlir_dir}"
      declare -g TC_BIN_DIR="${bin_dir}"
      declare -g TC_CIR_OPT="${bin_dir}/cir-opt"
      declare -g TC_CLANG="${bin_dir}/clang"
      declare -g TC_PROJECT_BUILD_DIR_DEFAULT="${ROOT_DIR}/build-clangir"
      ;;
  esac
}

toolchain_print_summary() {
  echo "Toolchain       : ${TC_ID} (${TC_DESC})"
  if [[ -n "${TC_BRANCH:-}" ]]; then
    echo "  Branch        : ${TC_BRANCH}"
  fi
  echo "  Source dir    : ${TC_SRC_DIR}"
  echo "  Build dir     : ${TC_BUILD_DIR}"
  if [[ -n "${TC_INSTALL_DIR}" ]]; then
    echo "  Install dir   : ${TC_INSTALL_DIR}"
  fi
  echo "  LLVM_DIR      : ${TC_LLVM_DIR}"
  echo "  MLIR_DIR      : ${TC_MLIR_DIR}"
  echo "  bin/          : ${TC_BIN_DIR}"
}

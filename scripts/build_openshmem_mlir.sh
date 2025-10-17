#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/scripts/lib/toolchain.sh"

print_usage() {
  cat <<USAGE
Usage: $0 [options]

Configure and build the OpenSHMEM MLIR project against a selected LLVM/Clang toolchain.

Options:
  -t, --toolchain <id>   Toolchain to use (default: incubator)
  -b, --build-dir <dir>  Build directory (default: build-<toolchain>)
  -j, --jobs <N>         Parallel build jobs (default: nproc/2)
  -G, --generator <gen>  CMake generator (default: Ninja)
  -h, --help             Show this help message

Environment overrides:
  MLIR_DIR, LLVM_DIR, CMAKE_BUILD_TYPE, GENERATOR, BUILD_DIR, CORES,
  CMAKE_ARGS, plus toolchain-specific variables (see build_toolchain.sh --help).
USAGE
}

TOOLCHAIN="incubator"
CLI_BUILD_DIR=""
CLI_JOBS=""
CLI_GENERATOR=""

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
    -b|--build-dir)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --build-dir requires an argument" >&2
        exit 1
      fi
      CLI_BUILD_DIR="$2"
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
    -G|--generator)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --generator requires an argument" >&2
        exit 1
      fi
      CLI_GENERATOR="$2"
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

ENV_CORES="${CORES:-}"
REQUESTED_JOBS="${CLI_JOBS:-${ENV_CORES}}"
JOBS=$(toolchain_resolve_jobs "${REQUESTED_JOBS}")

ENV_GENERATOR="${GENERATOR:-}"
GENERATOR="${CLI_GENERATOR:-${ENV_GENERATOR:-Ninja}}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

DEFAULT_BUILD_DIR="${TC_PROJECT_BUILD_DIR_DEFAULT}"
ENV_BUILD_DIR="${BUILD_DIR:-}"
BUILD_DIR="${CLI_BUILD_DIR:-${ENV_BUILD_DIR:-${DEFAULT_BUILD_DIR}}}"

ENV_MLIR_DIR="${MLIR_DIR:-}"
MLIR_DIR="${ENV_MLIR_DIR:-${TC_MLIR_DIR}}"

ENV_LLVM_DIR="${LLVM_DIR:-}"
LLVM_DIR="${ENV_LLVM_DIR:-${TC_LLVM_DIR}}"

if [[ "${GENERATOR}" == "Ninja" ]] && ! command -v ninja >/dev/null 2>&1; then
  echo "ERROR: Ninja is not installed. Install ninja-build or choose another generator." >&2
  exit 1
fi

if [[ -z "${MLIR_DIR}" ]] || [[ ! -d "${MLIR_DIR}" ]]; then
  echo "ERROR: MLIR_DIR not found. Set MLIR_DIR explicitly." >&2
  echo "  Expected default: ${TC_MLIR_DIR}" >&2
  exit 1
fi

if [[ -z "${LLVM_DIR}" ]] || [[ ! -d "${LLVM_DIR}" ]]; then
  echo "ERROR: LLVM_DIR not found. Set LLVM_DIR explicitly." >&2
  echo "  Expected default: ${TC_LLVM_DIR}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"

echo "==> Configuring OpenSHMEM MLIR"
toolchain_print_summary
cat <<SUMMARY
  Build dir     : ${BUILD_DIR}
  Generator     : ${GENERATOR}
  Build type    : ${CMAKE_BUILD_TYPE}
  Parallel jobs : ${JOBS}
  MLIR_DIR      : ${MLIR_DIR}
  LLVM_DIR      : ${LLVM_DIR}
SUMMARY

cmake_arguments=(
  -G "${GENERATOR}"
  -S "${ROOT_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  -DMLIR_DIR="${MLIR_DIR}"
  -DLLVM_DIR="${LLVM_DIR}"
)

if [[ -n "${TC_LLVM_PROJECT_SOURCE_DIR}" ]]; then
  cmake_arguments+=( -DLLVM_PROJECT_SOURCE_DIR="${TC_LLVM_PROJECT_SOURCE_DIR}" )
fi
if [[ -n "${TC_LLVM_PROJECT_BUILD_DIR}" ]]; then
  cmake_arguments+=( -DLLVM_PROJECT_BUILD_DIR="${TC_LLVM_PROJECT_BUILD_DIR}" )
fi
if [[ -n "${TC_CLANGIR_SRC_DIR}" ]]; then
  cmake_arguments+=( -DCLANGIR_SRC_DIR="${TC_CLANGIR_SRC_DIR}" )
fi
if [[ -n "${TC_CLANGIR_BUILD_DIR}" ]]; then
  cmake_arguments+=( -DCLANGIR_BUILD_DIR="${TC_CLANGIR_BUILD_DIR}" )
fi

if [[ -n "${CMAKE_ARGS:-}" ]]; then
  cmake_arguments+=( ${CMAKE_ARGS} )
fi

printf "\n==> Running CMake configure...\n"
cmake "${cmake_arguments[@]}"

printf "\n==> Building...\n"
cmake --build "${BUILD_DIR}" -- -j"${JOBS}"

cat <<SUMMARY

==> Build complete
Project build tree: ${BUILD_DIR}

Use this build with:
  ./scripts/test_end_to_end.sh --toolchain ${TOOLCHAIN}
SUMMARY

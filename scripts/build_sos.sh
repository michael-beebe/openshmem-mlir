#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd -P)"
source "${ROOT_DIR}/scripts/lib/toolchain.sh"
print_usage() {
  cat <<EOF
Usage: $0 [options]

Build libfabric and Sandia OpenSHMEM (SOS). By default the script uses a local
LLVM/Clang toolchain built via build_toolchain.sh.

Options:
  --toolchain <id>   Choose toolchain (upstream|incubator|auto). Default: auto
  --use-local        Force use of local clang toolchain (same as USE_LOCAL_CLANG=1)
  --use-system       Build with system compilers (same as USE_LOCAL_CLANG=0)
  --help             Show this message

Environment overrides:
  LIBFABRIC_VERSION, SOS_VERSION, RUNTIME_DIR, TOOLCHAIN, USE_LOCAL_CLANG,
  LLVM_BUILD_DIR, CORES, plus the settings documented in build_toolchain.sh.
EOF
}

# Build libfabric and Sandia OpenSHMEM (SOS) using local LLVM clang.
# This provides oshcc wrapper and libshmem for testing OpenSHMEM programs.
#
# - libfabric-<version>/
# - SOS-<version>/
#
# Configuration (override via env vars):
#   LIBFABRIC_VERSION - libfabric version (default: 1.15.1)
#   SOS_VERSION       - SOS git tag (default: v1.5.2)
#   RUNTIME_DIR       - where to build/install (default: <repo_root>/openshmem-runtime)
#   USE_LOCAL_CLANG   - use clang from llvm-project build (default: 1)
#   TOOLCHAIN         - which LLVM toolchain to use when USE_LOCAL_CLANG=1 (clangir|upstream), default: auto-detect
#   LLVM_BUILD_DIR    - explicit path to LLVM build (with bin/clang); overrides TOOLCHAIN and auto-detect
#   CORES             - parallel build jobs (default: nproc/2)

LIBFABRIC_VERSION="${LIBFABRIC_VERSION:-1.15.1}"
SOS_VERSION="${SOS_VERSION:-v1.5.2}"
RUNTIME_DIR="${RUNTIME_DIR:-${ROOT_DIR}/openshmem-runtime}"
USE_LOCAL_CLANG="${USE_LOCAL_CLANG:-1}"
TOOLCHAIN="${TOOLCHAIN:-}"

SCRIPT_TOOLCHAIN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --toolchain)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --toolchain requires an argument" >&2
        exit 1
      fi
      SCRIPT_TOOLCHAIN="$2"
      shift 2
      ;;
    --use-local)
      USE_LOCAL_CLANG=1
      shift
      ;;
    --use-system)
      USE_LOCAL_CLANG=0
      shift
      ;;
    --help|-h)
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


if [[ -n "${SCRIPT_TOOLCHAIN}" ]]; then
  TOOLCHAIN="${SCRIPT_TOOLCHAIN}"
fi

TOOLCHAIN="${TOOLCHAIN:-auto}"

DEFAULT_CORES="$(toolchain_default_jobs)"
ENV_CORES="${CORES:-${DEFAULT_CORES}}"
CORES="$(toolchain_resolve_jobs "${ENV_CORES}")"
# Paths for libfabric and SOS
LIBFABRIC_DIR="${RUNTIME_DIR}/libfabric-${LIBFABRIC_VERSION}"
SOS_DIR="${RUNTIME_DIR}/SOS-${SOS_VERSION}"
BUILD_TEMP="${RUNTIME_DIR}/build-temp"

echo "==> Repository root: ${ROOT_DIR}"
echo "==> Runtime directory: ${RUNTIME_DIR}"
echo "==> libfabric version: ${LIBFABRIC_VERSION}"
echo "==> SOS version: ${SOS_VERSION}"
echo "==> Parallel jobs: ${CORES}"

# Determine which compiler to use
if [[ "${USE_LOCAL_CLANG}" == "1" ]]; then
  LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-}"
  SELECTED_TOOLCHAIN=""

  if [[ -n "${LLVM_BUILD_DIR}" ]]; then
    SELECTED_TOOLCHAIN="custom"
  else
    if [[ "${TOOLCHAIN}" == "auto" ]]; then
      for candidate in incubator upstream; do
        toolchain_resolve "${candidate}"
        candidate_root="${TC_BIN_DIR%/bin}"
        if [[ -x "${TC_CLANG}" ]]; then
          LLVM_BUILD_DIR="${candidate_root}"
          SELECTED_TOOLCHAIN="${candidate}"
          break
        fi
      done
    else
      toolchain_require "${TOOLCHAIN}"
      toolchain_resolve "${TOOLCHAIN}"
      LLVM_BUILD_DIR="${TC_BIN_DIR%/bin}"
      SELECTED_TOOLCHAIN="${TC_ID}"
    fi
  fi

  if [[ -z "${LLVM_BUILD_DIR}" ]]; then
    echo "ERROR: Could not locate a local LLVM build with clang." >&2
    if [[ "${TOOLCHAIN}" == "auto" ]]; then
      echo "Tried toolchains: $(toolchain_available)" >&2
      echo "Run ./scripts/build_toolchain.sh --toolchain <id> to build one." >&2
    else
      echo "Selected toolchain '${TOOLCHAIN}' is not available. Run ./scripts/build_toolchain.sh --toolchain ${TOOLCHAIN}." >&2
    fi
    echo "Alternatively, set LLVM_BUILD_DIR or USE_LOCAL_CLANG=0." >&2
    exit 1
  fi

  CLANG="${LLVM_BUILD_DIR}/bin/clang"
  CLANGXX="${LLVM_BUILD_DIR}/bin/clang++"

  if [[ ! -x "${CLANG}" ]]; then
    echo "ERROR: clang not found at ${CLANG}" >&2
    exit 1
  fi

  if [[ -z "${SELECTED_TOOLCHAIN}" ]]; then
    if [[ "${LLVM_BUILD_DIR}" == *"/clangir/"* ]]; then
      SELECTED_TOOLCHAIN="incubator"
    elif [[ "${LLVM_BUILD_DIR}" == *"/llvm-project/"* ]]; then
      SELECTED_TOOLCHAIN="upstream"
    else
      SELECTED_TOOLCHAIN="custom"
    fi
  fi

  echo "==> Using local clang from: ${LLVM_BUILD_DIR} (toolchain: ${SELECTED_TOOLCHAIN})"
  echo "    - CC:  ${CLANG}"
  echo "    - CXX: ${CLANGXX}"
  export CC="${CLANG}"
  export CXX="${CLANGXX}"
else
  echo "==> Using system compiler"
  export CC="${CC:-gcc}"
  export CXX="${CXX:-g++}"
fi

echo "==> CC=${CC}"
echo "==> CXX=${CXX}"

# Create directories
mkdir -p "${RUNTIME_DIR}" "${BUILD_TEMP}"

#==============================================================================
# Build libfabric
#==============================================================================

echo ""
echo "==> Building libfabric ${LIBFABRIC_VERSION}..."

LIBFABRIC_TARBALL="libfabric-${LIBFABRIC_VERSION}.tar.bz2"
LIBFABRIC_URL="https://github.com/ofiwg/libfabric/releases/download/v${LIBFABRIC_VERSION}/${LIBFABRIC_TARBALL}"
LIBFABRIC_SRC="${BUILD_TEMP}/libfabric-${LIBFABRIC_VERSION}"

# Download libfabric if not already present
if [[ ! -f "${BUILD_TEMP}/${LIBFABRIC_TARBALL}" ]]; then
  echo "==> Downloading libfabric..."
  wget -P "${BUILD_TEMP}" "${LIBFABRIC_URL}"
fi

# Extract libfabric
if [[ ! -d "${LIBFABRIC_SRC}" ]]; then
  echo "==> Extracting libfabric..."
  tar -xjf "${BUILD_TEMP}/${LIBFABRIC_TARBALL}" -C "${BUILD_TEMP}"
fi

# Build libfabric
cd "${LIBFABRIC_SRC}"

echo "==> Configuring libfabric..."
./configure --prefix="${LIBFABRIC_DIR}"

echo "==> Building libfabric..."
make -j"${CORES}"

echo "==> Installing libfabric to ${LIBFABRIC_DIR}..."
rm -rf "${LIBFABRIC_DIR}"
make install -j"${CORES}"

echo "libfabric installed successfully"

#==============================================================================
# Build Sandia OpenSHMEM (SOS)
#==============================================================================

echo ""
echo "==> Building Sandia OpenSHMEM (SOS) ${SOS_VERSION}..."

SOS_SRC="${BUILD_TEMP}/SOS"

# Clone SOS if not already present
if [[ ! -d "${SOS_SRC}/.git" ]]; then
  echo "==> Cloning SOS repository..."
  git clone https://github.com/Sandia-OpenSHMEM/SOS.git "${SOS_SRC}"
fi

cd "${SOS_SRC}"

# Checkout specific version
echo "==> Checking out SOS ${SOS_VERSION}..."
git fetch --tags
git checkout "tags/${SOS_VERSION}"

# Update submodules
echo "==> Updating submodules..."
git submodule update --init --recursive

# Generate configure script
echo "==> Running autogen.sh..."
./autogen.sh

# Create build directory
SOS_BUILD="${SOS_SRC}/build"
rm -rf "${SOS_BUILD}"
mkdir -p "${SOS_BUILD}"
cd "${SOS_BUILD}"

# Configure SOS
echo "==> Configuring SOS..."
echo "    - Using libfabric: ${LIBFABRIC_DIR}"
echo "    - Installing to: ${SOS_DIR}"

../configure \
  --prefix="${SOS_DIR}" \
  --with-libfabric="${LIBFABRIC_DIR}" \
  --enable-pmi-simple \
  LDFLAGS="-latomic"

# Build SOS
echo "==> Building SOS..."
make -j"${CORES}"

# Install SOS
echo "==> Installing SOS to ${SOS_DIR}..."
rm -rf "${SOS_DIR}"
make -j"${CORES}" install

# Run tests (optional, can fail in some environments)
echo "==> Running SOS tests (non-fatal)..."
make -j"${CORES}" check || echo "Some tests failed (this is often expected in non-MPI environments)"

echo "SOS installed successfully"

#==============================================================================
# Summary
#==============================================================================

cat <<EOF

==> Build complete!

Installation directories:
  libfabric: ${LIBFABRIC_DIR}
  SOS:       ${SOS_DIR}

Compilers used:
  CC:  ${CC}
  CXX: ${CXX}

To use OpenSHMEM:
  export PATH="${SOS_DIR}/bin:\$PATH"
  export LD_LIBRARY_PATH="${SOS_DIR}/lib:${LIBFABRIC_DIR}/lib:\$LD_LIBRARY_PATH"

Available tools:
  oshcc:  ${SOS_DIR}/bin/oshcc   (C compiler wrapper)
  oshc++: ${SOS_DIR}/bin/oshc++  (C++ compiler wrapper)
  oshrun: ${SOS_DIR}/bin/oshrun  (launcher)

Example usage:
  oshcc -o hello examples/hello.c
  oshrun -n 4 ./hello

You can now compile C programs with #include <shmem.h> and link with -lshmem
EOF

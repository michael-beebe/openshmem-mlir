#!/bin/bash

set -euo pipefail

# Build libfabric and Sandia OpenSHMEM (SOS) using local LLVM clang.
# This provides oshcc wrapper and libshmem for testing OpenSHMEM programs.
#
# Dependencies are built to <repo_root>/openshmem-runtime/
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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd -P)"

LIBFABRIC_VERSION="${LIBFABRIC_VERSION:-1.15.1}"
SOS_VERSION="${SOS_VERSION:-v1.5.2}"
RUNTIME_DIR="${RUNTIME_DIR:-${ROOT_DIR}/openshmem-runtime}"
USE_LOCAL_CLANG="${USE_LOCAL_CLANG:-1}"
TOOLCHAIN="${TOOLCHAIN:-}"
TOTAL_CORES="$(nproc)"
DEFAULT_CORES=$(( TOTAL_CORES / 2 ))
if (( DEFAULT_CORES < 1 )); then DEFAULT_CORES=1; fi
unset CORES
CORES="${CORES:-${DEFAULT_CORES}}"
# Sanity bounds for CORES
if ! [[ "${CORES}" =~ ^[0-9]+$ ]]; then CORES=${DEFAULT_CORES}; fi
if (( CORES < 1 )); then CORES=1; fi
if (( CORES > TOTAL_CORES )); then CORES=${TOTAL_CORES}; fi

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
  # Select LLVM toolchain: explicit path > TOOLCHAIN choice > auto-detect
  LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-}"

  if [[ -z "${LLVM_BUILD_DIR}" ]]; then
    case "${TOOLCHAIN}" in
      clangir)
        LLVM_BUILD_DIR="${ROOT_DIR}/clangir/build-main"
        ;;
      upstream)
        LLVM_BUILD_DIR="${ROOT_DIR}/llvm-project/build-release-21.x"
        ;;
      ""|*)
        # Auto-detect: prefer incubator (clangir) first, then legacy/incubator, then upstream
        CANDIDATES=(
          "${ROOT_DIR}/clangir/build-main"
          "${ROOT_DIR}/llvm-project-incubator/build-main"
          "${ROOT_DIR}/llvm-project/build-release-21.x"
        )
        for d in "${CANDIDATES[@]}"; do
          if [[ -x "${d}/bin/clang" ]]; then
            LLVM_BUILD_DIR="${d}"
            break
          fi
        done
        ;;
    esac
  fi

  if [[ -z "${LLVM_BUILD_DIR}" ]]; then
    echo "ERROR: Could not locate a local LLVM build with clang." >&2
    echo "Searched candidates (in order):" >&2
    echo "  - ${ROOT_DIR}/clangir/build-main" >&2
    echo "  - ${ROOT_DIR}/llvm-project-incubator/build-main" >&2
    echo "  - ${ROOT_DIR}/llvm-project/build-release-21.x" >&2
    echo "Options:" >&2
    echo "  - Set TOOLCHAIN=clangir or TOOLCHAIN=upstream" >&2
    echo "  - Or set LLVM_BUILD_DIR to your build (with bin/clang)" >&2
    echo "  - Or set USE_LOCAL_CLANG=0 to use system compilers" >&2
    exit 1
  fi

  CLANG="${LLVM_BUILD_DIR}/bin/clang"
  CLANGXX="${LLVM_BUILD_DIR}/bin/clang++"

  if [[ ! -x "${CLANG}" ]]; then
    echo "ERROR: clang not found at ${CLANG}" >&2
    exit 1
  fi

  # Best-effort toolchain label for logs
  if [[ "${LLVM_BUILD_DIR}" == *"/clangir/"* || "${LLVM_BUILD_DIR}" == *"/clangir/build-main" ]]; then
    SELECTED_TOOLCHAIN="clangir"
  elif [[ "${LLVM_BUILD_DIR}" == *"/llvm-project/"* ]]; then
    SELECTED_TOOLCHAIN="upstream"
  else
    SELECTED_TOOLCHAIN="custom"
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


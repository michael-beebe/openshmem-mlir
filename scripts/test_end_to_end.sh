#!/bin/bash

set -euo pipefail

# End-to-end compilation pipeline: C → CIR → OpenSHMEM MLIR → LLVM IR → Binary
#
# This script demonstrates the full compilation flow:
#   1. C code → ClangIR (using clang -fclangir)
#   2. CIR → OpenSHMEM MLIR (using shmem-mlir-opt)
#   3. OpenSHMEM MLIR → LLVM MLIR (using cir-opt and shmem-mlir-opt)
#   4. LLVM MLIR → LLVM IR (using mlir-translate)
#   5. LLVM IR → Binary (using clang with OpenSHMEM runtime)
#
# Usage:
#   ./test_end_to_end.sh [options] [input_file]
#
# Options:
#   --test <TestName>    Run a specific test by name (e.g., --test HelloWorld)
#   --clean              Remove tmp/ directory before running
#
# Examples:
#   ./test_end_to_end.sh                                    # Run default test (HelloWorld)
#   ./test_end_to_end.sh --test HelloWorld                  # Run HelloWorld test
#   ./test_end_to_end.sh --clean --test HelloWorld          # Clean then run HelloWorld
#   ./test_end_to_end.sh test/EndToEnd/Atomics/test.c       # Run specific file

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd -P)"
source "${ROOT_DIR}/scripts/lib/toolchain.sh"

# Parse command-line arguments
usage() {
  local script_name
  script_name="$(basename "$0")"
  cat <<EOF
Usage: ./${script_name} [options] [input_file]

Options:
  --test <TestName>    Run a specific test by name (e.g., --test HelloWorld)
  --clean              Remove tmp/ directory before running
  --toolchain <id>     Select LLVM toolchain (upstream, incubator, or auto)
  --help, -h           Show this help message

Environment overrides:
  TOOLCHAIN            Default toolchain selection (defaults to upstream)
  PROJECT_BUILD_DIR    Override project build directory used for binaries
  LLVM_BUILD           Override LLVM/Clang build directory
  CIR_TO_LLVM_TOOL     Path to cir-opt-compatible binary for Step 3
  CIR_TO_LLVM_PASSES   Custom pass pipeline for Step 3 conversion
EOF
}

DO_CLEAN=0
TEST_NAME_ARG=""
INPUT_FILE_ARG=""
TOOLCHAIN="${TOOLCHAIN:-upstream}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      DO_CLEAN=1
      shift
      ;;
    --toolchain)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --toolchain requires an argument" >&2
        usage >&2
        exit 1
      fi
      TOOLCHAIN="$2"
      shift 2
      ;;
    --test)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --test requires a test name argument" >&2
        usage >&2
        exit 1
      fi
      TEST_NAME_ARG="$2"
      shift 2
      ;;
    --help|-h)
      usage
      script_name="$(basename "$0")"
      echo ""
      echo "Examples:"
      echo "  ./${script_name}                                    # Run default test (HelloWorld)"
      echo "  ./${script_name} --test HelloWorld                  # Run HelloWorld test"
      echo "  ./${script_name} --clean --test HelloWorld          # Clean then run HelloWorld"
      echo "  ./${script_name} test/EndToEnd/Atomics/test.c       # Run specific file"
      exit 0
      ;;
    -*)
      echo "ERROR: Unknown option: $1" >&2
      echo "Run '$0 --help' for usage information" >&2
      exit 1
      ;;
    *)
      INPUT_FILE_ARG="$1"
      shift
      ;;
  esac
done

# Resolve toolchain selection and derive paths
if [[ "${TOOLCHAIN}" == "auto" ]]; then
  DETECTED_TOOLCHAIN=""
  for candidate in incubator upstream; do
    toolchain_resolve "${candidate}"
    if [[ -x "${TC_CLANG}" ]]; then
      TOOLCHAIN="${candidate}"
      DETECTED_TOOLCHAIN="${candidate}"
      break
    fi
  done
  if [[ -z "${DETECTED_TOOLCHAIN}" ]]; then
    echo "ERROR: --toolchain auto could not find a usable clang toolchain." >&2
    echo "Run ./scripts/build_toolchain.sh --toolchain <id> first." >&2
    exit 1
  fi
else
  toolchain_require "${TOOLCHAIN}"
fi

# Ensure globals reflect the final toolchain choice
toolchain_resolve "${TOOLCHAIN}"

DEFAULT_PROJECT_BUILD_DIR="${TC_PROJECT_BUILD_DIR_DEFAULT}"
ENV_PROJECT_BUILD_DIR="${PROJECT_BUILD_DIR:-}"
PROJECT_BUILD_DIR="${ENV_PROJECT_BUILD_DIR:-${DEFAULT_PROJECT_BUILD_DIR}}"
SHMEM_MLIR_OPT="${PROJECT_BUILD_DIR}/tools/shmem-mlir-opt/shmem-mlir-opt"

ENV_LLVM_BUILD="${LLVM_BUILD:-}"
LLVM_BUILD="${ENV_LLVM_BUILD:-${TC_BIN_DIR%/bin}}"
LLVM_BIN_DIR="${LLVM_BUILD}/bin"
CLANG_BIN="${LLVM_BIN_DIR}/clang"
MLIR_OPT_BIN="${LLVM_BIN_DIR}/mlir-opt"
MLIR_TRANSLATE_BIN="${LLVM_BIN_DIR}/mlir-translate"
LLC_BIN="${LLVM_BIN_DIR}/llc"

CIR_TO_LLVM_TOOL="${CIR_TO_LLVM_TOOL:-${TC_CIR_OPT}}"
CIR_TO_LLVM_PASSES="${CIR_TO_LLVM_PASSES:---cir-to-llvm}"
EXTRA_STEP3_FLAGS="${EXTRA_STEP3_FLAGS:-}"

# Clean tmp/ directory if requested
if [[ ${DO_CLEAN} -eq 1 ]]; then
  echo "Cleaning tmp/ directory..."
  rm -rf "${ROOT_DIR}/tmp"
  echo "  Removed: ${ROOT_DIR}/tmp"
  echo ""
fi

# Paths
SOS_VERSION="${SOS_VERSION:-v1.5.2}"
SOS_DIR="${ROOT_DIR}/openshmem-runtime/SOS-${SOS_VERSION}"

# Check prerequisites
if [[ ! -x "${CLANG_BIN}" ]]; then
  echo "ERROR: clang not found at ${CLANG_BIN}." >&2
  echo "Run ./scripts/build_toolchain.sh --toolchain ${TOOLCHAIN} first." >&2
  exit 1
fi

if [[ ! -x "${SHMEM_MLIR_OPT}" ]]; then
  echo "ERROR: shmem-mlir-opt not found. Run ./scripts/build_openshmem_mlir.sh first" >&2
  exit 1
fi

if [[ ! -x "${CIR_TO_LLVM_TOOL}" ]]; then
  echo "ERROR: cir-opt driver not found at ${CIR_TO_LLVM_TOOL}" >&2
  echo "Set CIR_TO_LLVM_TOOL to your preferred binary (e.g., <clangir-build>/bin/cir-opt)" >&2
  exit 1
fi

for tool in "${MLIR_OPT_BIN}" "${MLIR_TRANSLATE_BIN}" "${LLC_BIN}"; do
  if [[ ! -x "${tool}" ]]; then
    echo "ERROR: Required LLVM tool not found at ${tool}." >&2
    echo "Ensure your ${TOOLCHAIN} toolchain build is complete." >&2
    exit 1
  fi
done

echo "Using LLVM toolchain (${TOOLCHAIN}): ${LLVM_BUILD}"
echo "Using project build       : ${PROJECT_BUILD_DIR}"

if [[ ! -d "${SOS_DIR}" ]]; then
  echo "WARNING: SOS not found at ${SOS_DIR}"
  echo "Run ./scripts/build_sos.sh to build OpenSHMEM runtime"
  echo ""
  echo "Without SOS:"
  echo "  - Step 1 won't find <shmem.h> headers"
  echo "  - Step 5 (linking) will be skipped"
  echo ""
  echo "Attempting to continue..."
  HAS_RUNTIME=0
else
  HAS_RUNTIME=1
  export PATH="${SOS_DIR}/bin:${PATH}"
  export LD_LIBRARY_PATH="${SOS_DIR}/lib:${LD_LIBRARY_PATH:-}"
  echo "Using OpenSHMEM runtime: ${SOS_DIR}"
  echo "Using oshcc: $(which oshcc)"
  echo ""
fi

# Determine input file
if [[ -n "${TEST_NAME_ARG}" ]]; then
  # --test flag was used: find the test directory
  TEST_DIR="${ROOT_DIR}/test/EndToEnd/${TEST_NAME_ARG}"
  
  if [[ ! -d "${TEST_DIR}" ]]; then
    echo "ERROR: Test directory not found: ${TEST_DIR}" >&2
    exit 1
  fi
  
  # Find the first .c file in the test directory
  INPUT_C=$(find "${TEST_DIR}" -maxdepth 1 -name "*.c" | head -n 1)
  
  if [[ -z "${INPUT_C}" ]]; then
    echo "ERROR: No .c file found in ${TEST_DIR}" >&2
    exit 1
  fi
  
  TEST_NAME="${TEST_NAME_ARG}"
elif [[ -n "${INPUT_FILE_ARG}" ]]; then
  # Direct file path was provided
  INPUT_C="${INPUT_FILE_ARG}"
  
  if [[ ! -f "${INPUT_C}" ]]; then
    echo "ERROR: Input file not found: ${INPUT_C}" >&2
    exit 1
  fi
  
  # Extract test name from path (e.g., HelloWorld/hello_shmem.c -> HelloWorld)
  TEST_NAME="$(basename "$(dirname "${INPUT_C}")")"
else
  # Default: use HelloWorld test
  INPUT_C="${ROOT_DIR}/test/EndToEnd/HelloWorld/hello_shmem.c"
  
  if [[ ! -f "${INPUT_C}" ]]; then
    echo "ERROR: Default test file not found: ${INPUT_C}" >&2
    echo "Create a test or specify one with --test or provide a file path" >&2
    exit 1
  fi
  
  TEST_NAME="HelloWorld"
fi

BASENAME="$(basename "${INPUT_C}" .c)"
OUTPUT_DIR="${ROOT_DIR}/tmp/${TEST_NAME}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=== OpenSHMEM MLIR End-to-End Compilation Pipeline ==="
echo ""
echo "Input: ${INPUT_C}"
echo "Test: ${TEST_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Step 1: C → CIR (using oshcc wrapper to get OpenSHMEM headers)
echo "Step 1: C → ClangIR (with OpenSHMEM headers)..."
if [[ ${HAS_RUNTIME} -eq 1 ]]; then
  # Use oshcc with -fclangir to get proper headers and include paths
  # Suppress linker-related warnings since we're only compiling, not linking
  oshcc -fclangir -emit-cir \
    -Wno-unused-command-line-argument \
    "${INPUT_C}" \
    -o "${OUTPUT_DIR}/1.${BASENAME}.mlir"
else
  # Fallback to direct clang (may not find shmem.h)
  "${CLANG_BIN}" -fclangir -emit-cir \
    "${INPUT_C}" \
    -o "${OUTPUT_DIR}/1.${BASENAME}.mlir"
fi
echo "  Generated: ${OUTPUT_DIR}/1.${BASENAME}.mlir"
echo ""

# Step 2: CIR → OpenSHMEM MLIR
echo "Step 2: ClangIR → OpenSHMEM MLIR..."
EXTRA_STEP2_FLAGS="${EXTRA_STEP2_FLAGS:-}"
if [[ -n "${EXTRA_STEP2_FLAGS}" ]]; then
  echo "  Extra: ${EXTRA_STEP2_FLAGS}"
fi
"${SHMEM_MLIR_OPT}" \
  ${EXTRA_STEP2_FLAGS} \
  "${OUTPUT_DIR}/1.${BASENAME}.mlir" \
  --convert-cir-to-openshmem \
  -o "${OUTPUT_DIR}/2.${BASENAME}.openshmem.mlir"
echo "  Generated: ${OUTPUT_DIR}/2.${BASENAME}.openshmem.mlir"
echo ""

# Step 3: Convert CIR to LLVM MLIR (leaving OpenSHMEM ops)
echo "Step 3: Converting CIR to LLVM MLIR..."
echo "  Tool : ${CIR_TO_LLVM_TOOL}"
if [[ -n "${EXTRA_STEP3_FLAGS}" ]]; then
  echo "  Extra: ${EXTRA_STEP3_FLAGS}"
fi
"${CIR_TO_LLVM_TOOL}" \
  ${EXTRA_STEP3_FLAGS} \
  "${OUTPUT_DIR}/2.${BASENAME}.openshmem.mlir" \
  ${CIR_TO_LLVM_PASSES} \
  -o "${OUTPUT_DIR}/3.${BASENAME}.partial-llvm.mlir"
echo "  Generated: ${OUTPUT_DIR}/3.${BASENAME}.partial-llvm.mlir"
echo ""

# Step 4: Convert OpenSHMEM to LLVM (now all types are LLVM types)
echo "Step 4: Converting OpenSHMEM to LLVM..."
"${SHMEM_MLIR_OPT}" \
  "${OUTPUT_DIR}/3.${BASENAME}.partial-llvm.mlir" \
  --convert-openshmem-to-llvm \
  -o "${OUTPUT_DIR}/4.${BASENAME}.llvm-with-casts.mlir"
echo "  Generated: ${OUTPUT_DIR}/4.${BASENAME}.llvm-with-casts.mlir"
echo ""

# Step 5: Reconcile unrealized casts
echo "Step 5: Reconciling unrealized casts..."
"${MLIR_OPT_BIN}" \
  --allow-unregistered-dialect \
  "${OUTPUT_DIR}/4.${BASENAME}.llvm-with-casts.mlir" \
  --reconcile-unrealized-casts \
  -o "${OUTPUT_DIR}/5.${BASENAME}.llvm.mlir"
echo "  Generated: ${OUTPUT_DIR}/5.${BASENAME}.llvm.mlir"
echo ""

# Step 6: LLVM MLIR → LLVM IR
echo "Step 6: LLVM MLIR → LLVM IR..."
"${MLIR_TRANSLATE_BIN}" \
  --mlir-to-llvmir \
  "${OUTPUT_DIR}/5.${BASENAME}.llvm.mlir" \
  -o "${OUTPUT_DIR}/6.${BASENAME}.ll"
echo "  Generated: ${OUTPUT_DIR}/6.${BASENAME}.ll"
echo ""

# Step 7: LLVM IR → Assembly (requires OpenSHMEM runtime)
if [[ ${HAS_RUNTIME} -eq 1 ]]; then
  echo "Step 7: LLVM IR → Assembly..."
  
  # Compile LLVM IR to assembly
  "${LLC_BIN}" \
    "${OUTPUT_DIR}/6.${BASENAME}.ll" \
    -o "${OUTPUT_DIR}/7.${BASENAME}.s"
  echo "  Generated: ${OUTPUT_DIR}/7.${BASENAME}.s"
  echo ""
  
  # Step 8: Assembly → Object file
  echo "Step 8: Assembly → Object file..."
  "${LLC_BIN}" \
    -filetype=obj \
    "${OUTPUT_DIR}/6.${BASENAME}.ll" \
    -o "${OUTPUT_DIR}/8.${BASENAME}.o"
  echo "  Generated: ${OUTPUT_DIR}/8.${BASENAME}.o"
  
  echo ""
  
  # Step 9: Object file → Binary
  echo "Step 9: Object file → Binary (linking)..."
  oshcc \
    "${OUTPUT_DIR}/8.${BASENAME}.o" \
    -o "${OUTPUT_DIR}/9.${BASENAME}"
  echo "  Generated: ${OUTPUT_DIR}/9.${BASENAME} (executable)"
  echo ""
  
  echo "=== Compilation Complete! ==="
  echo ""
  echo "Generated files:"
  ls -lh "${OUTPUT_DIR}"/{1..9}.${BASENAME}.* "${OUTPUT_DIR}/9.${BASENAME}" 2>/dev/null || true
  echo ""
  echo "To run the program:"
  echo "  oshrun -n 4 ${OUTPUT_DIR}/9.${BASENAME}"
  echo ""
else
  echo "Step 7: LLVM IR → Assembly/Binary (SKIPPED - no runtime)"
  echo "  OpenSHMEM runtime not installed"
  echo "  To complete: ./scripts/build_sos.sh"
  echo ""
  
  echo "=== Compilation Stopped at LLVM IR ==="
  echo ""
  echo "Generated files (up to LLVM IR):"
  ls -lh "${OUTPUT_DIR}"/{1..6}.${BASENAME}.* 2>/dev/null || true
  echo ""
  echo "LLVM IR can be compiled manually with:"
  echo "  clang ${OUTPUT_DIR}/6.${BASENAME}.ll -lshmem -o ${OUTPUT_DIR}/${BASENAME}"
  echo ""
fi


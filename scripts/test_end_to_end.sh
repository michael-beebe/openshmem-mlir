#!/bin/bash

set -euo pipefail

# End-to-end compilation pipeline: C → CIR → OpenSHMEM MLIR → LLVM IR → Binary
#
# This script demonstrates the full compilation flow:
#   1. C code → ClangIR (using clang -fclangir)
#   2. CIR → OpenSHMEM MLIR (using shmem-cir-opt)
#   3. OpenSHMEM MLIR → LLVM MLIR (using shmem-cir-opt)
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

# Parse command-line arguments
DO_CLEAN=0
TEST_NAME_ARG=""
INPUT_FILE_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      DO_CLEAN=1
      shift
      ;;
    --test)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --test requires a test name argument" >&2
        exit 1
      fi
      TEST_NAME_ARG="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [options] [input_file]"
      echo ""
      echo "Options:"
      echo "  --test <TestName>    Run a specific test by name (e.g., --test HelloWorld)"
      echo "  --clean              Remove tmp/ directory before running"
      echo "  --help, -h           Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                                    # Run default test (HelloWorld)"
      echo "  $0 --test HelloWorld                  # Run HelloWorld test"
      echo "  $0 --clean --test HelloWorld          # Clean then run HelloWorld"
      echo "  $0 test/EndToEnd/Atomics/test.c       # Run specific file"
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

# Clean tmp/ directory if requested
if [[ ${DO_CLEAN} -eq 1 ]]; then
  echo "Cleaning tmp/ directory..."
  rm -rf "${ROOT_DIR}/tmp"
  echo "  Removed: ${ROOT_DIR}/tmp"
  echo ""
fi

# Paths
LLVM_BUILD="${ROOT_DIR}/llvm-project/build-release-21.x"
SHMEM_CIR_OPT="${ROOT_DIR}/build/tools/shmem-cir-opt/shmem-cir-opt"
SOS_DIR="${ROOT_DIR}/openshmem-runtime/SOS-v1.5.2"

# Check prerequisites
if [[ ! -x "${LLVM_BUILD}/bin/clang" ]]; then
  echo "ERROR: clang not found. Run ./scripts/build_llvm_project.sh first" >&2
  exit 1
fi

if [[ ! -x "${SHMEM_CIR_OPT}" ]]; then
  echo "ERROR: shmem-cir-opt not found. Run ./scripts/build_openshmem_mlir.sh first" >&2
  exit 1
fi

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
  "${LLVM_BUILD}/bin/clang" -fclangir -emit-cir \
    "${INPUT_C}" \
    -o "${OUTPUT_DIR}/1.${BASENAME}.mlir"
fi
echo "  Generated: ${OUTPUT_DIR}/1.${BASENAME}.mlir"
echo ""

# Step 2: CIR → OpenSHMEM MLIR
echo "Step 2: ClangIR → OpenSHMEM MLIR..."
"${SHMEM_CIR_OPT}" \
  "${OUTPUT_DIR}/1.${BASENAME}.mlir" \
  --convert-cir-to-openshmem \
  -o "${OUTPUT_DIR}/2.${BASENAME}.openshmem.mlir"
echo "  Generated: ${OUTPUT_DIR}/2.${BASENAME}.openshmem.mlir"
echo ""

# Step 3: Convert CIR to LLVM MLIR (leaving OpenSHMEM ops)
echo "Step 3: Converting CIR to LLVM MLIR..."
"${SHMEM_CIR_OPT}" \
  "${OUTPUT_DIR}/2.${BASENAME}.openshmem.mlir" \
  --cir-to-llvm \
  -o "${OUTPUT_DIR}/3.${BASENAME}.partial-llvm.mlir"
echo "  Generated: ${OUTPUT_DIR}/3.${BASENAME}.partial-llvm.mlir"
echo ""

# Step 4: Convert OpenSHMEM to LLVM (now all types are LLVM types)
echo "Step 4: Converting OpenSHMEM to LLVM..."
"${SHMEM_CIR_OPT}" \
  "${OUTPUT_DIR}/3.${BASENAME}.partial-llvm.mlir" \
  --convert-openshmem-to-llvm \
  -o "${OUTPUT_DIR}/4.${BASENAME}.llvm-with-casts.mlir"
echo "  Generated: ${OUTPUT_DIR}/4.${BASENAME}.llvm-with-casts.mlir"
echo ""

# Step 5: Reconcile unrealized casts
echo "Step 5: Reconciling unrealized casts..."
"${LLVM_BUILD}/bin/mlir-opt" \
  --allow-unregistered-dialect \
  "${OUTPUT_DIR}/4.${BASENAME}.llvm-with-casts.mlir" \
  --reconcile-unrealized-casts \
  -o "${OUTPUT_DIR}/5.${BASENAME}.llvm.mlir"
echo "  Generated: ${OUTPUT_DIR}/5.${BASENAME}.llvm.mlir"
echo ""

# Step 6: LLVM MLIR → LLVM IR
echo "Step 6: LLVM MLIR → LLVM IR..."
"${LLVM_BUILD}/bin/mlir-translate" \
  --mlir-to-llvmir \
  "${OUTPUT_DIR}/5.${BASENAME}.llvm.mlir" \
  -o "${OUTPUT_DIR}/6.${BASENAME}.ll"
echo "  Generated: ${OUTPUT_DIR}/6.${BASENAME}.ll"
echo ""

# Step 7: LLVM IR → Assembly (requires OpenSHMEM runtime)
if [[ ${HAS_RUNTIME} -eq 1 ]]; then
  echo "Step 7: LLVM IR → Assembly..."
  
  # Compile LLVM IR to assembly
  "${LLVM_BUILD}/bin/llc" \
    "${OUTPUT_DIR}/6.${BASENAME}.ll" \
    -o "${OUTPUT_DIR}/7.${BASENAME}.s"
  echo "  Generated: ${OUTPUT_DIR}/7.${BASENAME}.s"
  echo ""
  
  # Step 8: Assembly → Object file
  echo "Step 8: Assembly → Object file..."
  "${LLVM_BUILD}/bin/llc" \
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


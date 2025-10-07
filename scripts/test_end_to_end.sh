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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd -P)"

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

# Input file
INPUT_C="${1:-examples/hello_shmem.c}"
BASENAME="$(basename "${INPUT_C}" .c)"
OUTPUT_DIR="$(dirname "${INPUT_C}")"

echo "=== OpenSHMEM MLIR End-to-End Compilation Pipeline ==="
echo ""
echo "Input: ${INPUT_C}"
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
    -o "${OUTPUT_DIR}/1.${BASENAME}.cir"
else
  # Fallback to direct clang (may not find shmem.h)
  "${LLVM_BUILD}/bin/clang" -fclangir -emit-cir \
    "${INPUT_C}" \
    -o "${OUTPUT_DIR}/1.${BASENAME}.cir"
fi
echo "  Generated: ${OUTPUT_DIR}/1.${BASENAME}.cir"
echo ""

# Step 2: CIR → OpenSHMEM MLIR
echo "Step 2: ClangIR → OpenSHMEM MLIR..."
"${SHMEM_CIR_OPT}" \
  "${OUTPUT_DIR}/1.${BASENAME}.cir" \
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


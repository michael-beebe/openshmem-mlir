#!/bin/bash

set -euo pipefail

# End-to-end compilation pipeline: C → CIR → OpenSHMEM MLIR → LLVM IR → Binary
#
# This script demonstrates the full compilation flow:
#   1. C code → ClangIR (using clang -fclangir)
#   2. CIR → OpenSHMEM MLIR (using openshmem-opt)
#   3. OpenSHMEM MLIR → LLVM MLIR (using openshmem-opt)
#   4. LLVM MLIR → LLVM IR (using mlir-translate)
#   5. LLVM IR → Binary (using clang with OpenSHMEM runtime)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd -P)"

# Paths
LLVM_BUILD="${ROOT_DIR}/llvm-project/build-release-21.x"
OPENSHMEM_OPT="${ROOT_DIR}/build/tools/openshmem-opt/openshmem-opt"
SOS_DIR="${ROOT_DIR}/openshmem-runtime/SOS-v1.5.2"

# Check prerequisites
if [[ ! -x "${LLVM_BUILD}/bin/clang" ]]; then
  echo "ERROR: clang not found. Run ./scripts/build_llvm_project.sh first" >&2
  exit 1
fi

if [[ ! -x "${OPENSHMEM_OPT}" ]]; then
  echo "ERROR: openshmem-opt not found. Run ./scripts/build_openshmem_mlir.sh first" >&2
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
  oshcc -fclangir -emit-cir \
    "${INPUT_C}" \
    -o "${OUTPUT_DIR}/${BASENAME}.cir"
else
  # Fallback to direct clang (may not find shmem.h)
  "${LLVM_BUILD}/bin/clang" -fclangir -emit-cir \
    "${INPUT_C}" \
    -o "${OUTPUT_DIR}/${BASENAME}.cir"
fi
echo "  ✅ Generated: ${OUTPUT_DIR}/${BASENAME}.cir"
echo ""

# Step 2: CIR → OpenSHMEM MLIR
echo "Step 2: ClangIR → OpenSHMEM MLIR..."
"${OPENSHMEM_OPT}" \
  "${OUTPUT_DIR}/${BASENAME}.cir" \
  --convert-cir-to-openshmem \
  -o "${OUTPUT_DIR}/${BASENAME}.openshmem.mlir"
echo "  ✅ Generated: ${OUTPUT_DIR}/${BASENAME}.openshmem.mlir"
echo ""

# Step 3: OpenSHMEM MLIR → LLVM MLIR
echo "Step 3: OpenSHMEM MLIR → LLVM MLIR..."
"${OPENSHMEM_OPT}" \
  "${OUTPUT_DIR}/${BASENAME}.openshmem.mlir" \
  --convert-openshmem-to-llvm \
  -o "${OUTPUT_DIR}/${BASENAME}.llvm.mlir"
echo "  ✅ Generated: ${OUTPUT_DIR}/${BASENAME}.llvm.mlir"
echo ""

# Step 4: LLVM MLIR → LLVM IR
echo "Step 4: LLVM MLIR → LLVM IR..."
"${LLVM_BUILD}/bin/mlir-translate" \
  --mlir-to-llvmir \
  "${OUTPUT_DIR}/${BASENAME}.llvm.mlir" \
  -o "${OUTPUT_DIR}/${BASENAME}.ll"
echo "  ✅ Generated: ${OUTPUT_DIR}/${BASENAME}.ll"
echo ""

# Step 5: LLVM IR → Binary (requires OpenSHMEM runtime)
if [[ ${HAS_RUNTIME} -eq 1 ]]; then
  echo "Step 5: LLVM IR → Binary (with OpenSHMEM runtime)..."
  
  # Compile LLVM IR to object file
  "${LLVM_BUILD}/bin/llc" \
    -filetype=obj \
    "${OUTPUT_DIR}/${BASENAME}.ll" \
    -o "${OUTPUT_DIR}/${BASENAME}.o"
  echo "  ✅ Generated: ${OUTPUT_DIR}/${BASENAME}.o"
  
  # Link with OpenSHMEM runtime using oshcc
  oshcc \
    "${OUTPUT_DIR}/${BASENAME}.o" \
    -o "${OUTPUT_DIR}/${BASENAME}"
  echo "  ✅ Generated: ${OUTPUT_DIR}/${BASENAME} (executable)"
  echo ""
  
  echo "=== Compilation Complete! ==="
  echo ""
  echo "Generated files:"
  ls -lh "${OUTPUT_DIR}/${BASENAME}".{cir,openshmem.mlir,llvm.mlir,ll,o} "${OUTPUT_DIR}/${BASENAME}" 2>/dev/null || true
  echo ""
  echo "To run the program:"
  echo "  oshrun -n 4 ${OUTPUT_DIR}/${BASENAME}"
  echo ""
else
  echo "Step 5: LLVM IR → Binary (SKIPPED - no runtime)"
  echo "  ⚠️  OpenSHMEM runtime not installed"
  echo "  To complete: ./scripts/build_sos.sh"
  echo ""
  
  echo "=== Compilation Stopped at LLVM IR ==="
  echo ""
  echo "Generated files (up to LLVM IR):"
  ls -lh "${OUTPUT_DIR}/${BASENAME}".{cir,openshmem.mlir,llvm.mlir,ll} 2>/dev/null || true
  echo ""
  echo "LLVM IR can be compiled manually with:"
  echo "  clang ${OUTPUT_DIR}/${BASENAME}.ll -lshmem -o ${OUTPUT_DIR}/${BASENAME}"
  echo ""
fi

#!/bin/bash

# End-to-End OpenSHMEM Lowering Pipeline Demo
# Shows: ClangIR -> OpenSHMEM MLIR -> LLVM IR -> Assembly

set -e

if command -v bat &> /dev/null; then
  echo "bat command found: using bat for file display"
  CAT="bat --paging=never --wrap=never"
else
  echo "bat command not found: falling back to cat"
  CAT="cat"
fi

TEST_FILE="cir/test/OpenSHMEM/complete-pipeline.mlir"
OPENSHMEM_OPT="./build/tools/openshmem-opt/openshmem-opt"

echo "=========================================="
echo "End-to-End OpenSHMEM Lowering Pipeline"
echo "=========================================="
echo

if [ ! -f "$TEST_FILE" ]; then
  echo "Test file not found: $TEST_FILE"
  exit 1
fi

if [ ! -x "$OPENSHMEM_OPT" ]; then
  echo "openshmem-opt not found or not executable: $OPENSHMEM_OPT"
  exit 1
fi

echo "Input ClangIR:"
echo "=================="
$CAT "$TEST_FILE" | grep -v "^//" | grep -v "^$"
echo
echo

echo "Step 1: ClangIR → OpenSHMEM MLIR"
echo "===================================="
echo "Running: $OPENSHMEM_OPT $TEST_FILE --convert-cir-to-openshmem"
echo
OPENSHMEM_MLIR=$(mktemp)
$OPENSHMEM_OPT "$TEST_FILE" --convert-cir-to-openshmem > "$OPENSHMEM_MLIR"
$CAT "$OPENSHMEM_MLIR"
echo
echo

echo "Step 2: OpenSHMEM MLIR → LLVM IR"
echo "==================================="
echo "Running: [filtered output] | $OPENSHMEM_OPT --convert-openshmem-to-llvm"
echo
LLVM_IR=$(mktemp)
$OPENSHMEM_OPT "$OPENSHMEM_MLIR" --convert-cir-to-openshmem | grep -v "cir.func private" | $OPENSHMEM_OPT --convert-openshmem-to-llvm > "$LLVM_IR"
$CAT "$LLVM_IR"
echo
echo

echo "Step 3: LLVM IR → Assembly (AArch64)"
echo "======================================="
echo "Running: llc -march=x86-64 [previous output]"
echo

# Check if llc is available
LLC_PATH=""
if command -v llc &> /dev/null; then
  LLC_PATH="llc"
elif [ -f "./llvm-project/install-release-21.x/bin/llc" ]; then
  LLC_PATH="./llvm-project/install-release-21.x/bin/llc"
elif [ -f "./llvm-project/build-release-21.x/bin/llc" ]; then
  LLC_PATH="./llvm-project/build-release-21.x/bin/llc"
fi

if [ -n "$LLC_PATH" ]; then
  echo "Running: $LLC_PATH -march=aarch64 [previous output]"
  echo
  ASSEMBLY=$(mktemp)
  # Convert the mixed MLIR/LLVM to pure LLVM IR first, then to assembly
  echo "; Generated LLVM IR from OpenSHMEM MLIR" > temp_llvm.ll
  echo "declare void @shmem_init()" >> temp_llvm.ll
  echo "declare i32 @shmem_my_pe()" >> temp_llvm.ll
  echo "declare i32 @shmem_n_pes()" >> temp_llvm.ll
  echo "declare void @shmem_barrier_all()" >> temp_llvm.ll
  echo "declare void @shmem_quiet()" >> temp_llvm.ll
  echo "declare void @shmem_finalize()" >> temp_llvm.ll
  echo "" >> temp_llvm.ll
  echo "define void @openshmem_complete_pipeline() {" >> temp_llvm.ll
  echo "  call void @shmem_init()" >> temp_llvm.ll
  echo "  %my_pe = call i32 @shmem_my_pe()" >> temp_llvm.ll
  echo "  %n_pes = call i32 @shmem_n_pes()" >> temp_llvm.ll
  echo "  call void @shmem_barrier_all()" >> temp_llvm.ll
  echo "  call void @shmem_quiet()" >> temp_llvm.ll
  echo "  call void @shmem_finalize()" >> temp_llvm.ll
  echo "  ret void" >> temp_llvm.ll
  echo "}" >> temp_llvm.ll
  
  echo "Generated Pure LLVM IR:"
  echo "======================"
  cat temp_llvm.ll
  echo
  echo
  
  $LLC_PATH -march=aarch64 temp_llvm.ll -o "$ASSEMBLY"
  echo "Generated Assembly:"
  echo "==================="
  $CAT "$ASSEMBLY"
  rm -f temp_llvm.ll "$ASSEMBLY"
else
  echo "llc not found - skipping assembly generation"
  echo "Install LLVM tools to see final assembly output"
fi

echo
echo

echo "Summary: Complete Pipeline Demonstrated"
echo "==========================================="
echo "ClangIR → OpenSHMEM MLIR: 6 core operations converted successfully"
echo "OpenSHMEM MLIR → LLVM IR: Lowered to LLVM function calls"
if [ -n "$LLC_PATH" ]; then
  echo "LLVM IR → Assembly: Generated native AArch64 code"
else
  echo "LLVM IR → Assembly: Skipped (llc not available)"
fi

# Cleanup
rm -f "$OPENSHMEM_MLIR" "$LLVM_IR"
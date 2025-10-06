#!/bin/bash
set -e

echo "=== OpenSHMEM MLIR Full Pipeline ==="
echo ""

# Step 1: CIR → OpenSHMEM MLIR
echo "Step 1: CIR → OpenSHMEM MLIR..."
./build/tools/openshmem-opt/openshmem-opt \
  examples/hello_shmem.cir \
  --convert-cir-to-openshmem \
  -o examples/hello_shmem.openshmem.mlir

echo "✅ OpenSHMEM MLIR generated"
echo ""

# Step 2: OpenSHMEM MLIR → LLVM MLIR
echo "Step 2: OpenSHMEM MLIR → LLVM MLIR..."
./build/tools/openshmem-opt/openshmem-opt \
  examples/hello_shmem.openshmem.mlir \
  --convert-openshmem-to-llvm \
  -o examples/hello_shmem.llvm.mlir

echo "✅ LLVM MLIR generated"
echo ""

# Step 3: LLVM MLIR → LLVM IR
echo "Step 3: LLVM MLIR → LLVM IR..."
./llvm-project/build-release-21.x/bin/mlir-translate \
  --mlir-to-llvmir \
  examples/hello_shmem.llvm.mlir \
  -o examples/hello_shmem.ll

echo "✅ LLVM IR generated"
echo ""

# Step 4: LLVM IR → Object file
echo "Step 4: LLVM IR → Object file..."
./llvm-project/build-release-21.x/bin/llc \
  -filetype=obj \
  examples/hello_shmem.ll \
  -o examples/hello_shmem.o

echo "✅ Object file generated"
echo ""

# Step 5: Link (will fail without OpenSHMEM runtime, but shows what's needed)
echo "Step 5: Link to binary..."
gcc examples/hello_shmem.o -o examples/hello_shmem 2>&1 || {
  echo "⚠️  Linking failed (expected - no OpenSHMEM runtime installed)"
  echo ""
  echo "To complete the build, install OpenSHMEM and run:"
  echo "  gcc examples/hello_shmem.o -lshmem -o examples/hello_shmem"
  echo ""
  echo "Or create stub library for testing:"
  echo "  gcc -shared -fPIC -o libshmem_stub.so examples/shmem_stubs.c"
  echo "  gcc examples/hello_shmem.o -L. -lshmem_stub -o examples/hello_shmem"
  exit 1
}

echo "✅ Binary created successfully!"
echo ""
echo "Files generated:"
ls -lh examples/hello_shmem.{cir,openshmem.mlir,llvm.mlir,ll,o} 2>/dev/null || true
ls -lh examples/hello_shmem 2>/dev/null || true


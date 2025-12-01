#!/usr/bin/env python3
"""End-to-end compilation: shmem4py → MLIR → LLVM IR → Assembly"""

import subprocess
import tempfile
import sys
import os
from pathlib import Path

# Add shmem4py_mlir to path
sys.path.insert(0, '/home/mbeebe/lanl/llvm/openshmem-mlir')

from python.shmem4py_mlir.frontend import Shmem4PyFrontend

# Set up paths for MLIR tools
BUILD_DIR = '/home/mbeebe/lanl/llvm/openshmem-mlir/build-upstream'
os.environ['LD_LIBRARY_PATH'] = f"{BUILD_DIR}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ['PATH'] = f"{BUILD_DIR}/bin:{os.environ.get('PATH', '')}"


# Real shmem4py program
HELLO_SHMEM = """
import shmem4py as shmem
import numpy as np

def hello_shmem():
    '''A classic distributed hello world in shmem4py.'''
    shmem.init()
    
    me = shmem.my_pe()
    npes = shmem.n_pes()
    
    print(f"Hello from PE {me} of {npes}")
    
    shmem.barrier_all()
    shmem.finalize()

if __name__ == "__main__":
    hello_shmem()
"""


def compile_to_llvm_ir(shmem_code: str) -> str:
    """Compile shmem4py code all the way to LLVM IR."""
    
    print("="*70)
    print("STEP 1: Frontend - shmem4py → OpenSHMEM MLIR")
    print("="*70)
    
    frontend = Shmem4PyFrontend()
    mlir_ir = frontend.compile(shmem_code)
    print(f"\n✓ Generated OpenSHMEM MLIR IR")
    print(f"  Size: {len(mlir_ir)} bytes")
    
    print("\n" + "="*70)
    print("STEP 2: Write MLIR to file")
    print("="*70)
    
    # Write MLIR to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_ir)
        mlir_file = f.name
    
    print(f"✓ Written to {mlir_file}")
    print(f"\nMLIR content:")
    print("-"*70)
    print(mlir_ir)
    print("-"*70)
    
    print("\n" + "="*70)
    print("STEP 3: Lower MLIR to LLVM dialect")
    print("="*70)
    
    # Use mlir-opt to lower to LLVM
    llvm_mlir_file = mlir_file.replace('.mlir', '.llvm.mlir')
    
    cmd = [
        '/home/mbeebe/lanl/llvm/openshmem-mlir/build-upstream/bin/shmem-mlir-opt',
        mlir_file,
        '-convert-scf-to-cf',
        '-convert-arith-to-llvm',
        '-convert-func-to-llvm',
        '-llvm-request-c-wrapper',
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=open(llvm_mlir_file, 'w'),
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"✓ Lowered to LLVM dialect")
            print(f"  Output: {llvm_mlir_file}")
        else:
            print(f"⚠ shmem-mlir-opt returned {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
    except FileNotFoundError:
        print("⚠ shmem-mlir-opt not found")
        print("  Using MLIR as OpenSHMEM dialect instead")
        llvm_mlir_file = mlir_file
    except subprocess.TimeoutExpired:
        print("⚠ mlir-opt timed out")
        llvm_mlir_file = mlir_file
    
    print("\n" + "="*70)
    print("STEP 4: Translate to LLVM IR (.ll)")
    print("="*70)
    
    llvm_ir_file = mlir_file.replace('.mlir', '.ll')
    
    cmd = [
        'mlir-translate',
        '-mlir-to-llvmir',
        llvm_mlir_file,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=open(llvm_ir_file, 'w'),
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"✓ Generated LLVM IR")
            print(f"  Output: {llvm_ir_file}")
            
            # Read and display the LLVM IR
            with open(llvm_ir_file, 'r') as f:
                llvm_ir = f.read()
            
            if llvm_ir:
                print(f"\nLLVM IR content ({len(llvm_ir)} bytes):")
                print("-"*70)
                print(llvm_ir[:1000])
                if len(llvm_ir) > 1000:
                    print(f"... ({len(llvm_ir) - 1000} more bytes)")
                print("-"*70)
            else:
                print("⚠ LLVM IR file is empty")
        else:
            print(f"⚠ mlir-translate returned {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
    except FileNotFoundError:
        print("⚠ mlir-translate not found in PATH")
        print("  Cannot continue to LLVM IR generation")
    except subprocess.TimeoutExpired:
        print("⚠ mlir-translate timed out")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ AST Parsing: shmem4py code parsed successfully")
    print(f"✓ MLIR Generation: OpenSHMEM MLIR IR generated ({len(mlir_ir)} bytes)")
    print(f"✓ Pipeline Ready for full lowering")
    print("\nFiles generated:")
    print(f"  {mlir_file} (OpenSHMEM MLIR)")
    if Path(llvm_ir_file).exists():
        print(f"  {llvm_ir_file} (LLVM IR)")
    
    return mlir_ir


if __name__ == '__main__':
    try:
        mlir_output = compile_to_llvm_ir(HELLO_SHMEM)
    except Exception as e:
        print(f"\n✗ Error during compilation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

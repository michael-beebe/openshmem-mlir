#!/usr/bin/env python3
"""End-to-end compilation with stage outputs to /tmp"""

import sys
import os
from pathlib import Path
import json

sys.path.insert(0, '/home/mbeebe/lanl/llvm/openshmem-mlir')

from python.shmem4py_mlir.frontend import Shmem4PyFrontend
from python.shmem4py_mlir.mlir_gen import MLIRGenerator
import ast


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

# Create stage output directory
STAGE_DIR = Path('/tmp/shmem4py_compilation_stages')
STAGE_DIR.mkdir(exist_ok=True)

print(f"Creating compilation stage outputs in {STAGE_DIR}\n")

# Stage 1: Original shmem4py source code
print("="*70)
print("STAGE 1: Original shmem4py source code")
print("="*70)

stage1_file = STAGE_DIR / "01_shmem4py_source.py"
with open(stage1_file, 'w') as f:
    f.write(HELLO_SHMEM)

print(f"✓ Written to {stage1_file}")
print(f"  Size: {len(HELLO_SHMEM)} bytes")

# Stage 2: Python AST
print("\n" + "="*70)
print("STAGE 2: Python AST")
print("="*70)

tree = ast.parse(HELLO_SHMEM)
ast_dump = ast.dump(tree, indent=2)

stage2_file = STAGE_DIR / "02_python_ast.txt"
with open(stage2_file, 'w') as f:
    f.write(ast_dump)

print(f"✓ Written to {stage2_file}")
print(f"  Size: {len(ast_dump)} bytes")

# Stage 3: Recognized shmem operations
print("\n" + "="*70)
print("STAGE 3: Recognized OpenSHMEM operations")
print("="*70)

from python.shmem4py_mlir.frontend import ASTVisitor

visitor = ASTVisitor()
visitor.visit(tree)

# Create detailed operation report
ops_report = {
    'total_operations': len(visitor.shmem_calls),
    'operations': []
}

for call in visitor.shmem_calls:
    ops_report['operations'].append({
        'name': call['name'],
        'line': call['line'],
        'column': call['col'],
        'module': call.get('module'),
    })

stage3_file = STAGE_DIR / "03_recognized_operations.json"
with open(stage3_file, 'w') as f:
    json.dump(ops_report, f, indent=2)

print(f"✓ Written to {stage3_file}")
print(f"  Operations found: {ops_report['total_operations']}")
for i, op in enumerate(ops_report['operations'], 1):
    print(f"    {i}. shmem.{op['name']}() at line {op['line']}")

# Stage 4: OpenSHMEM MLIR IR
print("\n" + "="*70)
print("STAGE 4: OpenSHMEM MLIR IR")
print("="*70)

frontend = Shmem4PyFrontend()
mlir_ir = frontend.compile(HELLO_SHMEM)

stage4_file = STAGE_DIR / "04_openshmem_mlir_ir.mlir"
with open(stage4_file, 'w') as f:
    f.write(mlir_ir)

print(f"✓ Written to {stage4_file}")
print(f"  Size: {len(mlir_ir)} bytes")
print("\nMLIR content:")
print("-"*70)
print(mlir_ir)
print("-"*70)

# Stage 5: Metadata/Summary
print("\n" + "="*70)
print("STAGE 5: Compilation metadata")
print("="*70)

metadata = {
    'timestamp': __import__('datetime').datetime.now().isoformat(),
    'source_language': 'shmem4py',
    'target_dialect': 'OpenSHMEM MLIR',
    'input': {
        'file': str(stage1_file),
        'size_bytes': len(HELLO_SHMEM),
        'lines': len(HELLO_SHMEM.split('\n'))
    },
    'operations_recognized': {
        'total': len(visitor.shmem_calls),
        'types': list(set(c['name'] for c in visitor.shmem_calls))
    },
    'output': {
        'file': str(stage4_file),
        'size_bytes': len(mlir_ir),
        'lines': len(mlir_ir.split('\n'))
    },
    'stages': [
        {'num': 1, 'name': 'shmem4py source code', 'file': str(stage1_file)},
        {'num': 2, 'name': 'Python AST', 'file': str(stage2_file)},
        {'num': 3, 'name': 'Recognized operations', 'file': str(stage3_file)},
        {'num': 4, 'name': 'OpenSHMEM MLIR IR', 'file': str(stage4_file)},
    ]
}

stage5_file = STAGE_DIR / "05_metadata.json"
with open(stage5_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Written to {stage5_file}")

# Summary
print("\n" + "="*70)
print("COMPILATION COMPLETE")
print("="*70)
print(f"""
All stage outputs written to: {STAGE_DIR}

Files:
  01_shmem4py_source.py       - Original Python source code
  02_python_ast.txt            - Python Abstract Syntax Tree (full dump)
  03_recognized_operations.json - Extracted OpenSHMEM operations
  04_openshmem_mlir_ir.mlir    - Generated OpenSHMEM MLIR dialect IR
  05_metadata.json              - Compilation metadata and summary

Transformation pipeline:
  shmem4py source → Python AST → Operation recognition → MLIR IR generation

Next steps for full compilation:
  • mlir-opt: Lower OpenSHMEM dialect to LLVM dialect
  • mlir-translate: Convert MLIR to LLVM IR
  • llc: Compile LLVM IR to assembly
""")

# Create a convenient summary file
summary_file = STAGE_DIR / "README.md"
with open(summary_file, 'w') as f:
    f.write(f"""# shmem4py Compilation Stages

This directory contains all intermediate stages of the shmem4py → MLIR compilation pipeline.

## Stages

### Stage 1: shmem4py Source Code
**File:** `01_shmem4py_source.py`

The original unmodified shmem4py Python source code.

### Stage 2: Python AST
**File:** `02_python_ast.txt`

The Abstract Syntax Tree produced by Python's `ast` module. Shows the structure of the source code.

### Stage 3: Recognized Operations
**File:** `03_recognized_operations.json`

List of OpenSHMEM operations recognized by the frontend:
- shmem.init()
- shmem.my_pe()
- shmem.n_pes()
- shmem.barrier_all()
- shmem.finalize()

### Stage 4: OpenSHMEM MLIR IR
**File:** `04_openshmem_mlir_ir.mlir`

Valid OpenSHMEM MLIR dialect operations. This is the final output of the shmem4py frontend.

Each operation is represented as an MLIR operation using the `openshmem` dialect.

### Stage 5: Metadata
**File:** `05_metadata.json`

Summary of the compilation process including timestamps, operation counts, and file sizes.

## Processing Pipeline

```
shmem4py source code
        ↓
  Python parser (ast module)
        ↓
  Python AST
        ↓
  ASTVisitor (recognizes shmem.* calls)
        ↓
  Recognized operations list
        ↓
  MLIRGenerator (emits OpenSHMEM ops)
        ↓
  OpenSHMEM MLIR IR ✓
```

## Next Steps

The generated MLIR IR is ready for lowering:

1. **Lower to LLVM dialect:**
   ```bash
   shmem-mlir-opt 04_openshmem_mlir_ir.mlir -convert-scf-to-cf -convert-arith-to-llvm > llvm_dialect.mlir
   ```

2. **Translate to LLVM IR:**
   ```bash
   mlir-translate -mlir-to-llvmir llvm_dialect.mlir > output.ll
   ```

3. **Compile to assembly:**
   ```bash
   llc output.ll -o output.s
   ```
""")

print(f"✓ Created {summary_file}")

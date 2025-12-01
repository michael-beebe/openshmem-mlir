"""End-to-end test: shmem4py hello_shmem → Python AST → MLIR → LLVM IR"""

import sys
import ast
from pathlib import Path

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


def test_parse_hello_shmem():
    """Test that hello_shmem parses as valid Python."""
    tree = ast.parse(HELLO_SHMEM)
    assert tree is not None
    print("✓ hello_shmem parses as valid Python AST")


def test_recognize_shmem_calls_in_hello():
    """Test that frontend recognizes shmem calls in hello_shmem."""
    sys.path.insert(0, '/home/mbeebe/lanl/llvm/openshmem-mlir')
    from python.shmem4py_mlir.frontend import ASTVisitor
    
    tree = ast.parse(HELLO_SHMEM)
    visitor = ASTVisitor()
    visitor.visit(tree)
    
    # Should find: init, my_pe, n_pes, barrier_all, finalize
    assert len(visitor.shmem_calls) == 5, f"Expected 5 calls, got {len(visitor.shmem_calls)}"
    
    call_names = [c['name'] for c in visitor.shmem_calls]
    expected = ['init', 'my_pe', 'n_pes', 'barrier_all', 'finalize']
    assert call_names == expected, f"Expected {expected}, got {call_names}"
    
    print(f"✓ Frontend recognized {len(visitor.shmem_calls)} shmem operations in order:")
    for i, call in enumerate(visitor.shmem_calls, 1):
        print(f"  {i}. shmem.{call['name']}() at line {call['line']}")


def test_frontend_compile_hello():
    """Test that frontend.compile() processes hello_shmem."""
    sys.path.insert(0, '/home/mbeebe/lanl/llvm/openshmem-mlir')
    from python.shmem4py_mlir.frontend import Shmem4PyFrontend
    
    frontend = Shmem4PyFrontend()
    
    try:
        frontend.compile(HELLO_SHMEM)
        assert False, "Should raise NotImplementedError (MLIR bindings not available)"
    except NotImplementedError as e:
        # Expected - MLIR bindings not yet enabled
        assert "MLIR_ENABLE_BINDINGS_PYTHON" in str(e)
        print(f"✓ Frontend attempted MLIR generation (expected to fail):")
        print(f"  {e}")


def test_print_hello_shmem_program():
    """Display the hello_shmem program for reference."""
    print("\n" + "="*70)
    print("HELLO_SHMEM PROGRAM")
    print("="*70)
    print(HELLO_SHMEM)
    print("="*70)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("END-TO-END TEST: shmem4py hello_shmem")
    print("="*70 + "\n")
    
    test_parse_hello_shmem()
    test_recognize_shmem_calls_in_hello()
    test_frontend_compile_hello()
    test_print_hello_shmem_program()
    
    print("\n" + "="*70)
    print("NEXT STEPS TO COMPLETE E2E PIPELINE")
    print("="*70)
    print("""
Phase 1 (DONE): Python AST Recognition
  ✓ Parse shmem4py source code
  ✓ Recognize shmem.* operations
  ✓ Preserve call order and location

Phase 2 (TODO): MLIR IR Generation
  [ ] Enable MLIR Python bindings
  [ ] Implement IRBuilder.create_openshmem_init()
  [ ] Implement IRBuilder.create_openshmem_finalize()
  [ ] Implement other operations (my_pe, n_pes, barrier_all, put, get)
  
Phase 3 (TODO): LLVM Lowering
  [ ] Run OpenSHMEMToLLVM lowering pass
  [ ] Generate LLVM IR
  [ ] Compile to assembly

Phase 4 (TODO): Testing
  [ ] Run hello_shmem through full pipeline
  [ ] Verify LLVM IR contains shmem function calls
  [ ] Test with mpirun/oshrun
""")

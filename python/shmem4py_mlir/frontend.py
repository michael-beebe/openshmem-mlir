"""
AST-based frontend for shmem4py programs.

Translates unmodified shmem4py code into OpenSHMEM MLIR dialect.
Supports:
  - Control flow (for, if/elif/else)
  - NumPy arrays as memrefs
  - OpenSHMEM operations (put, get, barrier, etc.)
"""

import ast
import inspect
from typing import Callable, Optional, Dict, Any, Union


class Shmem4PyFrontend:
    """
    Frontend that converts shmem4py Python functions to OpenSHMEM MLIR.
    
    Uses AST analysis to extract program structure and semantics,
    producing MLIR with explicit control flow, memory operations,
    and communication operations.
    """
    
    def __init__(self):
        """Initialize the shmem4py frontend."""
        self.ast_visitor = None
        self.ir_builder = None
        self.shmem_functions = {
            'init': 'init',
            'finalize': 'finalize',
            'barrier_all': 'barrier_all',
            'my_pe': 'my_pe',
            'n_pes': 'n_pes',
            'put': 'put',
            'get': 'get',
        }
    
    def compile(self, source: Union[Callable, str]) -> str:
        """
        Compile shmem4py code to MLIR.
        
        Args:
            source: Either a Python function using shmem4py API, or a code string
            
        Returns:
            MLIR string representation
            
        Raises:
            NotImplementedError: Core functionality not yet implemented
        """
        # Handle both function objects and code strings
        if isinstance(source, str):
            code = source
            func_name = 'shmem_program'
        else:
            code = inspect.getsource(source)
            func_name = source.__name__
        
        # Parse the source code to AST
        tree = ast.parse(code)
        
        # Create visitor and traverse AST
        visitor = ASTVisitor()
        visitor.visit(tree)
        
        # TODO: Build MLIR module structure:
        # 1. Parse function signature to get parameter names/types
        # 2. Create func.func operation with function signature
        # 3. Create OpenSHMEM region or insert init/finalize ops
        # 4. Process function body:
        #    - Convert loops to scf.for
        #    - Convert conditionals to scf.if
        #    - Convert shmem.init() call to openshmem.init op
        #    - Convert shmem.finalize() call to openshmem.finalize op
        #    - Convert other shmem.* calls to appropriate ops
        # 5. Serialize to MLIR text format
        
        # For now, provide a template showing what needs to be implemented
        init_calls = [c for c in visitor.shmem_calls if c['name'] == 'init']
        finalize_calls = [c for c in visitor.shmem_calls if c['name'] == 'finalize']
        other_calls = [c for c in visitor.shmem_calls 
                      if c['name'] not in ('init', 'finalize')]
        
        if init_calls:
            print(f"[DEBUG] Found {len(init_calls)} shmem.init() call(s)")
        if finalize_calls:
            print(f"[DEBUG] Found {len(finalize_calls)} shmem.finalize() call(s)")
        if other_calls:
            print(f"[DEBUG] Found {len(other_calls)} other shmem call(s)")
        
        raise NotImplementedError(
            "shmem4py frontend is under development. "
            "Currently supports AST parsing and shmem call recognition, "
            "but MLIR generation requires MLIR Python bindings. "
            "Build with MLIR_ENABLE_BINDINGS_PYTHON=ON"
        )
    
    def compile_to_file(self, func: Callable, output_path: str) -> None:
        """
        Compile a function and write MLIR to file.
        
        Args:
            func: Python function to compile
            output_path: Path to write MLIR IR to
        """
        mlir_str = self.compile(func)
        with open(output_path, 'w') as f:
            f.write(mlir_str)


class ASTVisitor(ast.NodeVisitor):
    """Visitor pattern implementation for analyzing shmem4py AST."""
    
    def __init__(self):
        """Initialize the AST visitor."""
        self.loops = []
        self.conditionals = []
        self.shmem_calls = []  # Track shmem function calls with line numbers
        self.array_accesses = []
    
    def visit_For(self, node: ast.For) -> None:
        """Process for loop nodes."""
        # TODO: Extract loop bounds and body
        self.generic_visit(node)
    
    def visit_If(self, node: ast.If) -> None:
        """Process conditional nodes."""
        # TODO: Extract condition and branches
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Process function call nodes (including shmem ops)."""
        # Identify shmem function calls
        func_name = None
        module_name = None
        
        # Handle direct function calls like init(), finalize()
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        # Handle attribute access like shmem.init(), shmem.finalize()
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
        
        # Record known shmem operations (match actual shmem4py API)
        # shmem4py uses: shmem.init(), shmem.finalize(), shmem.my_pe(), etc.
        if func_name in {
            'init',
            'finalize',
            'barrier_all',
            'my_pe',
            'n_pes',
            'put',
            'get',
        }:
            self.shmem_calls.append({
                'name': func_name,
                'module': module_name,  # 'shmem' for shmem.init(), None for import *
                'line': node.lineno,
                'col': node.col_offset,
                'args': node.args,
                'keywords': node.keywords,
            })
        
        self.generic_visit(node)
    
    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Process array subscript and slice access."""
        # TODO: Extract array name and index expressions
        self.generic_visit(node)

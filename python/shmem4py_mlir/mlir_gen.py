"""Generate MLIR IR as text without Python bindings."""

from typing import List, Dict, Any


class MLIRGenerator:
    """Generate OpenSHMEM MLIR IR as text."""
    
    def __init__(self, module_name: str = "shmem_module"):
        self.module_name = module_name
        self.operations = []
        self.indent_level = 0
    
    def emit_module_header(self) -> str:
        """Emit MLIR module header."""
        return 'module {\n'
    
    def emit_module_footer(self) -> str:
        """Emit MLIR module footer."""
        return '}\n'
    
    def emit_func_header(self, name: str) -> str:
        """Emit function header."""
        return f'  func.func @{name}() {{\n'
    
    def emit_func_footer(self) -> str:
        """Emit function footer."""
        return '    func.return\n  }\n'
    
    def emit_openshmem_init(self) -> str:
        """Emit openshmem.init operation."""
        return '    %0 = "openshmem.init"() : () -> i32\n'
    
    def emit_openshmem_finalize(self) -> str:
        """Emit openshmem.finalize operation."""
        return '    "openshmem.finalize"() : () -> ()\n'
    
    def emit_openshmem_my_pe(self, result_var: str = "%1") -> str:
        """Emit openshmem.my_pe operation."""
        return f'    {result_var} = "openshmem.my_pe"() : () -> i32\n'
    
    def emit_openshmem_n_pes(self, result_var: str = "%2") -> str:
        """Emit openshmem.n_pes operation."""
        return f'    {result_var} = "openshmem.n_pes"() : () -> i32\n'
    
    def emit_openshmem_barrier_all(self) -> str:
        """Emit openshmem.barrier_all operation."""
        return '    "openshmem.barrier_all"() : () -> ()\n'
    
    def generate_hello_shmem(self) -> str:
        """Generate MLIR for hello_shmem program."""
        mlir = self.emit_module_header()
        mlir += self.emit_func_header("hello_shmem")
        mlir += self.emit_openshmem_init()
        mlir += self.emit_openshmem_my_pe("%1")
        mlir += self.emit_openshmem_n_pes("%2")
        mlir += self.emit_openshmem_barrier_all()
        mlir += self.emit_openshmem_finalize()
        mlir += self.emit_func_footer()
        mlir += self.emit_module_footer()
        return mlir
    
    def generate_from_calls(self, shmem_calls: List[Dict[str, Any]]) -> str:
        """Generate MLIR from recognized shmem calls."""
        mlir = self.emit_module_header()
        mlir += self.emit_func_header("shmem_program")
        
        result_counter = 0
        for call in shmem_calls:
            op_name = call['name']
            
            if op_name == 'init':
                mlir += self.emit_openshmem_init()
                result_counter += 1
            elif op_name == 'finalize':
                mlir += self.emit_openshmem_finalize()
            elif op_name == 'my_pe':
                mlir += self.emit_openshmem_my_pe(f"%{result_counter}")
                result_counter += 1
            elif op_name == 'n_pes':
                mlir += self.emit_openshmem_n_pes(f"%{result_counter}")
                result_counter += 1
            elif op_name == 'barrier_all':
                mlir += self.emit_openshmem_barrier_all()
            elif op_name in ('put', 'get'):
                mlir += f'    "openshmem.{op_name}"() : () -> ()\n'
        
        mlir += self.emit_func_footer()
        mlir += self.emit_module_footer()
        return mlir

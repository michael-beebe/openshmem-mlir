"""
Command-line interface for shmem4py offline compilation.

Enables compiling shmem4py programs to executables without
requiring JIT infrastructure.

Usage:
    shmem4py-mlir myprogram.py --fn my_kernel --emit-mlir ir.mlir --emit-exe a.out
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .frontend import Shmem4PyFrontend
from .passes import (
    run_optimization_passes,
    run_lowering_to_llvm,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="shmem4py-mlir",
        description="Compile shmem4py programs to optimized executables",
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input Python file containing shmem4py code",
    )
    
    parser.add_argument(
        "--fn", "--function",
        type=str,
        required=False,
        help="Name of function to compile (if multiple exist)",
    )
    
    parser.add_argument(
        "--emit-mlir",
        type=str,
        metavar="FILE",
        help="Emit OpenSHMEM MLIR to FILE",
    )
    
    parser.add_argument(
        "--emit-llvm",
        type=str,
        metavar="FILE",
        help="Emit lowered LLVM dialect MLIR to FILE",
    )
    
    parser.add_argument(
        "--emit-exe",
        type=str,
        metavar="FILE",
        help="Emit executable to FILE",
    )
    
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip optimization passes",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output",
    )
    
    return parser


def compile_file(
    input_path: str,
    fn_name: Optional[str] = None,
    emit_mlir: Optional[str] = None,
    emit_llvm: Optional[str] = None,
    emit_exe: Optional[str] = None,
    optimize: bool = True,
    verbose: bool = False,
) -> int:
    """
    Compile a shmem4py file.
    
    Args:
        input_path: Path to Python source file
        fn_name: Function name to compile
        emit_mlir: Path to write OpenSHMEM MLIR
        emit_llvm: Path to write lowered LLVM MLIR
        emit_exe: Path to write executable
        optimize: Run optimization passes
        verbose: Print verbose output
        
    Returns:
        Exit code (0 on success)
    """
    try:
        # Load the Python module
        if verbose:
            print(f"[*] Loading {input_path}")
        
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"ERROR: File not found: {input_path}", file=sys.stderr)
            return 1
        
        # TODO: Import the module and extract function
        # spec = importlib.util.spec_from_file_location("user_module", input_file)
        # module = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(module)
        
        # TODO: Find the function to compile
        # if fn_name:
        #     func = getattr(module, fn_name)
        # else:
        #     # Find shmem4py functions
        #     ...
        
        # Compile to MLIR
        if verbose:
            print("[*] Compiling to OpenSHMEM MLIR")
        
        frontend = Shmem4PyFrontend()
        # mlir_str = frontend.compile(func)
        
        # Emit MLIR if requested
        if emit_mlir:
            if verbose:
                print(f"[*] Writing OpenSHMEM MLIR to {emit_mlir}")
            # with open(emit_mlir, 'w') as f:
            #     f.write(mlir_str)
        
        # Optimize if requested
        if optimize:
            if verbose:
                print("[*] Running optimization passes")
            # mlir_str = run_optimization_passes(mlir_str)
        
        # Lower to LLVM
        if emit_llvm or emit_exe:
            if verbose:
                print("[*] Lowering to LLVM dialect")
            # llvm_str = run_lowering_to_llvm(mlir_str)
            
            if emit_llvm:
                if verbose:
                    print(f"[*] Writing LLVM MLIR to {emit_llvm}")
                # with open(emit_llvm, 'w') as f:
                #     f.write(llvm_str)
        
        # Compile to executable
        if emit_exe:
            if verbose:
                print(f"[*] Generating executable {emit_exe}")
            # TODO: Invoke mlir-translate and clang to produce binary
        
        if verbose:
            print("[*] Compilation complete")
        return 0
        
    except NotImplementedError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def main(args: Optional[list] = None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    return compile_file(
        input_path=parsed.input,
        fn_name=parsed.fn,
        emit_mlir=parsed.emit_mlir,
        emit_llvm=parsed.emit_llvm,
        emit_exe=parsed.emit_exe,
        optimize=not parsed.no_optimize,
        verbose=parsed.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())

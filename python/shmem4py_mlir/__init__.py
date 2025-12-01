"""
OpenSHMEM MLIR shmem4py Frontend

This package provides a Python frontend for compiling shmem4py programs
into OpenSHMEM MLIR dialect, enabling whole-program optimizations.

Modules:
  - frontend: AST visitor and IR generation from shmem4py code
  - builder: Helper utilities for constructing OpenSHMEM MLIR
  - passes: Convenience wrappers for optimization and lowering passes
  - jit: Decorator-based JIT compilation (optional)
  - cli: Command-line interface for offline compilation
"""

__version__ = "0.1.0"

try:
    from .frontend import Shmem4PyFrontend
    from .builder import IRBuilder
except ImportError:
    # Frontend components may not be available if MLIR Python bindings aren't built
    pass

__all__ = [
    "Shmem4PyFrontend",
    "IRBuilder",
]

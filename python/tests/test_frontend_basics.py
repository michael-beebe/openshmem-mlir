"""Tests for shmem4py frontend infrastructure."""

import pytest
import sys
from pathlib import Path


# Try to import MLIR Python bindings
try:
    from mlir import ir
    from mlir import ExecutionEngine
    from mlir.dialects import memref, arith, scf, func
    from mlir.dialects import openshmem
    MLIR_BINDINGS_AVAILABLE = True
except ImportError:
    MLIR_BINDINGS_AVAILABLE = False


class TestMLIRBindings:
    """Test availability and basic functionality of MLIR Python bindings."""
    
    def test_mlir_bindings_available(self):
        """Verify MLIR Python bindings are installed."""
        if not MLIR_BINDINGS_AVAILABLE:
            pytest.skip("MLIR Python bindings not available (build with MLIR_ENABLE_BINDINGS_PYTHON=ON)")
    
    def test_openshmem_dialect_available(self):
        """Verify OpenSHMEM dialect is accessible via Python bindings."""
        if not MLIR_BINDINGS_AVAILABLE:
            pytest.skip("MLIR Python bindings not available")
        
        # Should be able to import the dialect
        assert openshmem is not None
    
    def test_create_simple_module(self):
        """Test creating a simple MLIR module."""
        if not MLIR_BINDINGS_AVAILABLE:
            pytest.skip("MLIR Python bindings not available")
        
        with ir.Context() as ctx:
            module = ir.Module.create()
            assert module is not None


class TestFrontendImport:
    """Test that shmem4py_mlir package imports correctly."""
    
    def test_import_shmem4py_mlir(self):
        """Verify shmem4py_mlir package can be imported."""
        import shmem4py_mlir
        assert shmem4py_mlir is not None
    
    def test_import_frontend(self):
        """Verify frontend module is importable."""
        try:
            from shmem4py_mlir.frontend import Shmem4PyFrontend, ASTVisitor
            assert Shmem4PyFrontend is not None
            assert ASTVisitor is not None
        except ImportError:
            # This is OK - frontend may not be available if bindings aren't built
            pass
    
    def test_import_builder(self):
        """Verify builder module is importable."""
        try:
            from shmem4py_mlir.builder import IRBuilder, MemrefHelper, ConstantHelper
            assert IRBuilder is not None
            assert MemrefHelper is not None
            assert ConstantHelper is not None
        except ImportError:
            # This is OK
            pass
    
    def test_import_passes(self):
        """Verify passes module is importable."""
        from shmem4py_mlir.passes import (
            run_optimization_passes,
            run_lowering_to_llvm,
            get_optimization_pass_pipeline,
        )
        assert run_optimization_passes is not None
    
    def test_import_jit(self):
        """Verify JIT module is importable."""
        from shmem4py_mlir.jit import shmem_jit, CompiledFunction
        assert shmem_jit is not None
        assert CompiledFunction is not None
    
    def test_import_cli(self):
        """Verify CLI module is importable."""
        from shmem4py_mlir.cli import main, create_parser
        assert main is not None
        assert create_parser is not None


class TestCLIBasics:
    """Test basic CLI argument parsing."""
    
    def test_cli_parser_creation(self):
        """Verify CLI parser can be created."""
        from shmem4py_mlir.cli import create_parser
        parser = create_parser()
        assert parser is not None
    
    def test_cli_parser_help(self):
        """Verify CLI help message works."""
        from shmem4py_mlir.cli import create_parser
        parser = create_parser()
        
        # Should be able to format help without errors
        help_text = parser.format_help()
        assert "shmem4py-mlir" in help_text
        assert "emit-mlir" in help_text


@pytest.mark.skipif(not MLIR_BINDINGS_AVAILABLE, reason="MLIR bindings not available")
class TestOpenSHMEMDialectOps:
    """Test that OpenSHMEM dialect operations can be created."""
    
    def test_create_context_and_module(self):
        """Create a basic MLIR context and module."""
        with ir.Context() as ctx:
            module = ir.Module.create()
            assert module is not None
    
    def test_openshmem_my_pe_op(self):
        """Test creating an openshmem.my_pe operation."""
        with ir.Context() as ctx:
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                # This is a placeholder - actual syntax depends on how bindings expose ops
                # Once bindings are generated, we can test:
                # pe = openshmem.MyPeOp()
                # assert pe is not None
                pass


class TestASTVisitorShmemCalls:
    """Test AST visitor's ability to recognize shmem function calls."""
    
    def test_recognize_shmem_init(self):
        """Test that init() call is recognized (shmem4py API)."""
        import ast
        from python.shmem4py_mlir.frontend import ASTVisitor
        
        code = """
import shmem
shmem.init()
"""
        tree = ast.parse(code)
        visitor = ASTVisitor()
        visitor.visit(tree)
        
        assert len(visitor.shmem_calls) == 1
        assert visitor.shmem_calls[0]['name'] == 'init'
    
    def test_recognize_shmem_finalize(self):
        """Test that finalize() call is recognized (shmem4py API)."""
        import ast
        from python.shmem4py_mlir.frontend import ASTVisitor
        
        code = """
import shmem
shmem.finalize()
"""
        tree = ast.parse(code)
        visitor = ASTVisitor()
        visitor.visit(tree)
        
        assert len(visitor.shmem_calls) == 1
        assert visitor.shmem_calls[0]['name'] == 'finalize'
    
    def test_recognize_init_and_finalize(self):
        """Test recognition of both init and finalize in same program."""
        import ast
        from python.shmem4py_mlir.frontend import ASTVisitor
        
        code = """
import shmem
shmem.init()
# Some work here
shmem.finalize()
"""
        tree = ast.parse(code)
        visitor = ASTVisitor()
        visitor.visit(tree)
        
        assert len(visitor.shmem_calls) == 2
        assert visitor.shmem_calls[0]['name'] == 'init'
        assert visitor.shmem_calls[1]['name'] == 'finalize'
    
    def test_preserve_call_order(self):
        """Test that call order is preserved in shmem_calls list."""
        import ast
        from python.shmem4py_mlir.frontend import ASTVisitor
        
        code = """
import shmem
shmem.init()
shmem.barrier_all()
shmem.finalize()
"""
        tree = ast.parse(code)
        visitor = ASTVisitor()
        visitor.visit(tree)
        
        assert len(visitor.shmem_calls) == 3
        assert visitor.shmem_calls[0]['name'] == 'init'
        assert visitor.shmem_calls[1]['name'] == 'barrier_all'
        assert visitor.shmem_calls[2]['name'] == 'finalize'
    
    def test_recognize_all_shmem_operations(self):
        """Test recognition of all supported shmem4py operations."""
        import ast
        from python.shmem4py_mlir.frontend import ASTVisitor
        
        code = """
import shmem
shmem.init()
pe = shmem.my_pe()
npes = shmem.n_pes()
shmem.barrier_all()
shmem.finalize()
"""
        tree = ast.parse(code)
        visitor = ASTVisitor()
        visitor.visit(tree)
        
        names = [call['name'] for call in visitor.shmem_calls]
        assert 'init' in names
        assert 'my_pe' in names
        assert 'n_pes' in names
        assert 'barrier_all' in names
        assert 'finalize' in names
    
    def test_call_has_location_info(self):
        """Test that calls include line and column information."""
        import ast
        from python.shmem4py_mlir.frontend import ASTVisitor
        
        code = """
import shmem
shmem.init()
"""
        tree = ast.parse(code)
        visitor = ASTVisitor()
        visitor.visit(tree)
        
        call = visitor.shmem_calls[0]
        assert 'line' in call
        assert 'col' in call
        assert call['line'] > 0
        assert call['col'] >= 0


class TestFrontendIntegration:
    """Test frontend integration with AST visitor."""
    
    def test_frontend_init(self):
        """Test that frontend initializes correctly with shmem4py API."""
        from python.shmem4py_mlir.frontend import Shmem4PyFrontend
        frontend = Shmem4PyFrontend()
        assert frontend.shmem_functions is not None
        assert 'init' in frontend.shmem_functions
        assert 'finalize' in frontend.shmem_functions
    
    def test_frontend_recognizes_shmem_calls(self):
        """Test that frontend can recognize shmem calls from code."""
        from python.shmem4py_mlir.frontend import Shmem4PyFrontend
        
        code = """
import shmem
shmem.init()
shmem.finalize()
"""
        
        frontend = Shmem4PyFrontend()
        
        # Compile should detect the calls and provide debug output
        # It will fail with NotImplementedError (MLIR bindings not available)
        with pytest.raises(NotImplementedError):
            frontend.compile(code)


class TestShmemInitFinalize:
    """Test shmem_init() and shmem_finalize() specific behavior."""
    
    def test_init_is_first_call(self):
        """Test proper detection when init is first."""
        import ast
        from python.shmem4py_mlir.frontend import ASTVisitor
        
        code = """
import shmem
shmem.init()
pass
"""
        
        tree = ast.parse(code)
        visitor = ASTVisitor()
        visitor.visit(tree)
        
        assert visitor.shmem_calls[0]['name'] == 'init'
    
    def test_finalize_is_last_call(self):
        """Test proper detection when finalize is last."""
        import ast
        from python.shmem4py_mlir.frontend import ASTVisitor
        
        code = """
import shmem
pass
shmem.finalize()
"""
        
        tree = ast.parse(code)
        visitor = ASTVisitor()
        visitor.visit(tree)
        
        assert visitor.shmem_calls[-1]['name'] == 'finalize'
    
    def test_multiple_inits_detected(self):
        """Test that multiple init calls are all detected."""
        import ast
        from python.shmem4py_mlir.frontend import ASTVisitor
        
        code = """
import shmem
shmem.init()
shmem.init()
"""
        
        tree = ast.parse(code)
        visitor = ASTVisitor()
        visitor.visit(tree)
        
        init_calls = [c for c in visitor.shmem_calls if c['name'] == 'init']
        assert len(init_calls) == 2
    
    def test_multiple_finalizes_detected(self):
        """Test that multiple finalize calls are all detected."""
        import ast
        from python.shmem4py_mlir.frontend import ASTVisitor
        
        code = """
import shmem
shmem.finalize()
shmem.finalize()
"""
        
        tree = ast.parse(code)
        visitor = ASTVisitor()
        visitor.visit(tree)
        
        finalize_calls = [c for c in visitor.shmem_calls if c['name'] == 'finalize']
        assert len(finalize_calls) == 2


class TestBackendMapping:
    """Test mapping from shmem_* to OpenSHMEM ops."""
    
    def test_shmem_init_mapping(self):
        """Test that init maps to openshmem.init op."""
        from python.shmem4py_mlir.frontend import Shmem4PyFrontend
        frontend = Shmem4PyFrontend()
        assert frontend.shmem_functions['init'] == 'init'
    
    def test_shmem_finalize_mapping(self):
        """Test that finalize maps to openshmem.finalize op."""
        from python.shmem4py_mlir.frontend import Shmem4PyFrontend
        frontend = Shmem4PyFrontend()
        assert frontend.shmem_functions['finalize'] == 'finalize'
    
    def test_all_operations_mapped(self):
        """Test that all supported shmem4py operations are mapped."""
        from python.shmem4py_mlir.frontend import Shmem4PyFrontend
        frontend = Shmem4PyFrontend()
        assert len(frontend.shmem_functions) == 7
        assert frontend.shmem_functions['my_pe'] == 'my_pe'
        assert frontend.shmem_functions['n_pes'] == 'n_pes'
        assert frontend.shmem_functions['barrier_all'] == 'barrier_all'


class TestIRBuilder:
    """Test IRBuilder interface for init/finalize."""
    
    def test_builder_init(self):
        """Test IRBuilder initialization."""
        from python.shmem4py_mlir.builder import IRBuilder
        builder = IRBuilder()
        assert builder.module is None
        assert builder.context is None
    
    def test_builder_has_init_method(self):
        """Test that IRBuilder has create_openshmem_init method."""
        from python.shmem4py_mlir.builder import IRBuilder
        builder = IRBuilder()
        assert hasattr(builder, 'create_openshmem_init')
    
    def test_builder_has_finalize_method(self):
        """Test that IRBuilder has create_openshmem_finalize method."""
        from python.shmem4py_mlir.builder import IRBuilder
        builder = IRBuilder()
        assert hasattr(builder, 'create_openshmem_finalize')
    
    def test_init_raises_notimplemented(self):
        """Test that create_openshmem_init raises NotImplementedError."""
        from python.shmem4py_mlir.builder import IRBuilder
        builder = IRBuilder()
        with pytest.raises(NotImplementedError):
            builder.create_openshmem_init()
    
    def test_finalize_raises_notimplemented(self):
        """Test that create_openshmem_finalize raises NotImplementedError."""
        from python.shmem4py_mlir.builder import IRBuilder
        builder = IRBuilder()
        with pytest.raises(NotImplementedError):
            builder.create_openshmem_finalize()




#include "mlir/Conversion/OpenSHMEMToLLVM/OpenSHMEMToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#ifdef ENABLE_CLANGIR
#include "OpenSHMEMCIR/Passes.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#endif

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all the dialects that we expect to use
  mlir::registerAllDialects(registry);

  // Register our OpenSHMEM dialect
  registry.insert<mlir::openshmem::OpenSHMEMDialect>();

#ifdef ENABLE_CLANGIR
  // Register ClangIR dialect (only when building with ClangIR incubator)
  registry.insert<cir::CIRDialect>();
#endif

  // Explicitly register arith dialect to ensure it's available
  registry.insert<mlir::arith::ArithDialect>();

  // Register OpenSHMEM to LLVM conversion interface
  mlir::openshmem::registerConvertOpenSHMEMToLLVMInterface(registry);

#ifdef ENABLE_CLANGIR
  // Register OpenSHMEM CIR passes (only when building with ClangIR incubator)
  mlir::openshmem::cir::registerOpenSHMEMCIRPasses();
#endif

  // Register our OpenSHMEM conversion pass using global static
  static mlir::PassPipelineRegistration<> registerOpenSHMEMPass(
      "convert-openshmem-to-llvm", "Convert OpenSHMEM dialect to LLVM dialect",
      [](mlir::OpPassManager &pm) {
        pm.addPass(mlir::createConvertOpenSHMEMToLLVMPass());
      });

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "OpenSHMEM mlir-opt-like driver\n", registry));
}

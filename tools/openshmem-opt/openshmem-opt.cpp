#include "mlir/Conversion/OpenSHMEMToLLVM/OpenSHMEMToLLVM.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "OpenSHMEMCIR/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all the dialects that we expect to use
  mlir::registerAllDialects(registry);

  // Register our OpenSHMEM dialect
  registry.insert<mlir::openshmem::OpenSHMEMDialect>();

  // Register OpenSHMEM to LLVM conversion interface
  mlir::openshmem::registerConvertOpenSHMEMToLLVMInterface(registry);

  // Register OpenSHMEM CIR passes
  mlir::openshmem::cir::registerOpenSHMEMCIRPasses();

  // Register our OpenSHMEM conversion pass using global static
  static mlir::PassPipelineRegistration<> registerOpenSHMEMPass(
      "convert-openshmem-to-llvm", "Convert OpenSHMEM dialect to LLVM dialect",
      [](mlir::OpPassManager &pm) {
        pm.addPass(mlir::createConvertOpenSHMEMToLLVMPass());
      });

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "OpenSHMEM mlir-opt-like driver\n", registry));
}

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/Conversion/OpenSHMEMToLLVM/OpenSHMEMToLLVM.h"
#include "mlir/InitAllDialects.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all the dialects that we expect to use
  mlir::registerAllDialects(registry);
  
  // Register our OpenSHMEM dialect
  registry.insert<mlir::openshmem::OpenSHMEMDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "OpenSHMEM mlir-opt-like driver\n", registry));
}

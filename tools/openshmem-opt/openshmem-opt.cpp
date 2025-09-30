#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"

// Include generated pass declarations and registration so we can register
// and construct the convert pass. Put the generated declarations into the
// `mlir` namespace so they match the symbols produced when the pass
// implementation was compiled into the library (that file includes the
// generated header inside namespace mlir as well).
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
namespace mlir {
#include "mlir/Conversion/OpenSHMEMToLLVM/OpenSHMEMConvertPasses.h.inc"
} // namespace mlir

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register the generated passes so they can be invoked from the tool.
  mlir::registerPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "OpenSHMEM mlir-opt-like driver\n", registry));
}

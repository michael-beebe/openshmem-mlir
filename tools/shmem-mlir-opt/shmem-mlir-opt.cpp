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
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

struct CIRToLLVMPipelineOptions
    : public mlir::PassPipelineOptions<CIRToLLVMPipelineOptions> {
  Option<bool> disableCCLowering{
      *this, "disable-cc-lowering",
      llvm::cl::desc("Skips calling convetion lowering pass."),
      llvm::cl::init(false)};
};
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

#ifdef ENABLE_CLANGIR
  // Register additional dialects typically used during CIR lowering that are
  // not covered by registerAllDialects for certain builds.
  registry.insert<mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect>();
#endif

  // Explicitly register arith dialect to ensure it's available
  registry.insert<mlir::arith::ArithDialect>();

  // Register OpenSHMEM to LLVM conversion interface
  mlir::openshmem::registerConvertOpenSHMEMToLLVMInterface(registry);

#ifdef ENABLE_CLANGIR
  // Register OpenSHMEM CIR passes (only when building with ClangIR incubator)
  mlir::openshmem::cir::registerOpenSHMEMCIRPasses();

  // Make the CIR-to-LLVM pipeline available so this driver can act as a
  // full replacement for cir-opt when desired.
  mlir::PassPipelineRegistration<CIRToLLVMPipelineOptions>
      cirToLLVMPipeline(
          "cir-to-llvm", "Convert CIR to LLVM dialect",
          [](mlir::OpPassManager &pm,
             const CIRToLLVMPipelineOptions &options) {
            cir::direct::populateCIRToLLVMPasses(pm,
                                                  options.disableCCLowering);
          });

  // Reconcile unrealized casts helper pass can be useful after conversions.
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createReconcileUnrealizedCastsPass();
  });
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

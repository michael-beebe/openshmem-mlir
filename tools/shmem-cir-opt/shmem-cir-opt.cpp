//===----------------------------------------------------------------------===//
//
// OpenSHMEM-enhanced CIR optimization tool
//
// This tool combines ClangIR (cir-opt) functionality with OpenSHMEM dialect
// support, allowing for complete CIR → OpenSHMEM → LLVM conversion in a
// single tool.
//
//===----------------------------------------------------------------------===//

#include "OpenSHMEMCIR/Passes.h"
#include "mlir/Conversion/OpenSHMEMToLLVM/OpenSHMEMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Passes.h"

struct CIRToLLVMPipelineOptions
    : public mlir::PassPipelineOptions<CIRToLLVMPipelineOptions> {};

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register core dialects
  registry.insert<cir::CIRDialect, mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect>();

  // Register OpenSHMEM dialect
  registry.insert<mlir::openshmem::OpenSHMEMDialect>();

  // Register OpenSHMEM to LLVM conversion interface
  mlir::openshmem::registerConvertOpenSHMEMToLLVMInterface(registry);

  // Register CIR passes
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIRCanonicalizePass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIRSimplifyPass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIRFlattenCFGPass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createHoistAllocasPass();
  });

  // Register CIR to LLVM pipeline
  mlir::PassPipelineRegistration<CIRToLLVMPipelineOptions> cirToLLVMPipeline(
      "cir-to-llvm", "Convert CIR to LLVM dialect",
      [](mlir::OpPassManager &pm, const CIRToLLVMPipelineOptions &options) {
        cir::direct::populateCIRToLLVMPasses(pm);
      });

  // Register OpenSHMEM CIR passes
  mlir::openshmem::cir::registerOpenSHMEMCIRPasses();

  // Register OpenSHMEM to LLVM conversion pass
  static mlir::PassPipelineRegistration<> openshmemToLLVMPipeline(
      "convert-openshmem-to-llvm", "Convert OpenSHMEM dialect to LLVM dialect",
      [](mlir::OpPassManager &pm) {
        pm.addPass(mlir::createConvertOpenSHMEMToLLVMPass());
      });

  // Register reconcile-unrealized-casts pass (useful for cleanup)
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createReconcileUnrealizedCastsPass();
  });

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "OpenSHMEM-enhanced ClangIR analysis and optimization tool\n",
      registry));
}

#include "mlir/Dialect/OpenSHMEM/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::openshmem;

namespace {
struct OpenSHMEMNoOp
    : public PassWrapper<OpenSHMEMNoOp, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpenSHMEMNoOp)
  StringRef getArgument() const final { return "openshmem-noop"; }
  StringRef getDescription() const final { return "No-op placeholder pass"; }
  void runOnOperation() final {}
};
} // namespace

std::unique_ptr<Pass> mlir::openshmem::createOpenSHMEMNoOpPass() {
  return std::make_unique<OpenSHMEMNoOp>();
}

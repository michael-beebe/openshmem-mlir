#ifndef MLIR_DIALECT_OPENSHMEM_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_OPENSHMEM_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace openshmem {

#define GEN_PASS_DECL
#include "mlir/Dialect/OpenSHMEM/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createMessageAggregationPass();
std::unique_ptr<Pass> createAtomicFusionPass();
std::unique_ptr<Pass> createAsyncConversionPass();
std::unique_ptr<Pass> createInjectNumPEsPass();
std::unique_ptr<Pass> createOpenSHMEMNoOpPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/OpenSHMEM/Transforms/Passes.h.inc"

} // namespace openshmem
} // namespace mlir

#endif // MLIR_DIALECT_OPENSHMEM_TRANSFORMS_PASSES_H

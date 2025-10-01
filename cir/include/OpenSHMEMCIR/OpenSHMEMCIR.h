//===- OpenSHMEMCIR.h - OpenSHMEM ClangIR Frontend ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the main interfaces for the OpenSHMEM ClangIR frontend.
//
//===----------------------------------------------------------------------===//

#ifndef OPENSHMEMCIR_OPENSHMEMCIR_H
#define OPENSHMEMCIR_OPENSHMEMCIR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace openshmem {
namespace cir {

/// Create a pass to convert ClangIR OpenSHMEM calls to OpenSHMEM dialect.
std::unique_ptr<Pass> createConvertCIRToOpenSHMEMPass();

/// Create a pass to recognize and annotate OpenSHMEM patterns in CIR.
std::unique_ptr<Pass> createOpenSHMEMRecognitionPass();

/// Create a pass to optimize OpenSHMEM operations in CIR form.
std::unique_ptr<Pass> createOpenSHMEMCIROptimizationPass();

/// Register all OpenSHMEM CIR passes.
void registerOpenSHMEMCIRPasses();

/// Check if a function call is an OpenSHMEM API function.
bool isOpenSHMEMAPICall(StringRef functionName);

/// Get the OpenSHMEM dialect operation name for a given API function.
StringRef getOpenSHMEMDialectOpName(StringRef apiFunction);

/// OpenSHMEM API function categories for analysis and transformation.
enum class OpenSHMEMAPICategory {
  Setup,           // shmem_init, shmem_finalize, etc.
  Memory,          // shmem_malloc, shmem_free, etc.
  RMA,             // shmem_put, shmem_get, etc.
  Atomics,         // shmem_atomic_*, etc.
  Collectives,     // shmem_broadcast, shmem_reduce, etc.
  Synchronization, // shmem_barrier, shmem_quiet, etc.
  Teams,           // shmem_team_*, etc.
  Contexts,        // shmem_ctx_*, etc.
  Pt2PtSync,       // shmem_wait, shmem_test, etc.
  Unknown
};

/// Classify an OpenSHMEM API function by category.
OpenSHMEMAPICategory classifyOpenSHMEMAPI(StringRef functionName);

} // namespace cir
} // namespace openshmem
} // namespace mlir

#endif // OPENSHMEMCIR_OPENSHMEMCIR_H

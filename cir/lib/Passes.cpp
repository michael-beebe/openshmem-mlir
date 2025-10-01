//===- Passes.cpp - OpenSHMEM CIR Pass Implementations -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements passes for transforming ClangIR OpenSHMEM code.
//
//===----------------------------------------------------------------------===//

#include "OpenSHMEMCIR/Passes.h"
#include "OpenSHMEMCIR/OpenSHMEMCIR.h"

#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include <memory>

#define DEBUG_TYPE "openshmem-cir"

using namespace mlir;
using namespace mlir::openshmem;

namespace mlir {
namespace openshmem {
namespace cir {

#define GEN_PASS_DEF_CONVERTCIRTOOPENSHMEM
#define GEN_PASS_DEF_OPENSHMEMRECOGNITION
#define GEN_PASS_DEF_OPENSHMEMCIROPTIMIZATION
#include "OpenSHMEMCIR/Passes.h.inc"

//===----------------------------------------------------------------------===//
// OpenSHMEM API Recognition Utilities
//===----------------------------------------------------------------------===//

/// Map of OpenSHMEM API function names to their categories and properties.
static const llvm::StringMap<OpenSHMEMAPICategory> openSHMEMAPIMap = {
    // Setup operations
    {"shmem_init", OpenSHMEMAPICategory::Setup},
    {"shmem_finalize", OpenSHMEMAPICategory::Setup},
    {"shmem_my_pe", OpenSHMEMAPICategory::Setup},
    {"shmem_n_pes", OpenSHMEMAPICategory::Setup},
    
    // Memory management
    {"shmem_malloc", OpenSHMEMAPICategory::Memory},
    {"shmem_calloc", OpenSHMEMAPICategory::Memory},
    {"shmem_realloc", OpenSHMEMAPICategory::Memory},
    {"shmem_align", OpenSHMEMAPICategory::Memory},
    {"shmem_free", OpenSHMEMAPICategory::Memory},
    
    // RMA operations
    {"shmem_put", OpenSHMEMAPICategory::RMA},
    {"shmem_get", OpenSHMEMAPICategory::RMA},
    {"shmem_put_nbi", OpenSHMEMAPICategory::RMA},
    {"shmem_get_nbi", OpenSHMEMAPICategory::RMA},
    {"shmem_putmem", OpenSHMEMAPICategory::RMA},
    {"shmem_getmem", OpenSHMEMAPICategory::RMA},
    {"shmem_putmem_nbi", OpenSHMEMAPICategory::RMA},
    {"shmem_getmem_nbi", OpenSHMEMAPICategory::RMA},
    
    // Context-aware RMA
    {"shmem_ctx_put", OpenSHMEMAPICategory::RMA},
    {"shmem_ctx_get", OpenSHMEMAPICategory::RMA},
    {"shmem_ctx_put_nbi", OpenSHMEMAPICategory::RMA},
    {"shmem_ctx_get_nbi", OpenSHMEMAPICategory::RMA},
    
    // Atomics
    {"shmem_atomic_fetch", OpenSHMEMAPICategory::Atomics},
    {"shmem_atomic_set", OpenSHMEMAPICategory::Atomics},
    {"shmem_atomic_add", OpenSHMEMAPICategory::Atomics},
    {"shmem_atomic_inc", OpenSHMEMAPICategory::Atomics},
    {"shmem_atomic_swap", OpenSHMEMAPICategory::Atomics},
    {"shmem_atomic_compare_swap", OpenSHMEMAPICategory::Atomics},
    {"shmem_atomic_fetch_add", OpenSHMEMAPICategory::Atomics},
    {"shmem_atomic_fetch_inc", OpenSHMEMAPICategory::Atomics},
    
    // Collectives
    {"shmem_broadcast", OpenSHMEMAPICategory::Collectives},
    {"shmem_collect", OpenSHMEMAPICategory::Collectives},
    {"shmem_fcollect", OpenSHMEMAPICategory::Collectives},
    {"shmem_alltoall", OpenSHMEMAPICategory::Collectives},
    {"shmem_alltoalls", OpenSHMEMAPICategory::Collectives},
    {"shmem_sum_reduce", OpenSHMEMAPICategory::Collectives},
    {"shmem_max_reduce", OpenSHMEMAPICategory::Collectives},
    {"shmem_min_reduce", OpenSHMEMAPICategory::Collectives},
    
    // Synchronization
    {"shmem_barrier_all", OpenSHMEMAPICategory::Synchronization},
    {"shmem_barrier", OpenSHMEMAPICategory::Synchronization},
    {"shmem_quiet", OpenSHMEMAPICategory::Synchronization},
    
    // Teams
    {"shmem_team_split_strided", OpenSHMEMAPICategory::Teams},
    {"shmem_team_split_2d", OpenSHMEMAPICategory::Teams},
    {"shmem_team_destroy", OpenSHMEMAPICategory::Teams},
    {"shmem_team_sync", OpenSHMEMAPICategory::Teams},
    
    // Contexts
    {"shmem_ctx_create", OpenSHMEMAPICategory::Contexts},
    {"shmem_ctx_destroy", OpenSHMEMAPICategory::Contexts},
    {"shmem_team_create_ctx", OpenSHMEMAPICategory::Contexts},
    
    // Point-to-point synchronization
    {"shmem_wait_until", OpenSHMEMAPICategory::Pt2PtSync},
    {"shmem_test", OpenSHMEMAPICategory::Pt2PtSync},
    {"shmem_wait_until_all", OpenSHMEMAPICategory::Pt2PtSync},
    {"shmem_wait_until_any", OpenSHMEMAPICategory::Pt2PtSync},
    {"shmem_wait_until_some", OpenSHMEMAPICategory::Pt2PtSync},
    {"shmem_test_all", OpenSHMEMAPICategory::Pt2PtSync},
    {"shmem_test_any", OpenSHMEMAPICategory::Pt2PtSync},
    {"shmem_signal_wait_until", OpenSHMEMAPICategory::Pt2PtSync},
};

bool isOpenSHMEMAPICall(StringRef functionName) {
  // Check exact matches first
  if (openSHMEMAPIMap.count(functionName))
    return true;
    
  // Check for typed variants (e.g., shmem_int_put, shmem_float_get)
  if (functionName.starts_with("shmem_") && 
      (functionName.contains("_put") || functionName.contains("_get") ||
       functionName.contains("_atomic") || functionName.contains("_reduce") ||
       functionName.contains("_broadcast") || functionName.contains("_collect")))
    return true;
    
  // Check for sized variants (e.g., shmem_put32, shmem_get64)
  if (functionName.starts_with("shmem_") && 
      (functionName.ends_with("8") || functionName.ends_with("16") ||
       functionName.ends_with("32") || functionName.ends_with("64") ||
       functionName.ends_with("128")))
    return true;
    
  return false;
}

OpenSHMEMAPICategory classifyOpenSHMEMAPI(StringRef functionName) {
  auto it = openSHMEMAPIMap.find(functionName);
  if (it != openSHMEMAPIMap.end())
    return it->second;
    
  // For typed/sized variants, extract the base operation
  if (functionName.starts_with("shmem_")) {
    // Remove type prefixes (int_, float_, etc.)
    StringRef baseName = functionName;
    if (baseName.contains("_put"))
      return OpenSHMEMAPICategory::RMA;
    if (baseName.contains("_get"))
      return OpenSHMEMAPICategory::RMA;
    if (baseName.contains("_atomic"))
      return OpenSHMEMAPICategory::Atomics;
    if (baseName.contains("_reduce"))
      return OpenSHMEMAPICategory::Collectives;
    if (baseName.contains("_broadcast") || baseName.contains("_collect"))
      return OpenSHMEMAPICategory::Collectives;
  }
  
  return OpenSHMEMAPICategory::Unknown;
}

StringRef getOpenSHMEMDialectOpName(StringRef apiFunction) {
  // Map API function names to dialect operation names
  // Remove "shmem_" prefix and handle special cases
  if (apiFunction.starts_with("shmem_")) {
    StringRef opName = apiFunction.drop_front(6); // Remove "shmem_"
    
    // Handle some special mappings
    if (opName == "barrier_all")
      return "barrier_all";
    if (opName == "my_pe")
      return "my_pe";
    if (opName == "n_pes")
      return "n_pes";
      
    return opName;
  }
  
  return apiFunction;
}

//===----------------------------------------------------------------------===//
// ConvertCIRToOpenSHMEM Pass Implementation
//===----------------------------------------------------------------------===//

struct ConvertCIRToOpenSHMEMPass 
    : public impl::ConvertCIRToOpenSHMEMBase<ConvertCIRToOpenSHMEMPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // TODO: Implement CIR to OpenSHMEM conversion patterns
    // This will be implemented once we have ClangIR dialect integration
    
    module.walk([&](Operation *op) {
      // Look for CIR call operations that match OpenSHMEM API
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        StringRef callee = callOp.getCallee();
        if (isOpenSHMEMAPICall(callee)) {
          // Mark for conversion
          LLVM_DEBUG(llvm::dbgs() << "Found OpenSHMEM API call: " << callee << "\n");
          
          // TODO: Convert to OpenSHMEM dialect operation
          // This requires pattern-based rewriting with proper type conversion
        }
      }
    });
  }
};

//===----------------------------------------------------------------------===//
// OpenSHMEMRecognition Pass Implementation
//===----------------------------------------------------------------------===//

struct OpenSHMEMRecognitionPass 
    : public impl::OpenSHMEMRecognitionBase<OpenSHMEMRecognitionPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Analyze OpenSHMEM usage patterns
    llvm::DenseMap<OpenSHMEMAPICategory, unsigned> apiUsageCount;
    
    module.walk([&](func::CallOp callOp) {
      StringRef callee = callOp.getCallee();
      if (isOpenSHMEMAPICall(callee)) {
        OpenSHMEMAPICategory category = classifyOpenSHMEMAPI(callee);
        apiUsageCount[category]++;
        
        // Add attributes to mark OpenSHMEM operations
        callOp->setAttr("openshmem.api_call", 
                       StringAttr::get(&getContext(), callee));
        callOp->setAttr("openshmem.category", 
                       IntegerAttr::get(IntegerType::get(&getContext(), 32), 
                                      static_cast<int>(category)));
      }
    });
    
    // Report analysis results
    LLVM_DEBUG({
      llvm::dbgs() << "OpenSHMEM API Usage Analysis:\n";
      for (auto &entry : apiUsageCount) {
        llvm::dbgs() << "  Category " << static_cast<int>(entry.first) 
                     << ": " << entry.second << " calls\n";
      }
    });
  }
};

//===----------------------------------------------------------------------===//
// OpenSHMEMCIROptimization Pass Implementation
//===----------------------------------------------------------------------===//

struct OpenSHMEMCIROptimizationPass 
    : public impl::OpenSHMEMCIROptimizationBase<OpenSHMEMCIROptimizationPass> {

  void runOnOperation() override {
    // TODO: Implement CIR-level OpenSHMEM optimizations
    // Examples:
    // - Combine multiple small put/get operations
    // - Eliminate redundant barriers
    // - Optimize allocation patterns
    (void)getOperation(); // Mark as used to avoid warnings
    
    LLVM_DEBUG(llvm::dbgs() << "Running OpenSHMEM CIR optimizations\n");
  }
};

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createConvertCIRToOpenSHMEMPass() {
  return std::make_unique<ConvertCIRToOpenSHMEMPass>();
}

std::unique_ptr<Pass> createOpenSHMEMRecognitionPass() {
  return std::make_unique<OpenSHMEMRecognitionPass>();
}

std::unique_ptr<Pass> createOpenSHMEMCIROptimizationPass() {
  return std::make_unique<OpenSHMEMCIROptimizationPass>();
}

// Note: registerOpenSHMEMCIRPasses() is generated by tablegen

} // namespace cir
} // namespace openshmem
} // namespace mlir

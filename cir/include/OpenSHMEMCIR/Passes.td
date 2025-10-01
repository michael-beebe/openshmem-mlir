//===- Passes.td - OpenSHMEM CIR pass definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENSHMEMCIR_PASSES
#define OPENSHMEMCIR_PASSES

include "mlir/Pass/PassBase.td"

def ConvertCIRToOpenSHMEM : Pass<"convert-cir-to-openshmem", "mlir::ModuleOp"> {
  let summary = "Convert ClangIR OpenSHMEM calls to OpenSHMEM dialect";
  let description = [{
    This pass converts ClangIR function calls that correspond to OpenSHMEM API
    functions into operations from the OpenSHMEM MLIR dialect. This enables
    semantic-level optimizations on OpenSHMEM operations and provides a path
    for further lowering to LLVM.

    Example transformation:
    ```
    cir.call @shmem_put(%dest, %src, %size, %pe) : (!cir.ptr<i32>, !cir.ptr<i32>, i64, i32) -> ()
    ```
    becomes:
    ```
    openshmem.put(%dest, %src, %size, %pe) : memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32>, index, i32
    ```
  }];
  let constructor = "mlir::openshmem::cir::createConvertCIRToOpenSHMEMPass()";
  let dependentDialects = ["openshmem::OpenSHMEMDialect"];
}

def OpenSHMEMRecognition : Pass<"openshmem-recognition", "mlir::ModuleOp"> {
  let summary = "Recognize and annotate OpenSHMEM patterns in CIR";
  let description = [{
    This pass analyzes ClangIR code to identify OpenSHMEM API usage patterns
    and annotates them with attributes that can guide later transformations.
    It performs analysis to understand:
    
    - OpenSHMEM memory allocation patterns
    - Communication patterns (put/get operations)
    - Synchronization patterns
    - Collective operation usage
    
    This information can be used for optimization and verification.
  }];
  let constructor = "mlir::openshmem::cir::createOpenSHMEMRecognitionPass()";
}

def OpenSHMEMCIROptimization : Pass<"openshmem-cir-optimize", "mlir::ModuleOp"> {
  let summary = "Optimize OpenSHMEM operations in CIR form";
  let description = [{
    This pass performs optimizations on OpenSHMEM operations while they are
    still in ClangIR form, taking advantage of higher-level semantic information
    available at this stage.
    
    Optimizations include:
    - Combining multiple small transfers into larger ones
    - Eliminating redundant synchronization
    - Optimizing collective operation patterns
    - Memory allocation and deallocation optimization
  }];
  let constructor = "mlir::openshmem::cir::createOpenSHMEMCIROptimizationPass()";
}

#endif // OPENSHMEMCIR_PASSES

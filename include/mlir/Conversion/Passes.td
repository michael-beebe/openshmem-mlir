//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_PASSES
#define MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Dialect/Vector/Transforms/VectorTransformsBase.td"

// TODO: should we include other dialects?

//===----------------------------------------------------------------------===//
// OpenSHMEMToLLVM
//===----------------------------------------------------------------------===//

def ConvertOpenSHMEMToLLVMPass : Pass<"convert-openshmem-to-llvm", "ModuleOp"> {
  let summary = "Convert OpenSHMEM dialect to LLVM dialect";
  let description = [{
    This pass converts OpenSHMEM operations to LLVM function calls to the
    OpenSHMEM runtime library. It transforms high-level OpenSHMEM operations
    like init, finalize, put, get, malloc, and free into corresponding
    shmem_* function calls.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif // MLIR_CONVERSION_PASSES

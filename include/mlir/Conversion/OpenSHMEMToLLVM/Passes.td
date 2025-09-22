//===- Passes.td - OpenSHMEM to LLVM conversion passes --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def ConvertOpenSHMEMToLLVMPass : Pass<"convert-openshmem-to-llvm", "ModuleOp"> {
  let summary = "Lower OpenSHMEM dialect to LLVM dialect";
  let description = [{
    Convert OpenSHMEM ops to the LLVM dialect.
  }];
}



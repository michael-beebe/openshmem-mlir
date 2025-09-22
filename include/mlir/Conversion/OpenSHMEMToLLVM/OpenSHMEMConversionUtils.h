//===- OpenSHMEMConversionUtils.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENSHMEMTOLLVM_OPENSHMEMCONVERSIONUTILS_H
#define MLIR_CONVERSION_OPENSHMEMTOLLVM_OPENSHMEMCONVERSIONUTILS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::openshmem {

/// Get or insert a declaration for an LLVM function with the given name/type.
LLVM::LLVMFuncOp getOrDefineFunction(ModuleOp module, Location loc,
                                     PatternRewriter &rewriter, StringRef name,
                                     LLVM::LLVMFunctionType funcTy);

} // namespace mlir::openshmem

#endif // MLIR_CONVERSION_OPENSHMEMTOLLVM_OPENSHMEMCONVERSIONUTILS_H

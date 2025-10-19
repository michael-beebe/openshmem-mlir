//===- BridgeOpsToLLVM.h - Bridge op conversion patterns -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENSHMEMTOLLVM_BRIDGEOPSTOLLVM_H
#define MLIR_CONVERSION_OPENSHMEMTOLLVM_BRIDGEOPSTOLLVM_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;

namespace openshmem {

void populateBridgeOpsToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns);

} // namespace openshmem
} // namespace mlir

#endif // MLIR_CONVERSION_OPENSHMEMTOLLVM_BRIDGEOPSTOLLVM_H

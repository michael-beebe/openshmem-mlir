//===- TeamOpsToLLVM.h - Team operations conversion patterns ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_CONVERSION_OPENSHMEMTOLLVM_TEAMOPSTOLLVM_H
#define MLIR_LIB_CONVERSION_OPENSHMEMTOLLVM_TEAMOPSTOLLVM_H

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;

namespace openshmem {

/// Populate conversion patterns for OpenSHMEM Team operations.
void populateTeamOpsToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);

} // namespace openshmem
} // namespace mlir

#endif // MLIR_LIB_CONVERSION_OPENSHMEMTOLLVM_TEAMOPSTOLLVM_H
 
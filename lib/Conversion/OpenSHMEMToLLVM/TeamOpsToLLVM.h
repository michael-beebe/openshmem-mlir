//===- TeamOpsToLLVM.h - Team Ops Conversion -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENSHMEM_LIB_CONVERSION_TEAMOPSTOLLVM_H
#define OPENSHMEM_LIB_CONVERSION_TEAMOPSTOLLVM_H

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;

namespace openshmem {
void populateTeamOpsToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);
} // namespace openshmem
} // namespace mlir

#endif // OPENSHMEM_LIB_CONVERSION_TEAMOPSTOLLVM_H

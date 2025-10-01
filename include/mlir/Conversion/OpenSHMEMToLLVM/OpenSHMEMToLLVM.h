//===- OpenSHMEMToLLVM.h - OpenSHMEM to LLVM conversion --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENSHMEMTOLLVM_H_
#define MLIR_CONVERSION_OPENSHMEMTOLLVM_H_

#include <memory>

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;
class Pass;
class DialectRegistry;

/// Create a pass to convert OpenSHMEM dialect to LLVM.
std::unique_ptr<Pass> createConvertOpenSHMEMToLLVMPass();

namespace openshmem {

/// Populate the given list with patterns that convert from OpenSHMEM to LLVM.
void populateOpenSHMEMToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns);

/// Register the OpenSHMEM to LLVM conversion interface.
void registerConvertOpenSHMEMToLLVMInterface(DialectRegistry &registry);

} // namespace openshmem
} // namespace mlir

#endif // MLIR_CONVERSION_OPENSHMEMTOLLVM_H_

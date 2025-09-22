//===- OpenSHMEMToLLVM.h - OpenSHMEM to LLVM conversion ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENSHMEMTOLLVM_OPENSHMEMTOLLVM_H
#define MLIR_CONVERSION_OPENSHMEMTOLLVM_OPENSHMEMTOLLVM_H

#include <memory>

namespace mlir {
class Pass;
class LLVMTypeConverter;
class RewritePatternSet;

// Local generated pass decls for this project
#define GEN_PASS_DECL_CONVERTOPENSHMEMTOLLVMPASS
#include "mlir/Conversion/OpenSHMEMToLLVM/OpenSHMEMConvertPasses.h.inc"

namespace openshmem {
// Converter population with type converter
void populateOpenSHMEMToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns);

// Optional registry hook (stub for now)
void registerConvertOpenSHMEMToLLVMInterface(class DialectRegistry &registry);

// Pass factory (keep exported for tools/tests)
std::unique_ptr<Pass> createConvertOpenSHMEMToLLVMPass();
} // namespace openshmem
} // namespace mlir

#endif // MLIR_CONVERSION_OPENSHMEMTOLLVM_OPENSHMEMTOLLVM_H

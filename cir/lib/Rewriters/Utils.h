//===- Utils.h - OpenSHMEM CIR Utilities ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utility functions for OpenSHMEM CIR transformations.
//
//===----------------------------------------------------------------------===//

#ifndef OPENSHMEMCIR_REWRITERS_UTILS_H
#define OPENSHMEMCIR_REWRITERS_UTILS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace openshmem {
namespace cir {

/// Populate patterns for converting CIR OpenSHMEM calls to OpenSHMEM dialect.
void populateCIRToOpenSHMEMConversionPatterns(RewritePatternSet &patterns);

/// Check if a type represents a symmetric memory allocation.
bool isSymmetricMemoryType(Type type);

/// Convert a CIR pointer type to an OpenSHMEM memref type.
Type convertCIRPtrToOpenSHMEMMemRef(Type cirPtrType, MLIRContext *context);

/// Extract element type from a CIR pointer type.
Type getElementTypeFromCIRPtr(Type cirPtrType);

} // namespace cir
} // namespace openshmem
} // namespace mlir

#endif // OPENSHMEMCIR_REWRITERS_UTILS_H

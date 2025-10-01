//===- Passes.h - OpenSHMEM CIR Passes ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines passes for OpenSHMEM ClangIR transformations.
//
//===----------------------------------------------------------------------===//

#ifndef OPENSHMEMCIR_PASSES_H
#define OPENSHMEMCIR_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace openshmem {
namespace cir {

#define GEN_PASS_DECL
#include "OpenSHMEMCIR/Passes.h.inc"

// Pass creation functions
std::unique_ptr<Pass> createConvertCIRToOpenSHMEMPass();
std::unique_ptr<Pass> createOpenSHMEMRecognitionPass();
std::unique_ptr<Pass> createOpenSHMEMCIROptimizationPass();

#define GEN_PASS_REGISTRATION
#include "OpenSHMEMCIR/Passes.h.inc"

} // namespace cir
} // namespace openshmem
} // namespace mlir

#endif // OPENSHMEMCIR_PASSES_H

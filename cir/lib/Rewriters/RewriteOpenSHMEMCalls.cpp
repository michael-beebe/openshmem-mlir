//===- RewriteOpenSHMEMCalls.cpp - CIR->OpenSHMEM registrar -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thin registrar for CIR -> OpenSHMEM conversion patterns. Concrete pattern
// implementations are split by category into separate translation units
// (RewriteSetup.cpp, RewriteMemory.cpp, RewriteSync.cpp). This file exposes a
// single entry point used to populate all CIR->OpenSHMEM patterns.
//
//===----------------------------------------------------------------------===//

#include "OpenSHMEMCIR/OpenSHMEMCIR.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h" // for RewritePatternSet

// Declarations for category registration functions implemented in separate
// translation units. These will be defined in the corresponding translation
// units (RewriteSetup.cpp, RewriteMemory.cpp, RewriteSync.cpp).

namespace mlir {
namespace openshmem {
namespace cir {

// Forward declarations for the category registration functions implemented
// in separate translation units.
void populateCIRToOpenSHMEMSetupPatterns(RewritePatternSet &patterns);
void populateCIRToOpenSHMEMMemoryPatterns(RewritePatternSet &patterns);
void populateCIRToOpenSHMEMSyncPatterns(RewritePatternSet &patterns);
void populateCIRToOpenSHMEMTeamPatterns(RewritePatternSet &patterns);
void populateCIRToOpenSHMEMContextPatterns(RewritePatternSet &patterns);
void populateCIRToOpenSHMEMCollectivePatterns(RewritePatternSet &patterns);
void populateCIRToOpenSHMEMRMAPatterns(RewritePatternSet &patterns);
void populateCIRToOpenSHMEMPt2ptSyncPatterns(RewritePatternSet &patterns);

void populateCIRToOpenSHMEMConversionPatterns(RewritePatternSet &patterns) {
  populateCIRToOpenSHMEMSetupPatterns(patterns);
  populateCIRToOpenSHMEMMemoryPatterns(patterns);
  populateCIRToOpenSHMEMSyncPatterns(patterns);
  populateCIRToOpenSHMEMTeamPatterns(patterns);
  populateCIRToOpenSHMEMContextPatterns(patterns);
  populateCIRToOpenSHMEMCollectivePatterns(patterns);
  populateCIRToOpenSHMEMRMAPatterns(patterns);
  populateCIRToOpenSHMEMPt2ptSyncPatterns(patterns);
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

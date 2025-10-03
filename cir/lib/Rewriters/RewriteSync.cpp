//===- RewriteSync.cpp - CIR to OpenSHMEM Sync Rewriters -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Patterns for converting CIR sync-related OpenSHMEM calls (barrier_all,
// quiet) to the OpenSHMEM dialect.
//
//===----------------------------------------------------------------------===//

#include "OpenSHMEMCIR/OpenSHMEMCIR.h"
#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace mlir::openshmem;

namespace mlir {
namespace openshmem {
namespace cir {

/// Pattern to convert shmem_barrier_all() calls to openshmem.barrier_all
struct ConvertShmemBarrierAllPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_barrier_all")
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::BarrierAllOp>(callOp);
    return success();
  }
};

/// Pattern to convert shmem_quiet() calls to openshmem.quiet
struct ConvertShmemQuietPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_quiet")
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::QuietOp>(callOp);
    return success();
  }
};

// Pattern registration for sync patterns
void populateCIRToOpenSHMEMSyncPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertShmemBarrierAllPattern, ConvertShmemQuietPattern>(
      patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

//===- RewriteSetup.cpp - CIR to OpenSHMEM Setup Rewriters -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains rewrite patterns for converting CIR OpenSHMEM setup and
// query calls to OpenSHMEM dialect operations (init/finalize/my_pe/n_pes).
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

/// Pattern to convert shmem_init() calls to openshmem.init
struct ConvertShmemInitPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_init")
      return failure();

    // Create openshmem.init operation
    rewriter.replaceOpWithNewOp<openshmem::InitOp>(callOp);
    return success();
  }
};

/// Pattern to convert shmem_finalize() calls to openshmem.finalize
struct ConvertShmemFinalizePattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_finalize")
      return failure();

    // Create openshmem.finalize operation
    rewriter.replaceOpWithNewOp<openshmem::FinalizeOp>(callOp);
    return success();
  }
};

/// Pattern to convert shmem_my_pe() calls to openshmem.my_pe
struct ConvertShmemMyPePattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_my_pe")
      return failure();

    // Create openshmem.my_pe operation
    Type i32Type = rewriter.getI32Type();
    auto myPeOp = rewriter.create<openshmem::MyPeOp>(callOp.getLoc(), i32Type);

    rewriter.replaceOp(callOp, myPeOp.getResult());
    return success();
  }
};

/// Pattern to convert shmem_n_pes() calls to openshmem.n_pes
struct ConvertShmemNPesPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_n_pes")
      return failure();

    // Create openshmem.n_pes operation
    Type i32Type = rewriter.getI32Type();
    auto nPesOp = rewriter.create<openshmem::NPesOp>(callOp.getLoc(), i32Type);

    rewriter.replaceOp(callOp, nPesOp.getResult());
    return success();
  }
};

// Pattern registration for setup/query patterns
void populateCIRToOpenSHMEMSetupPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertShmemInitPattern, ConvertShmemFinalizePattern,
               ConvertShmemMyPePattern, ConvertShmemNPesPattern>(
      patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

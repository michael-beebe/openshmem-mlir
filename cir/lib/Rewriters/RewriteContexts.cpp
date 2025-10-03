//===- RewriteContexts.cpp - CIR to OpenSHMEM Context Rewriters -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Patterns for converting CIR context-related OpenSHMEM calls to OpenSHMEM
// dialect operations (ctx_create, ctx_destroy, team_create_ctx, ctx_get_team).
//
//===----------------------------------------------------------------------===//

#include "OpenSHMEMCIR/OpenSHMEMCIR.h"
#include "Utils.h"

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

/// Pattern to convert shmem_ctx_create() to openshmem.ctx_create
struct ConvertCtxCreatePattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_ctx_create")
      return failure();

    if (callOp.getArgOperands().size() != 1)
      return failure();

    Value options = callOp.getArgOperands()[0];
    auto ctxType = rewriter.getType<::mlir::openshmem::CtxType>();
    auto i32Type = rewriter.getI32Type();

    auto res = rewriter.create<openshmem::CtxCreateOp>(
        callOp.getLoc(), ArrayRef<Type>{ctxType, i32Type}, ValueRange{options});

    rewriter.replaceOp(callOp, res.getResults());
    return success();
  }
};

/// Pattern to convert shmem_ctx_destroy() to openshmem.ctx_destroy
struct ConvertCtxDestroyPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_ctx_destroy")
      return failure();

    if (callOp.getArgOperands().size() != 1)
      return failure();

    Value ctx = callOp.getArgOperands()[0];
    rewriter.replaceOpWithNewOp<openshmem::CtxDestroyOp>(callOp, ctx);
    return success();
  }
};

/// Pattern to convert shmem_team_create_ctx() to openshmem.team_create_ctx
struct ConvertTeamCreateCtxPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_team_create_ctx")
      return failure();

    if (callOp.getArgOperands().size() != 2)
      return failure();

    Value team = callOp.getArgOperands()[0];
    Value options = callOp.getArgOperands()[1];

    auto ctxType = rewriter.getType<::mlir::openshmem::CtxType>();
    auto i32Type = rewriter.getI32Type();

    auto res = rewriter.create<openshmem::TeamCreateCtxOp>(
        callOp.getLoc(), ArrayRef<Type>{ctxType, i32Type},
        ValueRange{team, options});

    rewriter.replaceOp(callOp, res.getResults());
    return success();
  }
};

/// Pattern to convert shmem_ctx_get_team() to openshmem.ctx_get_team
struct ConvertCtxGetTeamPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_ctx_get_team")
      return failure();

    if (callOp.getArgOperands().size() != 1)
      return failure();

    Value ctx = callOp.getArgOperands()[0];
    auto teamType = rewriter.getType<::mlir::openshmem::TeamType>();
    auto i32Type = rewriter.getI32Type();

    auto res = rewriter.create<openshmem::CtxGetTeamOp>(
        callOp.getLoc(), ArrayRef<Type>{teamType, i32Type}, ValueRange{ctx});

    rewriter.replaceOp(callOp, res.getResults());
    return success();
  }
};

// Pattern registration for context patterns
void populateCIRToOpenSHMEMContextPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertCtxCreatePattern, ConvertCtxDestroyPattern,
               ConvertTeamCreateCtxPattern, ConvertCtxGetTeamPattern>(
      patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

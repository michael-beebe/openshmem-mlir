//===- RewriteTeams.cpp - CIR to OpenSHMEM Team Rewriters -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Patterns for converting CIR team-related OpenSHMEM calls to OpenSHMEM
// dialect operations (team_world, team_shared, team_split_strided,
// team_split_2d, team_destroy, team_sync, etc.).
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

/// Pattern to convert shmem_team_world() to openshmem.team_world
struct ConvertTeamWorldPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_team_world")
      return failure();

    auto teamType = rewriter.getType<::mlir::openshmem::TeamType>();
    auto teamOp =
        rewriter.create<openshmem::TeamWorldOp>(callOp.getLoc(), teamType);
    rewriter.replaceOp(callOp, teamOp.getResult());
    return success();
  }
};

/// Pattern to convert shmem_team_shared() to openshmem.team_shared
struct ConvertTeamSharedPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_team_shared")
      return failure();

    auto teamType = rewriter.getType<::mlir::openshmem::TeamType>();
    auto teamOp =
        rewriter.create<openshmem::TeamSharedOp>(callOp.getLoc(), teamType);
    rewriter.replaceOp(callOp, teamOp.getResult());
    return success();
  }
};

/// Pattern to convert shmem_team_split_strided() to
/// openshmem.team_split_strided
struct ConvertTeamSplitStridedPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_team_split_strided")
      return failure();

    // Expect args: parent team, start, stride, n_pes
    if (callOp.getArgOperands().size() != 4)
      return failure();

    Value parent = callOp.getArgOperands()[0];
    Value start = callOp.getArgOperands()[1];
    Value stride = callOp.getArgOperands()[2];
    Value nPes = callOp.getArgOperands()[3];

    auto teamType = rewriter.getType<::mlir::openshmem::TeamType>();
    auto i32Type = rewriter.getI32Type();

    auto newTeam = rewriter.create<openshmem::TeamSplitStridedOp>(
        callOp.getLoc(), ArrayRef<Type>{teamType, i32Type},
        ValueRange{parent, start, stride, nPes});

    // Replace with the results (team, retval)
    rewriter.replaceOp(callOp, newTeam.getResults());
    return success();
  }
};

/// Pattern to convert shmem_team_split_2d() to openshmem.team_split_2d
struct ConvertTeamSplit2dPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_team_split_2d")
      return failure();

    // Expect args: parent team, xrange
    if (callOp.getArgOperands().size() != 2)
      return failure();

    Value parent = callOp.getArgOperands()[0];
    Value xrange = callOp.getArgOperands()[1];

    auto teamType = rewriter.getType<::mlir::openshmem::TeamType>();
    auto i32Type = rewriter.getI32Type();

    auto newTeams = rewriter.create<openshmem::TeamSplit2dOp>(
        callOp.getLoc(), ArrayRef<Type>{teamType, teamType, i32Type},
        ValueRange{parent, xrange});

    rewriter.replaceOp(callOp, newTeams.getResults());
    return success();
  }
};

/// Pattern to convert shmem_team_destroy() to openshmem.team_destroy
struct ConvertTeamDestroyPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_team_destroy")
      return failure();

    if (callOp.getArgOperands().size() != 1)
      return failure();

    Value team = callOp.getArgOperands()[0];
    rewriter.replaceOpWithNewOp<openshmem::TeamDestroyOp>(callOp, team);
    return success();
  }
};

/// Pattern to convert shmem_team_sync() to openshmem.team_sync
struct ConvertTeamSyncPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_team_sync")
      return failure();

    if (callOp.getArgOperands().size() != 1)
      return failure();

    Value team = callOp.getArgOperands()[0];
    rewriter.replaceOpWithNewOp<openshmem::TeamSyncOp>(callOp, team);
    return success();
  }
};

// Pattern registration for team patterns
void populateCIRToOpenSHMEMTeamPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertTeamWorldPattern, ConvertTeamSharedPattern,
               ConvertTeamSplitStridedPattern, ConvertTeamSplit2dPattern,
               ConvertTeamDestroyPattern, ConvertTeamSyncPattern>(
      patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

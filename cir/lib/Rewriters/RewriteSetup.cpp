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

/// Pattern to wrap function body in openshmem.region (replaces init/finalize)
struct WrapFunctionInRegionPattern : public OpRewritePattern<::cir::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    // Only process functions with bodies (not declarations)
    if (funcOp.getBody().empty())
      return failure();

    // Check if the function contains shmem_init/finalize calls
    bool hasInit = false;
    bool hasFinalize = false;
    
    funcOp.walk([&](::cir::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (callee) {
        if (callee.value() == "shmem_init")
          hasInit = true;
        else if (callee.value() == "shmem_finalize")
          hasFinalize = true;
      }
    });

    // Only wrap if we found both init and finalize
    if (!hasInit || !hasFinalize)
      return failure();

    // Get the function body
    Block &bodyBlock = funcOp.getBody().front();
    
    // Find init and finalize calls to remove them
    ::cir::CallOp initCall = nullptr;
    ::cir::CallOp finalizeCall = nullptr;
    
    for (auto &op : llvm::make_early_inc_range(bodyBlock)) {
      if (auto callOp = dyn_cast<::cir::CallOp>(&op)) {
        auto callee = callOp.getCallee();
        if (callee) {
          if (callee.value() == "shmem_init")
            initCall = callOp;
          else if (callee.value() == "shmem_finalize")
            finalizeCall = callOp;
        }
      }
    }

    if (!initCall || !finalizeCall)
      return failure();

    // Create the region op at the location of the init call
    rewriter.setInsertionPoint(initCall);
    auto regionOp = rewriter.create<openshmem::Region>(initCall.getLoc());
    Block *regionBlock = rewriter.createBlock(&regionOp.getBody());

    // Move all operations between init and finalize into the region
    bool insideRegion = false;
    SmallVector<Operation *> opsToMove;
    
    for (auto &op : bodyBlock) {
      if (&op == initCall.getOperation()) {
        insideRegion = true;
        continue; // Skip the init call itself
      }
      if (&op == finalizeCall.getOperation()) {
        insideRegion = false;
        break; // Stop at finalize
      }
      if (insideRegion) {
        opsToMove.push_back(&op);
      }
    }

    // Move operations into the region
    for (Operation *op : opsToMove) {
      op->moveBefore(regionBlock, regionBlock->end());
    }

    // Add yield terminator to the region
    rewriter.setInsertionPointToEnd(regionBlock);
    rewriter.create<openshmem::YieldOp>(finalizeCall.getLoc());

    // Erase the init and finalize calls
    rewriter.eraseOp(initCall);
    rewriter.eraseOp(finalizeCall);

    return success();
  }
};

/// Pattern to convert shmem_init() calls to openshmem.init
/// (This is now only used if not wrapped in a region)
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
/// (This is now only used if not wrapped in a region)
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

    // Get the original return type from the CIR call
    Type resultType = callOp.getResultTypes()[0];
    
    // Create openshmem.my_pe operation with the same CIR type
    auto myPeOp = rewriter.create<openshmem::MyPeOp>(callOp.getLoc(), resultType);

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

    // Get the original return type from the CIR call
    Type resultType = callOp.getResultTypes()[0];
    
    // Create openshmem.n_pes operation with the same CIR type
    auto nPesOp = rewriter.create<openshmem::NPesOp>(callOp.getLoc(), resultType);

    rewriter.replaceOp(callOp, nPesOp.getResult());
    return success();
  }
};

// Pattern registration for setup/query patterns
void populateCIRToOpenSHMEMSetupPatterns(RewritePatternSet &patterns) {
  // Add the region wrapping pattern first (higher benefit)
  patterns.add<WrapFunctionInRegionPattern>(patterns.getContext(), 
                                            /*benefit=*/2);
  // Add fallback patterns for individual init/finalize calls
  patterns.add<ConvertShmemInitPattern, ConvertShmemFinalizePattern,
               ConvertShmemMyPePattern, ConvertShmemNPesPattern>(
      patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

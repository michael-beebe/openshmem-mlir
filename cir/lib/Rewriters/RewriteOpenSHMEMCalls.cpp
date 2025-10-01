//===- RewriteOpenSHMEMCalls.cpp - CIR to OpenSHMEM Rewriters -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains rewrite patterns for converting CIR OpenSHMEM calls
// to OpenSHMEM dialect operations.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "OpenSHMEMCIR/OpenSHMEMCIR.h"

#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::openshmem;

namespace mlir {
namespace openshmem {
namespace cir {

//===----------------------------------------------------------------------===//
// OpenSHMEM API Call Conversion Patterns
//===----------------------------------------------------------------------===//

/// Pattern to convert shmem_init() calls to openshmem.init
struct ConvertShmemInitPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (callOp.getCallee() != "shmem_init")
      return failure();

    // Create openshmem.init operation
    rewriter.replaceOpWithNewOp<openshmem::InitOp>(callOp);
    return success();
  }
};

/// Pattern to convert shmem_finalize() calls to openshmem.finalize
struct ConvertShmemFinalizePattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (callOp.getCallee() != "shmem_finalize")
      return failure();

    // Create openshmem.finalize operation
    rewriter.replaceOpWithNewOp<openshmem::FinalizeOp>(callOp);
    return success();
  }
};

/// Pattern to convert shmem_my_pe() calls to openshmem.my_pe
struct ConvertShmemMyPePattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (callOp.getCallee() != "shmem_my_pe")
      return failure();

    // Create openshmem.my_pe operation
    Type i32Type = rewriter.getI32Type();
    auto myPeOp = rewriter.create<openshmem::MyPeOp>(
        callOp.getLoc(), i32Type);
    
    rewriter.replaceOp(callOp, myPeOp.getResult());
    return success();
  }
};

/// Pattern to convert shmem_n_pes() calls to openshmem.n_pes  
struct ConvertShmemNPesPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (callOp.getCallee() != "shmem_n_pes")
      return failure();

    // Create openshmem.n_pes operation
    Type i32Type = rewriter.getI32Type();
    auto nPesOp = rewriter.create<openshmem::NPesOp>(
        callOp.getLoc(), i32Type);
    
    rewriter.replaceOp(callOp, nPesOp.getResult());
    return success();
  }
};

/// Pattern to convert shmem_malloc() calls to openshmem.malloc
struct ConvertShmemMallocPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (callOp.getCallee() != "shmem_malloc")
      return failure();

    if (callOp.getOperands().size() != 1)
      return failure();

    Value sizeArg = callOp.getOperand(0);
    
    // TODO: Create proper symmetric memory type
    // For now, use a placeholder memref type
    auto elementType = rewriter.getI8Type();
    auto memRefType = MemRefType::get({ShapedType::kDynamic}, elementType);
    
    auto mallocOp = rewriter.create<openshmem::MallocOp>(
        callOp.getLoc(), memRefType, sizeArg);
    
    rewriter.replaceOp(callOp, mallocOp.getResult());
    return success();
  }
};

/// Pattern to convert shmem_free() calls to openshmem.free
struct ConvertShmemFreePattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (callOp.getCallee() != "shmem_free")
      return failure();

    if (callOp.getOperands().size() != 1)
      return failure();

    Value ptrArg = callOp.getOperand(0);
    
    rewriter.replaceOpWithNewOp<openshmem::FreeOp>(callOp, ptrArg);
    return success();
  }
};

/// Pattern to convert shmem_barrier_all() calls to openshmem.barrier_all
struct ConvertShmemBarrierAllPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (callOp.getCallee() != "shmem_barrier_all")
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::BarrierAllOp>(callOp);
    return success();
  }
};

/// Pattern to convert shmem_quiet() calls to openshmem.quiet
struct ConvertShmemQuietPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (callOp.getCallee() != "shmem_quiet")
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::QuietOp>(callOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern Registration
//===----------------------------------------------------------------------===//

void populateCIRToOpenSHMEMConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<
      ConvertShmemInitPattern,
      ConvertShmemFinalizePattern,
      ConvertShmemMyPePattern,
      ConvertShmemNPesPattern,
      ConvertShmemMallocPattern,
      ConvertShmemFreePattern,
      ConvertShmemBarrierAllPattern,
      ConvertShmemQuietPattern
  >(patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

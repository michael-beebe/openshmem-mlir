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

//===----------------------------------------------------------------------===//
// OpenSHMEM API Call Conversion Patterns
//===----------------------------------------------------------------------===//

// TODO: some of these opps can be squashed into one pattern with a map of
// function names to ops

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

/// Pattern to convert shmem_malloc() calls to openshmem.malloc
struct ConvertShmemMallocPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_malloc")
      return failure();

    if (callOp.getArgOperands().size() != 1)
      return failure();

    Value sizeArg = callOp.getArgOperands()[0];

    // Convert size argument to index type using unrealized conversion cast
    auto indexType = rewriter.getIndexType();
    auto convertedSize = rewriter.create<UnrealizedConversionCastOp>(
        callOp.getLoc(), indexType, sizeArg).getResult(0);

    // Create symmetric memory space attribute
    auto symmetricMemSpace =
        openshmem::SymmetricMemorySpaceAttr::get(rewriter.getContext());

    // Create memref type with symmetric memory space
    auto elementType = rewriter.getI8Type();
    auto memRefType =
        MemRefType::get({ShapedType::kDynamic}, elementType,
                        MemRefLayoutAttrInterface{}, symmetricMemSpace);

    auto mallocOp = rewriter.create<openshmem::MallocOp>(
        callOp.getLoc(), memRefType, convertedSize);

    // Cast the result back to a CIR pointer type to match the original return type
    auto resultCast = rewriter.create<UnrealizedConversionCastOp>(
        callOp.getLoc(), callOp.getResultTypes(), mallocOp.getResult());

    rewriter.replaceOp(callOp, resultCast.getResult(0));
    return success();
  }
};

/// Pattern to convert shmem_free() calls to openshmem.free
struct ConvertShmemFreePattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_free")
      return failure();

    if (callOp.getArgOperands().size() != 1)
      return failure();

    Value ptrArg = callOp.getArgOperands()[0];

    // Create symmetric memory space attribute for the type conversion
    auto symmetricMemSpace =
        openshmem::SymmetricMemorySpaceAttr::get(rewriter.getContext());

    // Create memref type with symmetric memory space for the conversion
    auto elementType = rewriter.getI8Type();
    auto memRefType =
        MemRefType::get({ShapedType::kDynamic}, elementType,
                        MemRefLayoutAttrInterface{}, symmetricMemSpace);

    // Cast the CIR pointer to memref type for the free operation
    auto convertedPtr = rewriter.create<UnrealizedConversionCastOp>(
        callOp.getLoc(), memRefType, ptrArg).getResult(0);

    rewriter.replaceOpWithNewOp<openshmem::FreeOp>(callOp, convertedPtr);
    return success();
  }
};

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

//===----------------------------------------------------------------------===//
// Pattern Registration
//===----------------------------------------------------------------------===//

void populateCIRToOpenSHMEMConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertShmemInitPattern, ConvertShmemFinalizePattern,
               ConvertShmemMyPePattern, ConvertShmemNPesPattern,
               ConvertShmemMallocPattern, ConvertShmemFreePattern,
               ConvertShmemBarrierAllPattern, ConvertShmemQuietPattern>(
      patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

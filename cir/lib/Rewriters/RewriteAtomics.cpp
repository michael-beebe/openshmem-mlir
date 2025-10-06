//===- RewriteAtomics.cpp - CIR to OpenSHMEM Atomic Patterns -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion patterns from CIR call operations to
// OpenSHMEM atomic dialect operations.
//
//===----------------------------------------------------------------------===//

#include "OpenSHMEMCIR/Passes.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace cir;

namespace mlir {
namespace openshmem {
namespace cir {

// Helper function to create symmetric memref type
static MemRefType createSymmetricMemRefType(MLIRContext *ctx) {
  auto symmetricMemSpace = openshmem::SymmetricMemorySpaceAttr::get(ctx);
  auto elementType = IntegerType::get(ctx, 8);
  return MemRefType::get({ShapedType::kDynamic}, elementType,
                         MemRefLayoutAttrInterface{}, symmetricMemSpace);
}

// Helper functions for type conversion
static Value convertPtrToMemRef(OpBuilder &builder, Location loc, Value ptr) {
  auto memRefType = createSymmetricMemRefType(builder.getContext());
  return builder.create<UnrealizedConversionCastOp>(loc, memRefType, ptr)
      .getResult(0);
}

static Value convertToI32(OpBuilder &builder, Location loc, Value value) {
  return builder.create<UnrealizedConversionCastOp>(loc, builder.getI32Type(),
                                                     value)
      .getResult(0);
}

static Value convertToCtx(OpBuilder &builder, Location loc, Value val) {
  auto ctxType = openshmem::CtxType::get(builder.getContext());
  return builder.create<UnrealizedConversionCastOp>(loc, ctxType, val)
      .getResult(0);
}

static Value convertToAnyType(OpBuilder &builder, Location loc, Value value,
                               Type targetType) {
  return builder.create<UnrealizedConversionCastOp>(loc, targetType, value)
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// Basic Atomic Operations (Non-Context)
//===----------------------------------------------------------------------===//

// shmem_atomic_fetch
struct ConvertAtomicFetchPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_fetch")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 2)
      return failure();

    auto source = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto pe = convertToI32(rewriter, op.getLoc(), operands[1]);

    auto newOp = rewriter.create<mlir::openshmem::AtomicFetchOp>(
        op.getLoc(), rewriter.getI32Type(), source, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_atomic_set
struct ConvertAtomicSetPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_set")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[1],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicSetOp>(op, dest, value,
                                                               pe);
    return success();
  }
};

// shmem_atomic_compare_swap
struct ConvertAtomicCompareSwapPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_atomic_compare_swap")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto cond = convertToAnyType(rewriter, op.getLoc(), operands[1],
                                  rewriter.getI32Type());
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    auto newOp = rewriter.create<mlir::openshmem::AtomicCompareSwapOp>(
        op.getLoc(), rewriter.getI32Type(), dest, cond, value, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_atomic_swap
struct ConvertAtomicSwapPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_swap")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[1],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    auto newOp = rewriter.create<mlir::openshmem::AtomicSwapOp>(
        op.getLoc(), rewriter.getI32Type(), dest, value, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_atomic_fetch_inc
struct ConvertAtomicFetchIncPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_fetch_inc")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 2)
      return failure();

    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto pe = convertToI32(rewriter, op.getLoc(), operands[1]);

    auto newOp = rewriter.create<mlir::openshmem::AtomicFetchIncOp>(
        op.getLoc(), rewriter.getI32Type(), dest, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_atomic_inc
struct ConvertAtomicIncPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_inc")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 2)
      return failure();

    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto pe = convertToI32(rewriter, op.getLoc(), operands[1]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicIncOp>(op, dest, pe);
    return success();
  }
};

// shmem_atomic_fetch_add
struct ConvertAtomicFetchAddPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_fetch_add")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[1],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    auto newOp = rewriter.create<mlir::openshmem::AtomicFetchAddOp>(
        op.getLoc(), rewriter.getI32Type(), dest, value, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_atomic_add
struct ConvertAtomicAddPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_add")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[1],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicAddOp>(op, dest, value,
                                                               pe);
    return success();
  }
};

// shmem_atomic_fetch_and
struct ConvertAtomicFetchAndPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_fetch_and")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[1],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    auto newOp = rewriter.create<mlir::openshmem::AtomicFetchAndOp>(
        op.getLoc(), rewriter.getI32Type(), dest, value, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_atomic_fetch_or
struct ConvertAtomicFetchOrPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_fetch_or")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[1],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    auto newOp = rewriter.create<mlir::openshmem::AtomicFetchOrOp>(
        op.getLoc(), rewriter.getI32Type(), dest, value, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_atomic_or
struct ConvertAtomicOrPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_or")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[1],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicOrOp>(op, dest, value,
                                                              pe);
    return success();
  }
};

// shmem_atomic_fetch_xor
struct ConvertAtomicFetchXorPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_fetch_xor")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[1],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    auto newOp = rewriter.create<mlir::openshmem::AtomicFetchXorOp>(
        op.getLoc(), rewriter.getI32Type(), dest, value, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_atomic_xor
struct ConvertAtomicXorPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_xor")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[1],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicXorOp>(op, dest, value,
                                                               pe);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Context-Aware Atomic Operations
//===----------------------------------------------------------------------===//

// shmem_ctx_atomic_fetch
struct ConvertCtxAtomicFetchPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_ctx_atomic_fetch")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto source = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    auto newOp = rewriter.create<mlir::openshmem::CtxAtomicFetchOp>(
        op.getLoc(), rewriter.getI32Type(), ctx, source, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_ctx_atomic_set
struct ConvertCtxAtomicSetPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_ctx_atomic_set")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicSetOp>(op, ctx, dest,
                                                                  value, pe);
    return success();
  }
};

// shmem_ctx_atomic_compare_swap
struct ConvertCtxAtomicCompareSwapPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_compare_swap")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto cond = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                  rewriter.getI32Type());
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[3],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[4]);

    auto newOp = rewriter.create<mlir::openshmem::CtxAtomicCompareSwapOp>(
        op.getLoc(), rewriter.getI32Type(), ctx, dest, cond, value, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_ctx_atomic_swap
struct ConvertCtxAtomicSwapPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_ctx_atomic_swap")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    auto newOp = rewriter.create<mlir::openshmem::CtxAtomicSwapOp>(
        op.getLoc(), rewriter.getI32Type(), ctx, dest, value, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_ctx_atomic_fetch_inc
struct ConvertCtxAtomicFetchIncPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_fetch_inc")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    auto newOp = rewriter.create<mlir::openshmem::CtxAtomicFetchIncOp>(
        op.getLoc(), rewriter.getI32Type(), ctx, dest, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_ctx_atomic_inc
struct ConvertCtxAtomicIncPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_ctx_atomic_inc")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicIncOp>(op, ctx, dest,
                                                                  pe);
    return success();
  }
};

// shmem_ctx_atomic_fetch_add
struct ConvertCtxAtomicFetchAddPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_fetch_add")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    auto newOp = rewriter.create<mlir::openshmem::CtxAtomicFetchAddOp>(
        op.getLoc(), rewriter.getI32Type(), ctx, dest, value, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_ctx_atomic_add
struct ConvertCtxAtomicAddPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_ctx_atomic_add")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicAddOp>(op, ctx, dest,
                                                                  value, pe);
    return success();
  }
};

// shmem_ctx_atomic_fetch_and
struct ConvertCtxAtomicFetchAndPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_fetch_and")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    auto newOp = rewriter.create<mlir::openshmem::CtxAtomicFetchAndOp>(
        op.getLoc(), rewriter.getI32Type(), ctx, dest, value, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_ctx_atomic_fetch_or
struct ConvertCtxAtomicFetchOrPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_fetch_or")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    auto newOp = rewriter.create<mlir::openshmem::CtxAtomicFetchOrOp>(
        op.getLoc(), rewriter.getI32Type(), ctx, dest, value, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_ctx_atomic_or
struct ConvertCtxAtomicOrPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_ctx_atomic_or")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicOrOp>(op, ctx, dest,
                                                                 value, pe);
    return success();
  }
};

// shmem_ctx_atomic_fetch_xor
struct ConvertCtxAtomicFetchXorPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_fetch_xor")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    auto newOp = rewriter.create<mlir::openshmem::CtxAtomicFetchXorOp>(
        op.getLoc(), rewriter.getI32Type(), ctx, dest, value, pe);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// shmem_ctx_atomic_xor
struct ConvertCtxAtomicXorPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_ctx_atomic_xor")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicXorOp>(op, ctx, dest,
                                                                  value, pe);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Non-Blocking Atomic Operations  
//===----------------------------------------------------------------------===//

// Note: NBI (non-blocking) atomic operations follow the same pattern as
// blocking ones but use _nbi suffix operations. Implementing the most common ones:

// shmem_atomic_fetch_nbi
struct ConvertAtomicFetchNbiPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_fetch_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto source = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicFetchNbiOp>(
        op, fetch_ptr, source, pe);
    return success();
  }
};

// shmem_ctx_atomic_fetch_nbi
struct ConvertCtxAtomicFetchNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_fetch_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto source = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicFetchNbiOp>(
        op, ctx, fetch_ptr, source, pe);
    return success();
  }
};

// shmem_atomic_compare_swap_nbi
struct ConvertAtomicCompareSwapNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_atomic_compare_swap_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto cond = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                  rewriter.getI32Type());
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[3],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[4]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicCompareSwapNbiOp>(
        op, fetch_ptr, dest, cond, value, pe);
    return success();
  }
};

// shmem_ctx_atomic_compare_swap_nbi
struct ConvertCtxAtomicCompareSwapNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_compare_swap_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 6)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto cond = convertToAnyType(rewriter, op.getLoc(), operands[3],
                                  rewriter.getI32Type());
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[4],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[5]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicCompareSwapNbiOp>(
        op, ctx, fetch_ptr, dest, cond, value, pe);
    return success();
  }
};

// shmem_atomic_swap_nbi
struct ConvertAtomicSwapNbiPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_atomic_swap_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicSwapNbiOp>(
        op, fetch_ptr, dest, value, pe);
    return success();
  }
};

// shmem_ctx_atomic_swap_nbi
struct ConvertCtxAtomicSwapNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_swap_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[3],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[4]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicSwapNbiOp>(
        op, ctx, fetch_ptr, dest, value, pe);
    return success();
  }
};

// shmem_atomic_fetch_inc_nbi
struct ConvertAtomicFetchIncNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_atomic_fetch_inc_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto pe = convertToI32(rewriter, op.getLoc(), operands[2]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicFetchIncNbiOp>(
        op, fetch_ptr, dest, pe);
    return success();
  }
};

// shmem_ctx_atomic_fetch_inc_nbi
struct ConvertCtxAtomicFetchIncNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_fetch_inc_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicFetchIncNbiOp>(
        op, ctx, fetch_ptr, dest, pe);
    return success();
  }
};

// shmem_atomic_fetch_add_nbi
struct ConvertAtomicFetchAddNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_atomic_fetch_add_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicFetchAddNbiOp>(
        op, fetch_ptr, dest, value, pe);
    return success();
  }
};

// shmem_ctx_atomic_fetch_add_nbi
struct ConvertCtxAtomicFetchAddNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_fetch_add_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[3],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[4]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicFetchAddNbiOp>(
        op, ctx, fetch_ptr, dest, value, pe);
    return success();
  }
};

// shmem_atomic_fetch_and_nbi
struct ConvertAtomicFetchAndNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_atomic_fetch_and_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicFetchAndNbiOp>(
        op, fetch_ptr, dest, value, pe);
    return success();
  }
};

// shmem_ctx_atomic_fetch_and_nbi
struct ConvertCtxAtomicFetchAndNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_fetch_and_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[3],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[4]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicFetchAndNbiOp>(
        op, ctx, fetch_ptr, dest, value, pe);
    return success();
  }
};

// shmem_atomic_fetch_or_nbi
struct ConvertAtomicFetchOrNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_atomic_fetch_or_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicFetchOrNbiOp>(
        op, fetch_ptr, dest, value, pe);
    return success();
  }
};

// shmem_ctx_atomic_fetch_or_nbi
struct ConvertCtxAtomicFetchOrNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_fetch_or_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[3],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[4]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicFetchOrNbiOp>(
        op, ctx, fetch_ptr, dest, value, pe);
    return success();
  }
};

// shmem_atomic_fetch_xor_nbi
struct ConvertAtomicFetchXorNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_atomic_fetch_xor_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[3]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::AtomicFetchXorNbiOp>(
        op, fetch_ptr, dest, value, pe);
    return success();
  }
};

// shmem_ctx_atomic_fetch_xor_nbi
struct ConvertCtxAtomicFetchXorNbiPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_ctx_atomic_fetch_xor_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ctx = convertToCtx(rewriter, op.getLoc(), operands[0]);
    auto fetch_ptr = convertPtrToMemRef(rewriter, op.getLoc(), operands[1]);
    auto dest = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto value = convertToAnyType(rewriter, op.getLoc(), operands[3],
                                   rewriter.getI32Type());
    auto pe = convertToI32(rewriter, op.getLoc(), operands[4]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::CtxAtomicFetchXorNbiOp>(
        op, ctx, fetch_ptr, dest, value, pe);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateCIRToOpenSHMEMAtomicPatterns(RewritePatternSet &patterns) {
  // Basic atomic operations
  patterns.add<ConvertAtomicFetchPattern>(patterns.getContext());
  patterns.add<ConvertAtomicSetPattern>(patterns.getContext());
  patterns.add<ConvertAtomicCompareSwapPattern>(patterns.getContext());
  patterns.add<ConvertAtomicSwapPattern>(patterns.getContext());
  patterns.add<ConvertAtomicFetchIncPattern>(patterns.getContext());
  patterns.add<ConvertAtomicIncPattern>(patterns.getContext());
  patterns.add<ConvertAtomicFetchAddPattern>(patterns.getContext());
  patterns.add<ConvertAtomicAddPattern>(patterns.getContext());
  patterns.add<ConvertAtomicFetchAndPattern>(patterns.getContext());
  patterns.add<ConvertAtomicFetchOrPattern>(patterns.getContext());
  patterns.add<ConvertAtomicOrPattern>(patterns.getContext());
  patterns.add<ConvertAtomicFetchXorPattern>(patterns.getContext());
  patterns.add<ConvertAtomicXorPattern>(patterns.getContext());

  // Context-aware atomic operations
  patterns.add<ConvertCtxAtomicFetchPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicSetPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicCompareSwapPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicSwapPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicFetchIncPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicIncPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicFetchAddPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicAddPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicFetchAndPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicFetchOrPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicOrPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicFetchXorPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicXorPattern>(patterns.getContext());

  // Non-blocking atomic operations
  patterns.add<ConvertAtomicFetchNbiPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicFetchNbiPattern>(patterns.getContext());
  patterns.add<ConvertAtomicCompareSwapNbiPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicCompareSwapNbiPattern>(patterns.getContext());
  patterns.add<ConvertAtomicSwapNbiPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicSwapNbiPattern>(patterns.getContext());
  patterns.add<ConvertAtomicFetchIncNbiPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicFetchIncNbiPattern>(patterns.getContext());
  patterns.add<ConvertAtomicFetchAddNbiPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicFetchAddNbiPattern>(patterns.getContext());
  patterns.add<ConvertAtomicFetchAndNbiPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicFetchAndNbiPattern>(patterns.getContext());
  patterns.add<ConvertAtomicFetchOrNbiPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicFetchOrNbiPattern>(patterns.getContext());
  patterns.add<ConvertAtomicFetchXorNbiPattern>(patterns.getContext());
  patterns.add<ConvertCtxAtomicFetchXorNbiPattern>(patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

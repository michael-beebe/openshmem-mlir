//===- RewritePt2ptSync.cpp - CIR to OpenSHMEM Pt2pt Sync Patterns -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

static Value convertToIndex(OpBuilder &builder, Location loc, Value value) {
  return builder
      .create<UnrealizedConversionCastOp>(loc, builder.getIndexType(), value)
      .getResult(0);
}

static Value convertToI32(OpBuilder &builder, Location loc, Value value) {
  return builder
      .create<UnrealizedConversionCastOp>(loc, builder.getI32Type(), value)
      .getResult(0);
}

static Value convertToI64(OpBuilder &builder, Location loc, Value value) {
  return builder
      .create<UnrealizedConversionCastOp>(loc, builder.getI64Type(), value)
      .getResult(0);
}

static Value convertToAnyType(OpBuilder &builder, Location loc, Value value,
                              Type targetType) {
  return builder.create<UnrealizedConversionCastOp>(loc, targetType, value)
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// Wait Operations
//===----------------------------------------------------------------------===//

// Pattern for shmem_wait_until
struct ConvertWaitUntilPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_wait_until")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto ivar = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[1]);
    auto cmpValue = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                     rewriter.getI32Type());

    rewriter.replaceOpWithNewOp<mlir::openshmem::WaitUntilOp>(op, ivar, cmp,
                                                              cmpValue);
    return success();
  }
};

// Pattern for shmem_wait_until_all
struct ConvertWaitUntilAllPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_wait_until_all")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ivars = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto nelems = convertToIndex(rewriter, op.getLoc(), operands[1]);
    auto status = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[3]);
    auto cmpValue = convertToAnyType(rewriter, op.getLoc(), operands[4],
                                     rewriter.getI32Type());

    rewriter.replaceOpWithNewOp<mlir::openshmem::WaitUntilAllOp>(
        op, ivars, nelems, status, cmp, cmpValue);
    return success();
  }
};

// Pattern for shmem_wait_until_any
struct ConvertWaitUntilAnyPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_wait_until_any")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ivars = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto nelems = convertToIndex(rewriter, op.getLoc(), operands[1]);
    auto status = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[3]);
    auto cmpValue = convertToAnyType(rewriter, op.getLoc(), operands[4],
                                     rewriter.getI32Type());

    rewriter.replaceOpWithNewOp<mlir::openshmem::WaitUntilAnyOp>(
        op, ivars, nelems, status, cmp, cmpValue);
    return success();
  }
};

// Pattern for shmem_wait_until_some
struct ConvertWaitUntilSomePattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_wait_until_some")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 6)
      return failure();

    auto ivars = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto nelems = convertToIndex(rewriter, op.getLoc(), operands[1]);
    auto indices = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto status = convertPtrToMemRef(rewriter, op.getLoc(), operands[3]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[4]);
    auto cmpValue = convertToAnyType(rewriter, op.getLoc(), operands[5],
                                     rewriter.getI32Type());

    auto newOp = rewriter.create<mlir::openshmem::WaitUntilSomeOp>(
        op.getLoc(), rewriter.getIndexType(), ivars, nelems, indices, status,
        cmp, cmpValue);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Wait Vector Operations
//===----------------------------------------------------------------------===//

// Pattern for shmem_wait_until_all_vector
struct ConvertWaitUntilAllVectorPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_wait_until_all_vector")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ivars = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto nelems = convertToIndex(rewriter, op.getLoc(), operands[1]);
    auto status = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[3]);
    auto cmpValues = convertPtrToMemRef(rewriter, op.getLoc(), operands[4]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::WaitUntilAllVectorOp>(
        op, ivars, nelems, status, cmp, cmpValues);
    return success();
  }
};

// Pattern for shmem_wait_until_any_vector
struct ConvertWaitUntilAnyVectorPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_wait_until_any_vector")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ivars = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto nelems = convertToIndex(rewriter, op.getLoc(), operands[1]);
    auto status = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[3]);
    auto cmpValues = convertPtrToMemRef(rewriter, op.getLoc(), operands[4]);

    rewriter.replaceOpWithNewOp<mlir::openshmem::WaitUntilAnyVectorOp>(
        op, ivars, nelems, status, cmp, cmpValues);
    return success();
  }
};

// Pattern for shmem_wait_until_some_vector
struct ConvertWaitUntilSomeVectorPattern
    : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() ||
        op.getCallee().value() != "shmem_wait_until_some_vector")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 6)
      return failure();

    auto ivars = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto nelems = convertToIndex(rewriter, op.getLoc(), operands[1]);
    auto indices = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto status = convertPtrToMemRef(rewriter, op.getLoc(), operands[3]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[4]);
    auto cmpValues = convertPtrToMemRef(rewriter, op.getLoc(), operands[5]);

    auto newOp = rewriter.create<mlir::openshmem::WaitUntilSomeVectorOp>(
        op.getLoc(), rewriter.getIndexType(), ivars, nelems, indices, status,
        cmp, cmpValues);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Test Operations
//===----------------------------------------------------------------------===//

// Pattern for shmem_test
struct ConvertTestPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_test")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto ivar = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[1]);
    auto cmpValue = convertToAnyType(rewriter, op.getLoc(), operands[2],
                                     rewriter.getI32Type());

    auto newOp = rewriter.create<mlir::openshmem::TestOp>(
        op.getLoc(), rewriter.getI32Type(), ivar, cmp, cmpValue);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// Pattern for shmem_test_all
struct ConvertTestAllPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_test_all")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ivars = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto nelems = convertToIndex(rewriter, op.getLoc(), operands[1]);
    auto status = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[3]);
    auto cmpValue = convertToAnyType(rewriter, op.getLoc(), operands[4],
                                     rewriter.getI32Type());

    auto newOp = rewriter.create<mlir::openshmem::TestAllOp>(
        op.getLoc(), rewriter.getI32Type(), ivars, nelems, status, cmp,
        cmpValue);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// Pattern for shmem_test_any
struct ConvertTestAnyPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_test_any")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ivars = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto nelems = convertToIndex(rewriter, op.getLoc(), operands[1]);
    auto status = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[3]);
    auto cmpValue = convertToAnyType(rewriter, op.getLoc(), operands[4],
                                     rewriter.getI32Type());

    auto newOp = rewriter.create<mlir::openshmem::TestAnyOp>(
        op.getLoc(), rewriter.getIndexType(), ivars, nelems, status, cmp,
        cmpValue);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// Pattern for shmem_test_some
struct ConvertTestSomePattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_test_some")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 6)
      return failure();

    auto ivars = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto nelems = convertToIndex(rewriter, op.getLoc(), operands[1]);
    auto indices = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto status = convertPtrToMemRef(rewriter, op.getLoc(), operands[3]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[4]);
    auto cmpValue = convertToAnyType(rewriter, op.getLoc(), operands[5],
                                     rewriter.getI32Type());

    auto newOp = rewriter.create<mlir::openshmem::TestSomeOp>(
        op.getLoc(), rewriter.getIndexType(), ivars, nelems, indices, status,
        cmp, cmpValue);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Test Vector Operations
//===----------------------------------------------------------------------===//

// Pattern for shmem_test_all_vector
struct ConvertTestAllVectorPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_test_all_vector")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ivars = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto nelems = convertToIndex(rewriter, op.getLoc(), operands[1]);
    auto status = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[3]);
    auto cmpValues = convertPtrToMemRef(rewriter, op.getLoc(), operands[4]);

    auto newOp = rewriter.create<mlir::openshmem::TestAllVectorOp>(
        op.getLoc(), rewriter.getI32Type(), ivars, nelems, status, cmp,
        cmpValues);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// Pattern for shmem_test_any_vector
struct ConvertTestAnyVectorPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_test_any_vector")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ivars = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto nelems = convertToIndex(rewriter, op.getLoc(), operands[1]);
    auto status = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[3]);
    auto cmpValues = convertPtrToMemRef(rewriter, op.getLoc(), operands[4]);

    auto newOp = rewriter.create<mlir::openshmem::TestAnyVectorOp>(
        op.getLoc(), rewriter.getIndexType(), ivars, nelems, status, cmp,
        cmpValues);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// Pattern for shmem_test_some_vector
struct ConvertTestSomeVectorPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_test_some_vector")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 6)
      return failure();

    auto ivars = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto nelems = convertToIndex(rewriter, op.getLoc(), operands[1]);
    auto indices = convertPtrToMemRef(rewriter, op.getLoc(), operands[2]);
    auto status = convertPtrToMemRef(rewriter, op.getLoc(), operands[3]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[4]);
    auto cmpValues = convertPtrToMemRef(rewriter, op.getLoc(), operands[5]);

    auto newOp = rewriter.create<mlir::openshmem::TestSomeVectorOp>(
        op.getLoc(), rewriter.getIndexType(), ivars, nelems, indices, status,
        cmp, cmpValues);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Signal Operations
//===----------------------------------------------------------------------===//

// Pattern for shmem_signal_wait_until
struct ConvertSignalWaitUntilPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCallee().value() != "shmem_signal_wait_until")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 3)
      return failure();

    auto sigAddr = convertPtrToMemRef(rewriter, op.getLoc(), operands[0]);
    auto cmp = convertToI32(rewriter, op.getLoc(), operands[1]);
    auto cmpValue = convertToI64(rewriter, op.getLoc(), operands[2]);

    auto newOp = rewriter.create<mlir::openshmem::SignalWaitUntilOp>(
        op.getLoc(), rewriter.getI64Type(), sigAddr, cmp, cmpValue);

    if (!op.getResult().use_empty()) {
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateCIRToOpenSHMEMPt2ptSyncPatterns(RewritePatternSet &patterns) {
  // Wait operations
  patterns.add<ConvertWaitUntilPattern>(patterns.getContext());
  patterns.add<ConvertWaitUntilAllPattern>(patterns.getContext());
  patterns.add<ConvertWaitUntilAnyPattern>(patterns.getContext());
  patterns.add<ConvertWaitUntilSomePattern>(patterns.getContext());

  // Wait vector operations
  patterns.add<ConvertWaitUntilAllVectorPattern>(patterns.getContext());
  patterns.add<ConvertWaitUntilAnyVectorPattern>(patterns.getContext());
  patterns.add<ConvertWaitUntilSomeVectorPattern>(patterns.getContext());

  // Test operations
  patterns.add<ConvertTestPattern>(patterns.getContext());
  patterns.add<ConvertTestAllPattern>(patterns.getContext());
  patterns.add<ConvertTestAnyPattern>(patterns.getContext());
  patterns.add<ConvertTestSomePattern>(patterns.getContext());

  // Test vector operations
  patterns.add<ConvertTestAllVectorPattern>(patterns.getContext());
  patterns.add<ConvertTestAnyVectorPattern>(patterns.getContext());
  patterns.add<ConvertTestSomeVectorPattern>(patterns.getContext());

  // Signal operations
  patterns.add<ConvertSignalWaitUntilPattern>(patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

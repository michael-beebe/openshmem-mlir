//===- BridgeOpsToLLVM.cpp - Bridge helper conversion patterns -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BridgeOpsToLLVM.h"
#include "OpenSHMEMConversionUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using namespace mlir::openshmem;

namespace {

struct WrapSymmetricPtrOpLowering
    : public ConvertOpToLLVMPattern<openshmem::WrapSymmetricPtrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::WrapSymmetricPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Ensure operand is converted to the target (LLVM) type if needed.
    Value src = adaptor.getPtr();
    Type tgtTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!tgtTy)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    if (src.getType() != tgtTy) {
      // Materialize an UnrealizedConversionCastOp to produce a value with the
      // converted type. The rest of the pipeline will remove/resolve these.
      auto cast = rewriter.create<UnrealizedConversionCastOp>(
          op.getLoc(), TypeRange(tgtTy), ValueRange(src));
      rewriter.replaceOp(op, cast.getResults());
      return success();
    }

    rewriter.replaceOp(op, src);
    return success();
  }
};

struct WrapLocalPtrOpLowering
    : public ConvertOpToLLVMPattern<openshmem::WrapLocalPtrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::WrapLocalPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getPtr();
    Type tgtTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!tgtTy)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    if (src.getType() != tgtTy) {
      auto cast = rewriter.create<UnrealizedConversionCastOp>(
          op.getLoc(), TypeRange(tgtTy), ValueRange(src));
      rewriter.replaceOp(op, cast.getResults());
      return success();
    }

    rewriter.replaceOp(op, src);
    return success();
  }
};

struct WrapCtxOpLowering : public ConvertOpToLLVMPattern<openshmem::WrapCtxOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(openshmem::WrapCtxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const
      override {
    Value src = adaptor.getPtr();
    Type tgtTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!tgtTy)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    if (src.getType() != tgtTy) {
      auto cast = rewriter.create<UnrealizedConversionCastOp>(
          op.getLoc(), TypeRange(tgtTy), ValueRange(src));
      rewriter.replaceOp(op, cast.getResults());
      return success();
    }

    rewriter.replaceOp(op, src);
    return success();
  }
};

// Simplified pre-lowering for wrap_value: unconditionally replace any
// openshmem.wrap_value with an UnrealizedConversionCastOp that converts
// the incoming operand to the wrap's declared result type. This defers
// actual type materialization to the conversion machinery and avoids the
// complexity of handling all CIR type variants here.
struct WrapValuePreLowering : public OpRewritePattern<openshmem::WrapValueOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(openshmem::WrapValueOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value value = op.getValue();
    auto cast = rewriter.create<UnrealizedConversionCastOp>(loc,
                                                            TypeRange(op.getResult().getType()),
                                                            ValueRange(value));
    rewriter.replaceOp(op, cast.getResults());
    return success();
  }
};

// Lower wrap_value that takes a memref and produces a pointer-like CIR type
// by replacing it with an UnrealizedConversionCastOp from the memref to the
// declared result type. This removes the bridge op and defers type
// translation to the later conversion machinery.
struct WrapValueMemRefToPtrLowering
    : public ConvertOpToLLVMPattern<openshmem::WrapValueOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(openshmem::WrapValueOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value val = adaptor.getValue();
    Type valType = val.getType();
    if (!llvm::isa<MemRefType>(valType))
      return rewriter.notifyMatchFailure(op, "not a memref operand");

    Location loc = op.getLoc();
    auto cast = rewriter.create<UnrealizedConversionCastOp>(loc,
                                                            TypeRange(op.getResult().getType()),
                                                            ValueRange(val));
    rewriter.replaceOp(op, cast.getResults());
    return success();
  }
};

// Fallback conversion pattern: if for some reason the conversion pattern
// above doesn't run (e.g., operand types are still CIR types), this
// conversion pattern removes the bridge op by emitting an
// UnrealizedConversionCastOp that converts the operand to the original
// (unconverted) result type. This defers the real conversion to the rest
// of the pipeline.
struct WrapValueOpFallbackLowering
    : public ConvertOpToLLVMPattern<openshmem::WrapValueOp> {
  WrapValueOpFallbackLowering(LLVMTypeConverter &conv)
      : ConvertOpToLLVMPattern<openshmem::WrapValueOp>(conv) {}

  LogicalResult matchAndRewrite(openshmem::WrapValueOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value value = adaptor.getValue();
    auto cast = rewriter.create<UnrealizedConversionCastOp>(loc,
                                                            TypeRange(op.getResult().getType()),
                                                            ValueRange(value));
    rewriter.replaceOp(op, cast.getResults());
    return success();
  }
};

} // namespace

// Final fallback rewrite: unconditionally replace any remaining wrap_value
// ops with an UnrealizedConversionCastOp that converts the operand to the
// wrap's declared result type. This is a last resort to ensure the
// OpenSHMEM->LLVM conversion doesn't fail due to residual bridge ops.
struct WrapValueOpRewriteFallback
    : public OpRewritePattern<openshmem::WrapValueOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(openshmem::WrapValueOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value value = op.getValue();
    auto cast = rewriter.create<UnrealizedConversionCastOp>(loc,
                                                            TypeRange(op.getResult().getType()),
                                                            ValueRange(value));
    rewriter.replaceOp(op, cast.getResults());
    return success();
  }
};

void openshmem::populateBridgeOpsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<WrapSymmetricPtrOpLowering, WrapLocalPtrOpLowering,
               WrapCtxOpLowering, WrapValueMemRefToPtrLowering>(converter);
  // Add pre-lowering patterns for wrap_value so they are removed before the
  // rest of the OpenSHMEM->LLVM conversion runs. Integer/index variants are
  // handled separately from memref->ptr variants.
  patterns.add<WrapValuePreLowering>(patterns.getContext());
  patterns.add<WrapValueOpRewriteFallback>(patterns.getContext());
}

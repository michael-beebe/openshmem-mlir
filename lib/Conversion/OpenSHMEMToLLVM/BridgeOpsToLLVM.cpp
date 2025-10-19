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

using namespace mlir;
using namespace mlir::openshmem;

namespace {

struct WrapSymmetricPtrOpLowering
    : public ConvertOpToLLVMPattern<openshmem::WrapSymmetricPtrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::WrapSymmetricPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getPtr());
    return success();
  }
};

struct WrapLocalPtrOpLowering
    : public ConvertOpToLLVMPattern<openshmem::WrapLocalPtrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::WrapLocalPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getPtr());
    return success();
  }
};

struct WrapCtxOpLowering : public ConvertOpToLLVMPattern<openshmem::WrapCtxOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(openshmem::WrapCtxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const
      override {
    rewriter.replaceOp(op, adaptor.getPtr());
    return success();
  }
};

struct WrapValueOpLowering
    : public ConvertOpToLLVMPattern<openshmem::WrapValueOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::WrapValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value value = adaptor.getValue();
    Type targetType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (!targetType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    Type operandType = value.getType();
    if (operandType == targetType) {
      rewriter.replaceOp(op, value);
      return success();
    }

    if (targetType.isIntOrIndex()) {
      if (isa<LLVM::LLVMPointerType>(operandType)) {
        Value converted =
            rewriter.create<LLVM::PtrToIntOp>(loc, targetType, value);
        rewriter.replaceOp(op, converted);
        return success();
      }

      if (!operandType.isIntOrIndex())
        return rewriter.notifyMatchFailure(op, "expected integer operand");

      Value converted = value;
      unsigned operandWidth = operandType.getIntOrFloatBitWidth();
      unsigned targetWidth = targetType.getIntOrFloatBitWidth();
      if (operandWidth == targetWidth) {
        converted = value;
      } else if (operandWidth > targetWidth) {
        converted = rewriter.create<LLVM::TruncOp>(loc, targetType, value);
      } else if (operandType.isSignedInteger()) {
        converted = rewriter.create<LLVM::SExtOp>(loc, targetType, value);
      } else {
        // Default to zero-extension for signless/unsigned integers to match
        // the semantics of PE arguments used today.
        converted = rewriter.create<LLVM::ZExtOp>(loc, targetType, value);
      }
      rewriter.replaceOp(op, converted);
      return success();
    }

    if (auto targetPtrTy = dyn_cast<LLVM::LLVMPointerType>(targetType)) {
      if (auto operandPtrTy = dyn_cast<LLVM::LLVMPointerType>(operandType)) {
        Value converted = value;
        if (operandPtrTy != targetPtrTy)
          converted =
              rewriter.create<LLVM::BitcastOp>(loc, targetType, value);
        rewriter.replaceOp(op, converted);
        return success();
      }

      if (operandType.isIntOrIndex()) {
        Value converted =
            rewriter.create<LLVM::IntToPtrOp>(loc, targetType, value);
        rewriter.replaceOp(op, converted);
        return success();
      }

      return rewriter.notifyMatchFailure(
          op, "unsupported operand type for pointer wrap");
    }

    return rewriter.notifyMatchFailure(op, "unsupported wrap target type");
  }
};

} // namespace

void openshmem::populateBridgeOpsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<WrapSymmetricPtrOpLowering, WrapLocalPtrOpLowering,
               WrapCtxOpLowering, WrapValueOpLowering>(converter);
}

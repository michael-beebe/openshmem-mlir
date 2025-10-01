//===- SyncOpsToLLVM.cpp - Sync operations conversion patterns -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SyncOpsToLLVM.h"
#include "OpenSHMEMConversionUtils.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::openshmem;

namespace {

//===----------------------------------------------------------------------===//
// BarrierAllOp Lowering
//===----------------------------------------------------------------------===//

struct BarrierAllOpLowering
    : public ConvertOpToLLVMPattern<openshmem::BarrierAllOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::BarrierAllOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // void shmem_barrier_all(void)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_barrier_all", funcType);

    // Replace with function call
    rewriter.create<LLVM::CallOp>(loc, funcDecl, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// QuietOp Lowering
//===----------------------------------------------------------------------===//

struct QuietOpLowering : public ConvertOpToLLVMPattern<openshmem::QuietOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::QuietOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // void shmem_quiet(void)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_quiet", funcType);

    // Replace with function call
    rewriter.create<LLVM::CallOp>(loc, funcDecl, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BarrierOp Lowering
//===----------------------------------------------------------------------===//

struct BarrierOpLowering : public ConvertOpToLLVMPattern<openshmem::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void shmem_barrier(int PE_start, int logPE_stride, int PE_size, long
    // *pSync)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type(),
         ptrType});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_barrier", funcType);

    // The psync argument is already a pointer (memref with symmetric memory
    // space converts to pointer)
    Value psyncPtr = adaptor.getPsync();

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{adaptor.getPeStart(),
                                             adaptor.getLogPeStride(),
                                             adaptor.getPeSize(), psyncPtr});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void openshmem::populateSyncOpsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<BarrierAllOpLowering, QuietOpLowering, BarrierOpLowering>(
      converter);
}

//===- ContextOpsToLLVM.cpp - Context operations conversion patterns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ContextOpsToLLVM.h"
#include "OpenSHMEMConversionUtils.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::openshmem;

namespace {

struct CtxCreateOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxCreateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::CtxCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type i64Type = rewriter.getI64Type();
    Type i32Type = rewriter.getI32Type();

    // int shmem_ctx_create(long options, shmem_ctx_t *ctx)
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, {i64Type, ptrType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_ctx_create", funcType);

    // Allocate space for the context handle
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(1));
    Value ctxPtr = rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType, one);

    // Call the function
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{adaptor.getOptions(), ctxPtr});

    // Load the context handle
    Value ctx = rewriter.create<LLVM::LoadOp>(loc, ptrType, ctxPtr);

    SmallVector<Value> replacements;
    replacements.push_back(ctx);
    replacements.push_back(callOp.getResult());
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

struct TeamCreateCtxOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TeamCreateCtxOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TeamCreateCtxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type i64Type = rewriter.getI64Type();
    Type i32Type = rewriter.getI32Type();

    // int shmem_team_create_ctx(shmem_team_t team, long options, shmem_ctx_t
    // *ctx)
    auto funcType =
        LLVM::LLVMFunctionType::get(i32Type, {ptrType, i64Type, ptrType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_team_create_ctx", funcType);

    // Allocate space for the context handle
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(1));
    Value ctxPtr = rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType, one);

    // Call the function
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), adaptor.getOptions(), ctxPtr});

    // Load the context handle
    Value ctx = rewriter.create<LLVM::LoadOp>(loc, ptrType, ctxPtr);

    SmallVector<Value> replacements;
    replacements.push_back(ctx);
    replacements.push_back(callOp.getResult());
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

struct CtxDestroyOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxDestroyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::CtxDestroyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void shmem_ctx_destroy(shmem_ctx_t ctx)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_ctx_destroy", funcType);

    rewriter.create<LLVM::CallOp>(loc, funcDecl, ValueRange{adaptor.getCtx()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct CtxGetTeamOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxGetTeamOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::CtxGetTeamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type i32Type = rewriter.getI32Type();

    // int shmem_ctx_get_team(shmem_ctx_t ctx, shmem_team_t *team)
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_ctx_get_team", funcType);

    // Allocate space for the team handle
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(1));
    Value teamPtr = rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType, one);

    // Call the function
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{adaptor.getCtx(), teamPtr});

    // Load the team handle
    Value team = rewriter.create<LLVM::LoadOp>(loc, ptrType, teamPtr);

    SmallVector<Value> replacements;
    replacements.push_back(team);
    replacements.push_back(callOp.getResult());
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

} // namespace

void openshmem::populateContextOpsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<CtxCreateOpLowering, TeamCreateCtxOpLowering,
               CtxDestroyOpLowering, CtxGetTeamOpLowering>(converter);
} 
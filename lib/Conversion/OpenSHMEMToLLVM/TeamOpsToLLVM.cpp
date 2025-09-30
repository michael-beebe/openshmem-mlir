//===- TeamOpsToLLVM.cpp - Team operations conversion patterns -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TeamOpsToLLVM.h"
#include "OpenSHMEMConversionUtils.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::openshmem;

namespace {

//===----------------------------------------------------------------------===//
// TeamSplitStridedOp Lowering
//===----------------------------------------------------------------------===//

struct TeamSplitStridedOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TeamSplitStridedOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TeamSplitStridedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // int shmem_team_split_strided(shmem_team_t parent_team, int start, int
    // stride, int size,
    //                              const shmem_team_config_t *config, long
    //                              config_mask, shmem_team_t *new_team)
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(),
        {ptrType, rewriter.getI32Type(), rewriter.getI32Type(),
         rewriter.getI32Type(), ptrType, rewriter.getI64Type(), ptrType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_team_split_strided", funcType);

    // Allocate space for the new team
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
    Value newTeamPtr =
        rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType, one);

    // Pass NULL for config and 0 for config_mask (simplified)
    Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
    Value zeroMask = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getParentTeam(), adaptor.getStart(),
                   adaptor.getStride(), adaptor.getSize(), nullPtr, zeroMask,
                   newTeamPtr});

    // Load the new team pointer
    Value newTeam = rewriter.create<LLVM::LoadOp>(loc, ptrType, newTeamPtr);

    SmallVector<Value> replacements;
    replacements.push_back(newTeam);
    replacements.push_back(callOp.getResult());

    rewriter.replaceOp(op, replacements);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TeamSplit2dOp Lowering
//===----------------------------------------------------------------------===//

struct TeamSplit2dOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TeamSplit2dOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TeamSplit2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // int shmem_team_split_2d(shmem_team_t parent_team, int xrange,
    //                         const shmem_team_config_t *xaxis_config, long
    //                         xaxis_mask, shmem_team_t *xaxis_team, const
    //                         shmem_team_config_t *yaxis_config, long
    //                         yaxis_mask, shmem_team_t *yaxis_team)
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(),
        {ptrType, rewriter.getI32Type(), ptrType, rewriter.getI64Type(),
         ptrType, ptrType, rewriter.getI64Type(), ptrType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_team_split_2d", funcType);

    // Allocate space for the new teams
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
    Value xaxisTeamPtr =
        rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType, one);
    Value yaxisTeamPtr =
        rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType, one);

    // Pass NULL for configs and 0 for config_masks (simplified)
    Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
    Value zeroMask = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getParentTeam(), adaptor.getXrange(), nullPtr,
                   zeroMask, xaxisTeamPtr, nullPtr, zeroMask, yaxisTeamPtr});

    // Load the new team pointers
    Value xaxisTeam = rewriter.create<LLVM::LoadOp>(loc, ptrType, xaxisTeamPtr);
    Value yaxisTeam = rewriter.create<LLVM::LoadOp>(loc, ptrType, yaxisTeamPtr);

    SmallVector<Value> replacements;
    replacements.push_back(xaxisTeam);
    replacements.push_back(yaxisTeam);
    replacements.push_back(callOp.getResult());

    rewriter.replaceOp(op, replacements);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TeamMyPeOp Lowering
//===----------------------------------------------------------------------===//

struct TeamMyPeOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TeamMyPeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TeamMyPeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // int shmem_team_my_pe(shmem_team_t team)
    auto funcType =
        LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {ptrType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_team_my_pe", funcType);

    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                                ValueRange{adaptor.getTeam()});
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TeamNPesOp Lowering
//===----------------------------------------------------------------------===//

struct TeamNPesOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TeamNPesOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TeamNPesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // int shmem_team_n_pes(shmem_team_t team)
    auto funcType =
        LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {ptrType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_team_n_pes", funcType);

    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                                ValueRange{adaptor.getTeam()});
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TeamSyncOp Lowering
//===----------------------------------------------------------------------===//

struct TeamSyncOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TeamSyncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TeamSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void shmem_team_sync(shmem_team_t team)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_team_sync", funcType);

    rewriter.create<LLVM::CallOp>(loc, funcDecl, ValueRange{adaptor.getTeam()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TeamDestroyOp Lowering
//===----------------------------------------------------------------------===//

struct TeamDestroyOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TeamDestroyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TeamDestroyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void shmem_team_destroy(shmem_team_t team)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_team_destroy", funcType);

    rewriter.create<LLVM::CallOp>(loc, funcDecl, ValueRange{adaptor.getTeam()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TeamWorldOp Lowering
//===----------------------------------------------------------------------===//

struct TeamWorldOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TeamWorldOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TeamWorldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Get SHMEM_TEAM_WORLD as a global constant
    // In practice, this would be a predefined constant in the OpenSHMEM library
    LLVM::GlobalOp globalTeamWorld;
    if (!(globalTeamWorld =
              moduleOp.lookupSymbol<LLVM::GlobalOp>("SHMEM_TEAM_WORLD"))) {
      ConversionPatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      globalTeamWorld = rewriter.create<LLVM::GlobalOp>(
          loc, ptrType, /*isConstant=*/true, LLVM::Linkage::External,
          "SHMEM_TEAM_WORLD", Attribute{});
    }

    Value teamWorldAddr =
        rewriter.create<LLVM::AddressOfOp>(loc, globalTeamWorld);
    rewriter.replaceOp(op, teamWorldAddr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TeamSharedOp Lowering
//===----------------------------------------------------------------------===//

struct TeamSharedOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TeamSharedOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TeamSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Get SHMEM_TEAM_SHARED as a global constant
    LLVM::GlobalOp globalTeamShared;
    if (!(globalTeamShared =
              moduleOp.lookupSymbol<LLVM::GlobalOp>("SHMEM_TEAM_SHARED"))) {
      ConversionPatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      globalTeamShared = rewriter.create<LLVM::GlobalOp>(
          loc, ptrType, /*isConstant=*/true, LLVM::Linkage::External,
          "SHMEM_TEAM_SHARED", Attribute{});
    }

    Value teamSharedAddr =
        rewriter.create<LLVM::AddressOfOp>(loc, globalTeamShared);
    rewriter.replaceOp(op, teamSharedAddr);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void openshmem::populateTeamOpsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<TeamSplitStridedOpLowering, TeamSplit2dOpLowering,
               TeamMyPeOpLowering, TeamNPesOpLowering, TeamSyncOpLowering,
               TeamDestroyOpLowering, TeamWorldOpLowering,
               TeamSharedOpLowering>(converter);
}
 
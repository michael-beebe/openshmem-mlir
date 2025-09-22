//===- SetupOpsToLLVM.cpp - Setup Operations Conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion patterns for OpenSHMEM setup operations
// to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "SetupOpsToLLVM.h"
#include "OpenSHMEMConversionUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::openshmem;

namespace {

//===----------------------------------------------------------------------===//
// RegionOp Lowering
//===----------------------------------------------------------------------===//

struct RegionOpLowering : public ConvertOpToLLVMPattern<openshmem::Region> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::Region op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // Create init call
    auto initFuncType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {});
    LLVM::LLVMFuncOp initFuncDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_init", initFuncType);
    rewriter.create<LLVM::CallOp>(loc, initFuncDecl, ValueRange{});

    // Convert the region body by moving operations one by one
    // This ensures the openshmem.yield terminators are handled properly
    Block &regionBlock = op.getBody().front();
    Block *parentBlock = op->getBlock();
    auto insertionPoint = rewriter.getInsertionPoint();

    // Move all operations except the terminator
    for (auto it = regionBlock.begin(); it != regionBlock.end();) {
      Operation &operation = *it;
      ++it; // Increment before moving to avoid invalidating iterator

      if (isa<openshmem::YieldOp>(operation)) {
        // Erase yield operations - they're just terminators for the region
        rewriter.eraseOp(&operation);
      } else {
        // Move other operations to the parent block
        operation.moveBefore(parentBlock, insertionPoint);
      }
    }

    // Create finalize call
    auto finalizeFuncType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {});
    LLVM::LLVMFuncOp finalizeFuncDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_finalize", finalizeFuncType);
    rewriter.create<LLVM::CallOp>(loc, finalizeFuncDecl, ValueRange{});

    // Remove the region operation
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// InitOp Lowering
//===----------------------------------------------------------------------===//

struct InitOpLowering : public ConvertOpToLLVMPattern<openshmem::InitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::InitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // void shmem_init(void)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_init", funcType);

    // Replace with function call
    rewriter.create<LLVM::CallOp>(loc, funcDecl, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FinalizeOp Lowering
//===----------------------------------------------------------------------===//

struct FinalizeOpLowering
    : public ConvertOpToLLVMPattern<openshmem::FinalizeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::FinalizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // void shmem_finalize(void)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(moduleOp, loc, rewriter,
                                                    "shmem_finalize", funcType);

    // Replace with function call
    rewriter.create<LLVM::CallOp>(loc, funcDecl, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MyPeOp Lowering
//===----------------------------------------------------------------------===//

struct MyPeOpLowering : public ConvertOpToLLVMPattern<openshmem::MyPeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::MyPeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // int shmem_my_pe(void)
    auto funcType = LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_my_pe", funcType);

    // Replace with function call
    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcDecl, ValueRange{});
    rewriter.replaceOp(op, callOp.getResult());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// NPesOp Lowering
//===----------------------------------------------------------------------===//

struct NPesOpLowering : public ConvertOpToLLVMPattern<openshmem::NPesOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::NPesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    // int shmem_n_pes(void)
    auto funcType = LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_n_pes", funcType);

    // Replace with function call
    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcDecl, ValueRange{});
    rewriter.replaceOp(op, callOp.getResult());

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

struct OpenSHMEMYieldOpLowering
    : public ConvertOpToLLVMPattern<openshmem::YieldOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Yield operations are erased during region inlining
    rewriter.eraseOp(op);
    return success();
  }
};

void openshmem::populateSetupOpsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<RegionOpLowering, OpenSHMEMYieldOpLowering, InitOpLowering,
               FinalizeOpLowering, MyPeOpLowering, NPesOpLowering>(converter);
}

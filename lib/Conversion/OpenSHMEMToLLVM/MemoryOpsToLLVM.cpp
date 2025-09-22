//===- MemoryOpsToLLVM.cpp - Memory Operations Conversion -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion patterns for OpenSHMEM memory operations
// to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "MemoryOpsToLLVM.h"
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
// MallocOp Lowering
//===----------------------------------------------------------------------===//

struct MallocOpLowering : public ConvertOpToLLVMPattern<openshmem::MallocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::MallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void *shmem_malloc(size_t size)
    // size_t is typically the same as index type on the target platform
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(ptrType, {sizeType});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_malloc", funcType);

    // Replace with function call
    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                                ValueRange{adaptor.getSize()});

    // Return the pointer as the memref with symmetric memory space
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FreeOp Lowering
//===----------------------------------------------------------------------===//

struct FreeOpLowering : public ConvertOpToLLVMPattern<openshmem::FreeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::FreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void shmem_free(void *ptr)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrType});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_free", funcType);

    // The memref with symmetric memory space is just a pointer
    Value dataPtr = adaptor.getPtr();

    // Replace with function call
    rewriter.create<LLVM::CallOp>(loc, funcDecl, ValueRange{dataPtr});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReallocOp Lowering
//===----------------------------------------------------------------------===//

struct ReallocOpLowering : public ConvertOpToLLVMPattern<openshmem::ReallocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::ReallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // void *shmem_realloc(void *ptr, size_t size)
    auto funcType = LLVM::LLVMFunctionType::get(ptrType, {ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_realloc", funcType);

    // Replace with function call
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{adaptor.getPtr(), adaptor.getSize()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AlignOp Lowering
//===----------------------------------------------------------------------===//

struct AlignOpLowering : public ConvertOpToLLVMPattern<openshmem::AlignOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::AlignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // void *shmem_align(size_t alignment, size_t size)
    auto funcType = LLVM::LLVMFunctionType::get(ptrType, {sizeType, sizeType});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_align", funcType);

    // Replace with function call
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{adaptor.getAlignment(), adaptor.getSize()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CallocOp Lowering
//===----------------------------------------------------------------------===//

struct CallocOpLowering : public ConvertOpToLLVMPattern<openshmem::CallocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::CallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // void *shmem_calloc(size_t count, size_t size)
    auto funcType = LLVM::LLVMFunctionType::get(ptrType, {sizeType, sizeType});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_calloc", funcType);

    // Replace with function call
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{adaptor.getCount(), adaptor.getSize()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// OffsetOp Lowering
//===----------------------------------------------------------------------===//

static unsigned getScalarTypeSizeInBytes(Type t) {
  if (auto it = dyn_cast<IntegerType>(t))
    return it.getWidth() / 8;
  if (t.isF16())
    return 2;
  if (t.isBF16())
    return 2;
  if (t.isF32())
    return 4;
  if (t.isF64())
    return 8;
  if (t.isF128())
    return 16;
  return 1;
}

struct OffsetOpLowering : public ConvertOpToLLVMPattern<openshmem::OffsetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::OffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value basePtr = adaptor.getBase();
    Value offElems = adaptor.getOffsetElems();

    auto shmType = cast<MemRefType>(op.getBase().getType());
    unsigned elemSizeBytes = getScalarTypeSizeInBytes(shmType.getElementType());

    Type idxTy = getTypeConverter()->getIndexType();
    Value elemSizeConst = rewriter.create<LLVM::ConstantOp>(
        loc, idxTy, rewriter.getIntegerAttr(idxTy, elemSizeBytes));
    Value byteOffset =
        rewriter.create<LLVM::MulOp>(loc, offElems, elemSizeConst);

    auto i8Ty = IntegerType::get(ctx, 8);
    auto i8PtrTy = LLVM::LLVMPointerType::get(ctx);
    Value baseAsI8 = rewriter.create<LLVM::BitcastOp>(loc, i8PtrTy, basePtr);

    Value gep = rewriter.create<LLVM::GEPOp>(loc, i8PtrTy, i8Ty, baseAsI8,
                                             ArrayRef<Value>{byteOffset});

    Value resultPtr = rewriter.create<LLVM::BitcastOp>(loc, i8PtrTy, gep);
    rewriter.replaceOp(op, resultPtr);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void openshmem::populateMemoryOpsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<MallocOpLowering, FreeOpLowering, ReallocOpLowering,
               AlignOpLowering, CallocOpLowering, OffsetOpLowering>(converter);
}

//===- RMAOpsToLLVM.cpp - RMA operations conversion patterns -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RMAOpsToLLVM.h"
#include "OpenSHMEMConversionUtils.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
// Typed Put Operations Lowering
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PutOp Lowering (Typed - Generic)
//===----------------------------------------------------------------------===//

struct PutOpLowering : public ConvertOpToLLVMPattern<openshmem::PutOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::PutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Get element type from symmetric memref
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getRMASizedFunctionName("put", elementType);

    // void shmem_put(TYPE *dest, const TYPE *source, size_t nelems, int pe)
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // source: memref (need to extract pointer)
    Value sourcePtr = getMemRefDataPtr(loc, rewriter, adaptor.getSource());

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{destPtr, sourcePtr, adaptor.getNelems(), adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxPutOp Lowering (Typed - Context-aware)
//===----------------------------------------------------------------------===//

struct CtxPutOpLowering : public ConvertOpToLLVMPattern<openshmem::CtxPutOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::CtxPutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Get element type from symmetric memref
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getRMASizedFunctionName("ctx_put", elementType);

    // void shmem_put(shmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t
    // nelems, int pe)
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // source: memref (need to extract pointer)
    Value sourcePtr = getMemRefDataPtr(loc, rewriter, adaptor.getSource());

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{ctxPtr, destPtr, sourcePtr,
                                             adaptor.getNelems(),
                                             adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PutNbiOp Lowering (Typed - Non-blocking)
//===----------------------------------------------------------------------===//

struct PutNbiOpLowering : public ConvertOpToLLVMPattern<openshmem::PutNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::PutNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Get element type from symmetric memref
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getRMASizedFunctionName("put_nbi", elementType);

    // void shmem_put_n(TYPE *dest, const TYPE *source, size_t nelems, int pe)
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // source: memref (need to extract pointer)
    Value sourcePtr = getMemRefDataPtr(loc, rewriter, adaptor.getSource());

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{destPtr, sourcePtr, adaptor.getNelems(), adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxPutNbiOp Lowering (Typed - Context-aware Non-blocking)
//===----------------------------------------------------------------------===//

struct CtxPutNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxPutNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::CtxPutNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Get element type from symmetric memref
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getRMASizedFunctionName("ctx_put_nbi", elementType);

    // void shmem_ctx_put_nbi(shmem_ctx_t ctx, TYPE *dest, const TYPE *source,
    // size_t nelems, int pe)
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // source: memref (need to extract pointer)
    Value sourcePtr = getMemRefDataPtr(loc, rewriter, adaptor.getSource());

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{ctxPtr, destPtr, sourcePtr,
                                             adaptor.getNelems(),
                                             adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PutmemOp Lowering
//===----------------------------------------------------------------------===//

struct PutmemOpLowering : public ConvertOpToLLVMPattern<openshmem::PutmemOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::PutmemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // src: memref (need to extract pointer)
    Value srcPtr = getMemRefDataPtr(loc, rewriter, adaptor.getSrc());

    // void shmem_putmem(void *dest, const void *source, size_t nelems, int pe)
    // Note: for putmem, nelems represents the number of bytes to transfer
    // size_t is typically the same as index type on the target platform
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_putmem", funcType);

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{destPtr, srcPtr, adaptor.getSize(), adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PutmemNbiOp Lowering
//===----------------------------------------------------------------------===//

struct PutmemNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::PutmemNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::PutmemNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // src: memref (need to extract pointer)
    Value srcPtr = getMemRefDataPtr(loc, rewriter, adaptor.getSrc());

    // void shmem_putmem_nbi(void *dest, const void *source, size_t nelems, int
    // pe)
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_putmem_nbi", funcType);

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{destPtr, srcPtr, adaptor.getNelems(), adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GetOp Lowering (Typed - Generic)
//===----------------------------------------------------------------------===//
struct GetOpLowering : public ConvertOpToLLVMPattern<openshmem::GetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::GetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getSource());
    if (!elementType)
      return failure();
    std::string funcName = getRMASizedFunctionName("get", elementType);
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);
    Value destPtr = getMemRefDataPtr(loc, rewriter, adaptor.getDest());
    Value sourcePtr = adaptor.getSource();
    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{destPtr, sourcePtr, adaptor.getNelems(), adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxGetOp Lowering (Typed - Context-aware)
//===----------------------------------------------------------------------===//
struct CtxGetOpLowering : public ConvertOpToLLVMPattern<openshmem::CtxGetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getSource());
    if (!elementType)
      return failure();
    std::string funcName = getRMASizedFunctionName("ctx_get", elementType);
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);
    Value ctxPtr = adaptor.getCtx();
    Value destPtr = getMemRefDataPtr(loc, rewriter, adaptor.getDest());
    Value sourcePtr = adaptor.getSource();
    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{ctxPtr, destPtr, sourcePtr,
                                             adaptor.getNelems(),
                                             adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GetNbiOp Lowering (Typed - Non-blocking)
//===----------------------------------------------------------------------===//
struct GetNbiOpLowering : public ConvertOpToLLVMPattern<openshmem::GetNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::GetNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getSource());
    if (!elementType)
      return failure();
    std::string funcName = getRMASizedFunctionName("get_nbi", elementType);
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);
    Value destPtr = getMemRefDataPtr(loc, rewriter, adaptor.getDest());
    Value sourcePtr = adaptor.getSource();
    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{destPtr, sourcePtr, adaptor.getNelems(), adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxGetNbiOp Lowering (Typed - Context-aware Non-blocking)
//===----------------------------------------------------------------------===//
struct CtxGetNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxGetNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxGetNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getSource());
    if (!elementType)
      return failure();
    std::string funcName = getRMASizedFunctionName("ctx_get_nbi", elementType);
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);
    Value ctxPtr = adaptor.getCtx();
    Value destPtr = getMemRefDataPtr(loc, rewriter, adaptor.getDest());
    Value sourcePtr = adaptor.getSource();
    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{ctxPtr, destPtr, sourcePtr,
                                             adaptor.getNelems(),
                                             adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Get8/16/32/64/128Op Lowering (Sized)
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// GetmemOp Lowering
//===----------------------------------------------------------------------===//

struct GetmemOpLowering : public ConvertOpToLLVMPattern<openshmem::GetmemOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::GetmemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // dest: memref (need to extract pointer)
    Value destPtr = getMemRefDataPtr(loc, rewriter, adaptor.getDest());
    // src: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value srcPtr = adaptor.getSrc();

    // void shmem_getmem(void *dest, const void *source, size_t nelems, int pe)
    // Note: for getmem, nelems represents the number of bytes to transfer
    // size_t is typically the same as index type on the target platform
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_getmem", funcType);

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{destPtr, srcPtr, adaptor.getSize(), adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GetmemNbiOp Lowering
//===----------------------------------------------------------------------===//

struct GetmemNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::GetmemNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::GetmemNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // dest: memref (need to extract pointer)
    Value destPtr = getMemRefDataPtr(loc, rewriter, adaptor.getDest());
    // src: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value srcPtr = adaptor.getSrc();

    // void shmem_getmem_nbi(void *dest, const void *source, size_t nelems, int
    // pe)
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_getmem_nbi", funcType);

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{destPtr, srcPtr, adaptor.getNelems(), adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

// Note: single-element P/G ops and their sized variants were removed from the
// dialect in the simplified RMA op set. Lowering for those ops was therefore
// removed. Remaining lowerings handle the generic typed ops, ctx variants,
// non-blocking variants, and mem-level (byte) ops.

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void openshmem::populateRMAOpsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Register conversion patterns for the simplified RMA op set. Sized and
  // single-element ops were removed from the dialect and therefore their
  // lowerings are not registered here.
  patterns.add<PutOpLowering, CtxPutOpLowering, PutNbiOpLowering,
               CtxPutNbiOpLowering, PutmemOpLowering, PutmemNbiOpLowering,
               GetOpLowering, CtxGetOpLowering, GetNbiOpLowering,
               CtxGetNbiOpLowering, GetmemOpLowering, GetmemNbiOpLowering>(
      converter);
}

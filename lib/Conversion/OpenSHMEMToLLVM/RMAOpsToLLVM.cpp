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
    std::string funcName = getRMASizedFunctionName("put_nbi", elementType);

    // void shmem_put_nbi(shmem_ctx_t ctx, TYPE *dest, const TYPE *source,
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
// PutSizedOp Lowering (Sized variants)
//===----------------------------------------------------------------------===//

#define GEN_PUT_SIZED_LOWERING(SZ)                                             \
  struct Put##SZ##OpLowering                                                   \
      : public ConvertOpToLLVMPattern<openshmem::Put##SZ##Op> {                \
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;                      \
    LogicalResult                                                              \
    matchAndRewrite(openshmem::Put##SZ##Op op, OpAdaptor adaptor,              \
                    ConversionPatternRewriter &rewriter) const override {      \
      Location loc = op.getLoc();                                              \
      auto moduleOp = op->getParentOfType<ModuleOp>();                         \
      Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());        \
      Type sizeType = getTypeConverter()->getIndexType();                      \
      auto funcType = LLVM::LLVMFunctionType::get(                             \
          mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),                \
          {ptrType, ptrType, sizeType, rewriter.getI32Type()});                \
      LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(                         \
          moduleOp, loc, rewriter, "shmem_put" #SZ, funcType);                 \
      Value destPtr = adaptor.getDest();                                       \
      Value sourcePtr = getMemRefDataPtr(loc, rewriter, adaptor.getSource());  \
      rewriter.create<LLVM::CallOp>(loc, funcDecl,                             \
                                    ValueRange{destPtr, sourcePtr,             \
                                               adaptor.getNelems(),            \
                                               adaptor.getPe()});              \
      rewriter.eraseOp(op);                                                    \
      return success();                                                        \
    }                                                                          \
  };
GEN_PUT_SIZED_LOWERING(8)
GEN_PUT_SIZED_LOWERING(16)
GEN_PUT_SIZED_LOWERING(32)
GEN_PUT_SIZED_LOWERING(64)
GEN_PUT_SIZED_LOWERING(128)
#undef GEN_PUT_SIZED_LOWERING

//===----------------------------------------------------------------------===//
// CtxPut8/16/32/64/128Op Lowering (Context-aware Sized)
//===----------------------------------------------------------------------===//
#define GEN_CTX_PUT_SIZED_LOWERING(SZ)                                         \
  struct CtxPut##SZ##OpLowering                                                \
      : public ConvertOpToLLVMPattern<openshmem::CtxPut##SZ##Op> {             \
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;                      \
    LogicalResult                                                              \
    matchAndRewrite(openshmem::CtxPut##SZ##Op op, OpAdaptor adaptor,           \
                    ConversionPatternRewriter &rewriter) const override {      \
      Location loc = op.getLoc();                                              \
      auto moduleOp = op->getParentOfType<ModuleOp>();                         \
      Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());        \
      Type sizeType = getTypeConverter()->getIndexType();                      \
      auto funcType = LLVM::LLVMFunctionType::get(                             \
          mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),                \
          {ptrType, ptrType, ptrType, sizeType, rewriter.getI32Type()});       \
      LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(                         \
          moduleOp, loc, rewriter, "shmem_ctx_put" #SZ, funcType);             \
      Value ctxPtr = adaptor.getCtx();                                         \
      Value destPtr = adaptor.getDest();                                       \
      Value sourcePtr = getMemRefDataPtr(loc, rewriter, adaptor.getSource());  \
      rewriter.create<LLVM::CallOp>(loc, funcDecl,                             \
                                    ValueRange{ctxPtr, destPtr, sourcePtr,     \
                                               adaptor.getNelems(),            \
                                               adaptor.getPe()});              \
      rewriter.eraseOp(op);                                                    \
      return success();                                                        \
    }                                                                          \
  };
GEN_CTX_PUT_SIZED_LOWERING(8)
GEN_CTX_PUT_SIZED_LOWERING(16)
GEN_CTX_PUT_SIZED_LOWERING(32)
GEN_CTX_PUT_SIZED_LOWERING(64)
GEN_CTX_PUT_SIZED_LOWERING(128)
#undef GEN_CTX_PUT_SIZED_LOWERING

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
#define GEN_GET_SIZED_LOWERING(SZ)                                             \
  struct Get##SZ##OpLowering                                                   \
      : public ConvertOpToLLVMPattern<openshmem::Get##SZ##Op> {                \
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;                      \
    LogicalResult                                                              \
    matchAndRewrite(openshmem::Get##SZ##Op op, OpAdaptor adaptor,              \
                    ConversionPatternRewriter &rewriter) const override {      \
      Location loc = op.getLoc();                                              \
      auto moduleOp = op->getParentOfType<ModuleOp>();                         \
      Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());        \
      Type sizeType = getTypeConverter()->getIndexType();                      \
      auto funcType = LLVM::LLVMFunctionType::get(                             \
          mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),                \
          {ptrType, ptrType, sizeType, rewriter.getI32Type()});                \
      LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(                         \
          moduleOp, loc, rewriter, "shmem_get" #SZ, funcType);                 \
      Value destPtr = getMemRefDataPtr(loc, rewriter, adaptor.getDest());      \
      Value sourcePtr = adaptor.getSource();                                   \
      rewriter.create<LLVM::CallOp>(loc, funcDecl,                             \
                                    ValueRange{destPtr, sourcePtr,             \
                                               adaptor.getNelems(),            \
                                               adaptor.getPe()});              \
      rewriter.eraseOp(op);                                                    \
      return success();                                                        \
    }                                                                          \
  };
GEN_GET_SIZED_LOWERING(8)
GEN_GET_SIZED_LOWERING(16)
GEN_GET_SIZED_LOWERING(32)
GEN_GET_SIZED_LOWERING(64)
GEN_GET_SIZED_LOWERING(128)
#undef GEN_GET_SIZED_LOWERING

//===----------------------------------------------------------------------===//
// CtxGet8/16/32/64/128Op Lowering (Context-aware Sized)
//===----------------------------------------------------------------------===//
#define GEN_CTX_GET_SIZED_LOWERING(SZ)                                         \
  struct CtxGet##SZ##OpLowering                                                \
      : public ConvertOpToLLVMPattern<openshmem::CtxGet##SZ##Op> {             \
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;                      \
    LogicalResult                                                              \
    matchAndRewrite(openshmem::CtxGet##SZ##Op op, OpAdaptor adaptor,           \
                    ConversionPatternRewriter &rewriter) const override {      \
      Location loc = op.getLoc();                                              \
      auto moduleOp = op->getParentOfType<ModuleOp>();                         \
      Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());        \
      Type sizeType = getTypeConverter()->getIndexType();                      \
      auto funcType = LLVM::LLVMFunctionType::get(                             \
          mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),                \
          {ptrType, ptrType, ptrType, sizeType, rewriter.getI32Type()});       \
      LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(                         \
          moduleOp, loc, rewriter, "shmem_ctx_get" #SZ, funcType);             \
      Value ctxPtr = adaptor.getCtx();                                         \
      Value destPtr = getMemRefDataPtr(loc, rewriter, adaptor.getDest());      \
      Value sourcePtr = adaptor.getSource();                                   \
      rewriter.create<LLVM::CallOp>(loc, funcDecl,                             \
                                    ValueRange{ctxPtr, destPtr, sourcePtr,     \
                                               adaptor.getNelems(),            \
                                               adaptor.getPe()});              \
      rewriter.eraseOp(op);                                                    \
      return success();                                                        \
    }                                                                          \
  };
GEN_CTX_GET_SIZED_LOWERING(8)
GEN_CTX_GET_SIZED_LOWERING(16)
GEN_CTX_GET_SIZED_LOWERING(32)
GEN_CTX_GET_SIZED_LOWERING(64)
GEN_CTX_GET_SIZED_LOWERING(128)
#undef GEN_CTX_GET_SIZED_LOWERING

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

//===----------------------------------------------------------------------===//
// POp Lowering
//===----------------------------------------------------------------------===//

struct POpLowering : public ConvertOpToLLVMPattern<openshmem::POp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::POp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: any type
    Value value = adaptor.getValue();

    // void shmem_p(void *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, value.getType(), rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_p", funcType);

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxPOp Lowering
//===----------------------------------------------------------------------===//

struct CtxPOpLowering : public ConvertOpToLLVMPattern<openshmem::CtxPOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::CtxPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // ctx: ctx (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: any type
    Value value = adaptor.getValue();

    // void shmem_ctx_p(shmem_ctx_t ctx, void *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, value.getType(), rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_ctx_p", funcType);

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GOp Lowering
//===----------------------------------------------------------------------===//

struct GOpLowering : public ConvertOpToLLVMPattern<openshmem::GOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::GOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Get the return type from the operation result
    Type resultType = op.getValue().getType();

    // source: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value sourcePtr = adaptor.getSource();

    // TYPE shmem_g(const void *source, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        resultType, {ptrType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_g", funcType);

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{sourcePtr, adaptor.getPe()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxGOp Lowering
//===----------------------------------------------------------------------===//

struct CtxGOpLowering : public ConvertOpToLLVMPattern<openshmem::CtxGOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::CtxGOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Get the return type from the operation result
    Type resultType = op.getValue().getType();

    // ctx: ctx (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // source: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value sourcePtr = adaptor.getSource();

    // TYPE shmem_ctx_g(shmem_ctx_t ctx, TYPE *source, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        resultType, {ptrType, ptrType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, "shmem_ctx_g", funcType);

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, sourcePtr, adaptor.getPe()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void openshmem::populateRMAOpsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<
      // Typed put operations
      PutOpLowering, CtxPutOpLowering, PutNbiOpLowering, CtxPutNbiOpLowering,
      // Sized put operations
      Put8OpLowering, Put16OpLowering, Put32OpLowering, Put64OpLowering,
      Put128OpLowering,
      // Context-aware sized put operations
      CtxPut8OpLowering, CtxPut16OpLowering, CtxPut32OpLowering,
      CtxPut64OpLowering, CtxPut128OpLowering,
      // Memory put operations
      PutmemOpLowering, PutmemNbiOpLowering,
      // Typed get operations
      GetOpLowering, CtxGetOpLowering, GetNbiOpLowering, CtxGetNbiOpLowering,
      // Sized get operations
      Get8OpLowering, Get16OpLowering, Get32OpLowering, Get64OpLowering,
      Get128OpLowering,
      // Context-aware sized get operations
      CtxGet8OpLowering, CtxGet16OpLowering, CtxGet32OpLowering,
      CtxGet64OpLowering, CtxGet128OpLowering,
      // Memory get operations
      GetmemOpLowering, GetmemNbiOpLowering,
      // Single-element operations
      POpLowering, CtxPOpLowering, GOpLowering, CtxGOpLowering>(converter);
}

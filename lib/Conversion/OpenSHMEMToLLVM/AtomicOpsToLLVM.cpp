//===- AtomicOpsToLLVM.cpp - Atomic operations conversion patterns -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AtomicOpsToLLVM.h"
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
// AtomicFetchOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicFetchOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicFetchOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicFetchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Get element type from symmetric memref
    Type elementType = getSymmetricMemRefElementType(op.getSource());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("atomic_fetch", elementType);

    // Determine the return type for the function call
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_atomic_fetch(const TYPE *source, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType, {ptrType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // source: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value sourcePtr = adaptor.getSource();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{sourcePtr, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicFetchOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicFetchOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicFetchOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicFetchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Get element type from symmetric memref
    Type elementType = getSymmetricMemRefElementType(op.getSource());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_fetch", elementType);

    // Determine the return type for the function call
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_atomic_fetch(shmem_ctx_t ctx, const TYPE *source, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType, {ptrType, ptrType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // source: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value sourcePtr = adaptor.getSource();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, sourcePtr, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicSetOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicSetOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicSetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("atomic_set", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_atomic_set(TYPE *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicSetOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicSetOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicSetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("ctx_atomic_set", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_ctx_atomic_set(shmem_ctx_t ctx, TYPE *dest, TYPE value, int
    // pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicCompareSwapOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicCompareSwapOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicCompareSwapOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicCompareSwapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("atomic_compare_swap", elementType);

    // Determine the type for the value arguments in the function call
    Type condCallType = elementType;
    Type valueCallType = elementType;
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      condCallType = rewriter.getI64Type();
      valueCallType = rewriter.getI64Type();
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      condCallType = rewriter.getI32Type();
      valueCallType = rewriter.getI32Type();
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_atomic_compare_swap(TYPE *dest, TYPE cond, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType,
        {ptrType, condCallType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // cond: scalar value, may need casting
    Value cond = adaptor.getCond();
    if (elementType.isF64()) {
      cond = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), cond);
    } else if (elementType.isF32()) {
      cond = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), cond);
    }

    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{destPtr, cond, value, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicCompareSwapOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicCompareSwapOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicCompareSwapOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicCompareSwapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_compare_swap", elementType);

    // Determine the type for the value arguments in the function call
    Type condCallType = elementType;
    Type valueCallType = elementType;
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      condCallType = rewriter.getI64Type();
      valueCallType = rewriter.getI64Type();
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      condCallType = rewriter.getI32Type();
      valueCallType = rewriter.getI32Type();
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_ctx_atomic_compare_swap(shmem_ctx_t ctx, TYPE *dest, TYPE
    // cond, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType,
        {ptrType, ptrType, condCallType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // cond: scalar value, may need casting
    Value cond = adaptor.getCond();
    if (elementType.isF64()) {
      cond = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), cond);
    } else if (elementType.isF32()) {
      cond = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), cond);
    }

    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ctxPtr, destPtr, cond, value, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicSwapOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicSwapOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicSwapOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicSwapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("atomic_swap", elementType);

    // Determine the type for the value arguments in the function call
    Type valueCallType = elementType;
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_atomic_swap(TYPE *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType, {ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{destPtr, value, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicSwapOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicSwapOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicSwapOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicSwapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("ctx_atomic_swap", elementType);

    // Determine the type for the value arguments in the function call
    Type valueCallType = elementType;
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_ctx_atomic_swap(shmem_ctx_t ctx, TYPE *dest, TYPE value, int
    // pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType,
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, destPtr, value, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicFetchIncOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicFetchIncOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicFetchIncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicFetchIncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("atomic_fetch_inc", elementType);

    // Determine the return type for the function call
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_atomic_fetch_inc(TYPE *dest, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType, {ptrType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{destPtr, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicFetchIncOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicFetchIncOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicFetchIncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicFetchIncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_fetch_inc", elementType);

    // Determine the return type for the function call
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_ctx_atomic_fetch_inc(shmem_ctx_t ctx, TYPE *dest, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType, {ptrType, ptrType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, destPtr, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicIncOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicIncOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicIncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicIncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("atomic_inc", elementType);

    // void shmem_atomic_inc(TYPE *dest, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{destPtr, adaptor.getPe()});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicIncOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicIncOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicIncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicIncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("ctx_atomic_inc", elementType);

    // void shmem_ctx_atomic_inc(shmem_ctx_t ctx, TYPE *dest, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{ctxPtr, destPtr, adaptor.getPe()});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicFetchAddOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicFetchAddOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicFetchAddOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicFetchAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("atomic_fetch_add", elementType);

    // Determine the type for the value arguments in the function call
    Type valueCallType = elementType;
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_atomic_fetch_add(TYPE *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType, {ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{destPtr, value, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicFetchAddOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicFetchAddOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicFetchAddOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicFetchAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_fetch_add", elementType);

    // Determine the type for the value arguments in the function call
    Type valueCallType = elementType;
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_ctx_atomic_fetch_add(shmem_ctx_t ctx, TYPE *dest, TYPE value,
    // int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType,
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, destPtr, value, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicAddOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicAddOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicAddOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("atomic_add", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_atomic_add(TYPE *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicAddOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicAddOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicAddOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("ctx_atomic_add", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_ctx_atomic_add(shmem_ctx_t ctx, TYPE *dest, TYPE value, int
    // pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicFetchAndOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicFetchAndOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicFetchAndOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicFetchAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("atomic_fetch_and", elementType);

    // Determine the type for the value arguments in the function call
    Type valueCallType = elementType;
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_atomic_fetch_and(TYPE *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType, {ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{destPtr, value, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicFetchAndOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicFetchAndOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicFetchAndOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicFetchAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_fetch_and", elementType);

    // Determine the type for the value arguments in the function call
    Type valueCallType = elementType;
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_ctx_atomic_fetch_and(shmem_ctx_t ctx, TYPE *dest, TYPE value,
    // int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType,
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, destPtr, value, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicFetchOrOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicFetchOrOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicFetchOrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicFetchOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("atomic_fetch_or", elementType);

    // Determine the type for the value arguments in the function call
    Type valueCallType = elementType;
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_atomic_fetch_or(TYPE *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType, {ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{destPtr, value, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicFetchOrOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicFetchOrOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicFetchOrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicFetchOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_fetch_or", elementType);

    // Determine the type for the value arguments in the function call
    Type valueCallType = elementType;
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_ctx_atomic_fetch_or(shmem_ctx_t ctx, TYPE *dest, TYPE value,
    // int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType,
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, destPtr, value, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicOrOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicOrOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicOrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("atomic_or", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_atomic_or(TYPE *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicOrOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicOrOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicOrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("ctx_atomic_or", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_ctx_atomic_or(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicFetchXorOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicFetchXorOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicFetchXorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicFetchXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("atomic_fetch_xor", elementType);

    // Determine the type for the value arguments in the function call
    Type valueCallType = elementType;
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_atomic_fetch_xor(TYPE *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType, {ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{destPtr, value, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicFetchXorOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicFetchXorOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicFetchXorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicFetchXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_fetch_xor", elementType);

    // Determine the type for the value arguments in the function call
    Type valueCallType = elementType;
    Type returnCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
      returnCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
      returnCallType = rewriter.getI32Type();
    }

    // TYPE shmem_ctx_atomic_fetch_xor(shmem_ctx_t ctx, TYPE *dest, TYPE value,
    // int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        returnCallType,
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, destPtr, value, adaptor.getPe()});

    Value result = callOp.getResult();
    // Convert result back to original type if needed
    if (elementType.isF64()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    } else if (elementType.isF32()) {
      result = rewriter.create<LLVM::BitcastOp>(loc, elementType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicXorOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicXorOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicXorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("atomic_xor", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_atomic_xor(TYPE *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicXorOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicXorOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicXorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("ctx_atomic_xor", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_ctx_atomic_xor(shmem_ctx_t ctx, TYPE *dest, TYPE value, int
    // pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicFetchNbiOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicFetchNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicFetchNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicFetchNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getSource());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("atomic_fetch_nbi", elementType);

    // void shmem_atomic_fetch_nbi(TYPE *fetch, const TYPE *source, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // fetch: local buffer (extract pointer from memref descriptor)
    Value fetchPtr = adaptor.getFetch();
    // source: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value sourcePtr = adaptor.getSource();

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{fetchPtr, sourcePtr, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicFetchNbiOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicFetchNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicFetchNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicFetchNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getSource());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_fetch_nbi", elementType);

    // void shmem_ctx_atomic_fetch_nbi(shmem_ctx_t ctx, TYPE *fetch, const TYPE
    // *source, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, ptrType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // fetch: local buffer (extract pointer from memref descriptor)
    Value fetchPtr = adaptor.getFetch();
    // source: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value sourcePtr = adaptor.getSource();

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ctxPtr, fetchPtr, sourcePtr, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicCompareSwapNbiOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicCompareSwapNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicCompareSwapNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicCompareSwapNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("atomic_compare_swap_nbi", elementType);

    // Determine the type for the value arguments in the function call
    Type condCallType = elementType;
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      condCallType = rewriter.getI64Type();
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      condCallType = rewriter.getI32Type();
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_atomic_compare_swap_nbi(TYPE *fetch, TYPE *dest, TYPE cond,
    // TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, condCallType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // cond: scalar value, may need casting
    Value cond = adaptor.getCond();
    if (elementType.isF64()) {
      cond = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), cond);
    } else if (elementType.isF32()) {
      cond = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), cond);
    }

    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{fetchPtr, destPtr, cond, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicCompareSwapNbiOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicCompareSwapNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicCompareSwapNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicCompareSwapNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_compare_swap_nbi", elementType);

    // Determine the type for the value arguments in the function call
    Type condCallType = elementType;
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      condCallType = rewriter.getI64Type();
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      condCallType = rewriter.getI32Type();
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_ctx_atomic_compare_swap_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE
    // *dest, TYPE cond, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, ptrType, condCallType, valueCallType,
         rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // cond: scalar value, may need casting
    Value cond = adaptor.getCond();
    if (elementType.isF64()) {
      cond = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), cond);
    } else if (elementType.isF32()) {
      cond = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), cond);
    }

    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ctxPtr, fetchPtr, destPtr, cond, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicSwapNbiOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicSwapNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicSwapNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicSwapNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName = getTypedFunctionName("atomic_swap_nbi", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_atomic_swap_nbi(TYPE *fetch, TYPE *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{fetchPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicSwapNbiOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicSwapNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicSwapNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicSwapNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_swap_nbi", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_ctx_atomic_swap_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE *dest,
    // TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ctxPtr, fetchPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicFetchIncNbiOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicFetchIncNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicFetchIncNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicFetchIncNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("atomic_fetch_inc_nbi", elementType);

    // void shmem_atomic_fetch_inc_nbi(TYPE *fetch, TYPE *dest, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{fetchPtr, destPtr, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicFetchIncNbiOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicFetchIncNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicFetchIncNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicFetchIncNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_fetch_inc_nbi", elementType);

    // void shmem_ctx_atomic_fetch_inc_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE
    // *dest, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, ptrType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{ctxPtr, fetchPtr, destPtr, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicFetchAddNbiOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicFetchAddNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicFetchAddNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicFetchAddNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("atomic_fetch_add_nbi", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_atomic_fetch_add_nbi(TYPE *fetch, TYPE *dest, TYPE value, int
    // pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{fetchPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicFetchAddNbiOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicFetchAddNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicFetchAddNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicFetchAddNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_fetch_add_nbi", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_ctx_atomic_fetch_add_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE
    // *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ctxPtr, fetchPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicFetchAndNbiOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicFetchAndNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicFetchAndNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicFetchAndNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("atomic_fetch_and_nbi", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_atomic_fetch_and_nbi(TYPE *fetch, TYPE *dest, TYPE value, int
    // pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{fetchPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicFetchAndNbiOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicFetchAndNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicFetchAndNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicFetchAndNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_fetch_and_nbi", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_ctx_atomic_fetch_and_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE
    // *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ctxPtr, fetchPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicFetchOrNbiOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicFetchOrNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicFetchOrNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicFetchOrNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("atomic_fetch_or_nbi", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_atomic_fetch_or_nbi(TYPE *fetch, TYPE *dest, TYPE value, int
    // pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{fetchPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicFetchOrNbiOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicFetchOrNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicFetchOrNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicFetchOrNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_fetch_or_nbi", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_ctx_atomic_fetch_or_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE
    // *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ctxPtr, fetchPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicFetchXorNbiOp Lowering
//===----------------------------------------------------------------------===//

struct AtomicFetchXorNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AtomicFetchXorNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::AtomicFetchXorNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("atomic_fetch_xor_nbi", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_atomic_fetch_xor_nbi(TYPE *fetch, TYPE *dest, TYPE value, int
    // pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl, ValueRange{fetchPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CtxAtomicFetchXorNbiOp Lowering
//===----------------------------------------------------------------------===//

struct CtxAtomicFetchXorNbiOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CtxAtomicFetchXorNbiOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(openshmem::CtxAtomicFetchXorNbiOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type elementType = getSymmetricMemRefElementType(op.getDest());
    if (!elementType) {
      return failure();
    }

    // Generate function name based on type
    std::string funcName =
        getTypedFunctionName("ctx_atomic_fetch_xor_nbi", elementType);

    // Determine the type for the value argument in the function call
    Type valueCallType = elementType;
    if (elementType.isF64()) {
      valueCallType = rewriter.getI64Type();
    } else if (elementType.isF32()) {
      valueCallType = rewriter.getI32Type();
    }

    // void shmem_ctx_atomic_fetch_xor_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE
    // *dest, TYPE value, int pe)
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, ptrType, ptrType, valueCallType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ctx: context (already a pointer after type conversion)
    Value ctxPtr = adaptor.getCtx();
    // fetch: local buffer (already converted to pointer)
    Value fetchPtr = adaptor.getFetch();
    // dest: memref with symmetric memory space (already a pointer after type
    // conversion)
    Value destPtr = adaptor.getDest();
    // value: scalar value, may need casting
    Value value = adaptor.getValue();
    if (elementType.isF64()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), value);
    } else if (elementType.isF32()) {
      value =
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), value);
    }

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ctxPtr, fetchPtr, destPtr, value, adaptor.getPe()});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void openshmem::populateAtomicOpsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<
      AtomicFetchOpLowering, CtxAtomicFetchOpLowering, AtomicSetOpLowering,
      CtxAtomicSetOpLowering, AtomicCompareSwapOpLowering,
      CtxAtomicCompareSwapOpLowering, AtomicSwapOpLowering,
      CtxAtomicSwapOpLowering, AtomicFetchIncOpLowering,
      CtxAtomicFetchIncOpLowering, AtomicIncOpLowering, CtxAtomicIncOpLowering,
      AtomicFetchAddOpLowering, CtxAtomicFetchAddOpLowering,
      AtomicAddOpLowering, CtxAtomicAddOpLowering, AtomicFetchAndOpLowering,
      CtxAtomicFetchAndOpLowering, AtomicFetchOrOpLowering,
      CtxAtomicFetchOrOpLowering, AtomicOrOpLowering, CtxAtomicOrOpLowering,
      AtomicFetchXorOpLowering, CtxAtomicFetchXorOpLowering,
      AtomicXorOpLowering, CtxAtomicXorOpLowering, AtomicFetchNbiOpLowering,
      CtxAtomicFetchNbiOpLowering, AtomicCompareSwapNbiOpLowering,
      CtxAtomicCompareSwapNbiOpLowering, AtomicSwapNbiOpLowering,
      CtxAtomicSwapNbiOpLowering, AtomicFetchIncNbiOpLowering,
      CtxAtomicFetchIncNbiOpLowering, AtomicFetchAddNbiOpLowering,
      CtxAtomicFetchAddNbiOpLowering, AtomicFetchAndNbiOpLowering,
      CtxAtomicFetchAndNbiOpLowering, AtomicFetchOrNbiOpLowering,
      CtxAtomicFetchOrNbiOpLowering, AtomicFetchXorNbiOpLowering,
      CtxAtomicFetchXorNbiOpLowering>(converter);
}

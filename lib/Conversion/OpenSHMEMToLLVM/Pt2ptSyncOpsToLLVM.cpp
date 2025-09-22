//===- Pt2ptSyncOpsToLLVM.cpp - Point-to-point synchronization ops conversion
// patterns -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pt2ptSyncOpsToLLVM.h"
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
// WaitUntilOp Lowering
//===----------------------------------------------------------------------===//

struct WaitUntilOpLowering
    : public ConvertOpToLLVMPattern<openshmem::WaitUntilOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::WaitUntilOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void shmem_wait_until(TYPE *ivar, int cmp, TYPE cmp_value)
    Type cmpValueType = adaptor.getCmpValue().getType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, rewriter.getI32Type(), cmpValueType});

    std::string funcName =
        getPt2ptSyncSizedFunctionName("wait_until", cmpValueType);
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivar is a symmetric memref (already converted to pointer)
    Value ivarPtr = adaptor.getIvar();

    rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ivarPtr, adaptor.getCmp(), adaptor.getCmpValue()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// WaitUntilAllOp Lowering
//===----------------------------------------------------------------------===//

struct WaitUntilAllOpLowering
    : public ConvertOpToLLVMPattern<openshmem::WaitUntilAllOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::WaitUntilAllOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void shmem_wait_until_all(TYPE *ivars, size_t nelems, const int *status,
    //                           int cmp, TYPE cmp_value)
    Type sizeType = getTypeConverter()->getIndexType();
    Type cmpValueType = adaptor.getCmpValue().getType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, sizeType, ptrType, rewriter.getI32Type(), cmpValueType});

    std::string funcName =
        getPt2ptSyncSizedFunctionName("wait_until_all", cmpValueType);
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivars is a symmetric memref (already converted to pointer)
    Value ivarsPtr = adaptor.getIvars();

    // status is a regular memref (need to extract data pointer)
    Value statusPtr = getMemRefDataPtr(loc, rewriter, adaptor.getStatus());

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{ivarsPtr, adaptor.getNelems(),
                                             statusPtr, adaptor.getCmp(),
                                             adaptor.getCmpValue()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// WaitUntilAnyOp Lowering
//===----------------------------------------------------------------------===//

struct WaitUntilAnyOpLowering
    : public ConvertOpToLLVMPattern<openshmem::WaitUntilAnyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::WaitUntilAnyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void shmem_wait_until_any(TYPE *ivars, size_t nelems, const int *status,
    //                           int cmp, TYPE cmp_value)
    Type sizeType = getTypeConverter()->getIndexType();
    Type cmpValueType = adaptor.getCmpValue().getType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, sizeType, ptrType, rewriter.getI32Type(), cmpValueType});

    std::string funcName =
        getPt2ptSyncSizedFunctionName("wait_until_any", cmpValueType);
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivars is a symmetric memref (already converted to pointer)
    Value ivarsPtr = adaptor.getIvars();
    // status is a regular memref (need to extract data pointer)
    Value statusPtr = getMemRefDataPtr(loc, rewriter, adaptor.getStatus());

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{ivarsPtr, adaptor.getNelems(),
                                             statusPtr, adaptor.getCmp(),
                                             adaptor.getCmpValue()});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// WaitUntilSomeOp Lowering
//===----------------------------------------------------------------------===//

struct WaitUntilSomeOpLowering
    : public ConvertOpToLLVMPattern<openshmem::WaitUntilSomeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::WaitUntilSomeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // size_t shmem_wait_until_some(TYPE *ivars, size_t nelems, size_t *indices,
    //                              const int *status, int cmp, TYPE cmp_value)
    Type sizeType = getTypeConverter()->getIndexType();
    Type cmpValueType = adaptor.getCmpValue().getType();
    auto funcType = LLVM::LLVMFunctionType::get(
        sizeType, {ptrType, sizeType, ptrType, ptrType, rewriter.getI32Type(),
                   cmpValueType});

    std::string funcName =
        getPt2ptSyncSizedFunctionName("wait_until_some", cmpValueType);
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivars is a symmetric memref (already converted to pointer)
    Value ivarsPtr = adaptor.getIvars();
    // indices and status are regular memrefs (need to extract data pointers)
    Value indicesPtr = getMemRefDataPtr(loc, rewriter, adaptor.getIndices());
    Value statusPtr = getMemRefDataPtr(loc, rewriter, adaptor.getStatus());

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ivarsPtr, adaptor.getNelems(), indicesPtr, statusPtr,
                   adaptor.getCmp(), adaptor.getCmpValue()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// WaitUntilAllVectorOp Lowering
//===----------------------------------------------------------------------===//

struct WaitUntilAllVectorOpLowering
    : public ConvertOpToLLVMPattern<openshmem::WaitUntilAllVectorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::WaitUntilAllVectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void shmem_wait_until_all_vector(TYPE *ivars, size_t nelems,
    //                                  const int *status, int cmp,
    //                                  TYPE *cmp_values)
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, sizeType, ptrType, rewriter.getI32Type(), ptrType});

    // Vector operations use simple naming without size suffixes
    std::string funcName =
        getPt2ptSyncVectorFunctionName("wait_until_all_vector");
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivars is a symmetric memref (already converted to pointer)
    Value ivarsPtr = adaptor.getIvars();
    // status and cmp_values are regular memrefs (need to extract data pointers)
    Value statusPtr = getMemRefDataPtr(loc, rewriter, adaptor.getStatus());
    Value cmpValuesPtr =
        getMemRefDataPtr(loc, rewriter, adaptor.getCmpValues());

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{ivarsPtr, adaptor.getNelems(),
                                             statusPtr, adaptor.getCmp(),
                                             cmpValuesPtr});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// WaitUntilAnyVectorOp Lowering
//===----------------------------------------------------------------------===//

struct WaitUntilAnyVectorOpLowering
    : public ConvertOpToLLVMPattern<openshmem::WaitUntilAnyVectorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::WaitUntilAnyVectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // void shmem_wait_until_any_vector(TYPE *ivars, size_t nelems,
    //                                  const int *status, int cmp,
    //                                  TYPE *cmp_values)
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()),
        {ptrType, sizeType, ptrType, rewriter.getI32Type(), ptrType});

    // Vector operations use simple naming without size suffixes
    std::string funcName =
        getPt2ptSyncVectorFunctionName("wait_until_any_vector");
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivars is a symmetric memref (already converted to pointer)
    Value ivarsPtr = adaptor.getIvars();
    // status and cmp_values are regular memrefs (need to extract data pointers)
    Value statusPtr = getMemRefDataPtr(loc, rewriter, adaptor.getStatus());
    Value cmpValuesPtr =
        getMemRefDataPtr(loc, rewriter, adaptor.getCmpValues());

    rewriter.create<LLVM::CallOp>(loc, funcDecl,
                                  ValueRange{ivarsPtr, adaptor.getNelems(),
                                             statusPtr, adaptor.getCmp(),
                                             cmpValuesPtr});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// WaitUntilSomeVectorOp Lowering
//===----------------------------------------------------------------------===//

struct WaitUntilSomeVectorOpLowering
    : public ConvertOpToLLVMPattern<openshmem::WaitUntilSomeVectorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::WaitUntilSomeVectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // size_t shmem_wait_until_some_vector(TYPE *ivars, size_t nelems,
    //                                     size_t *indices, const int *status,
    //                                     int cmp, TYPE *cmp_values)
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        sizeType,
        {ptrType, sizeType, ptrType, ptrType, rewriter.getI32Type(), ptrType});

    // Vector operations use simple naming without size suffixes
    std::string funcName =
        getPt2ptSyncVectorFunctionName("wait_until_some_vector");
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivars is a symmetric memref (already converted to pointer)
    Value ivarsPtr = adaptor.getIvars();
    // indices, status, and cmp_values are regular memrefs (need to extract data
    // pointers)
    Value indicesPtr = getMemRefDataPtr(loc, rewriter, adaptor.getIndices());
    Value statusPtr = getMemRefDataPtr(loc, rewriter, adaptor.getStatus());
    Value cmpValuesPtr =
        getMemRefDataPtr(loc, rewriter, adaptor.getCmpValues());

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ivarsPtr, adaptor.getNelems(), indicesPtr, statusPtr,
                   adaptor.getCmp(), cmpValuesPtr});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TestOp Lowering
//===----------------------------------------------------------------------===//

struct TestOpLowering : public ConvertOpToLLVMPattern<openshmem::TestOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TestOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // int shmem_test(TYPE *ivar, int cmp, TYPE cmp_value)
    Type cmpValueType = adaptor.getCmpValue().getType();
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, rewriter.getI32Type(), cmpValueType});

    std::string funcName = getPt2ptSyncSizedFunctionName("test", cmpValueType);
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivar is a symmetric memref (already converted to pointer)
    Value ivarPtr = adaptor.getIvar();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ivarPtr, adaptor.getCmp(), adaptor.getCmpValue()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TestAllOp Lowering
//===----------------------------------------------------------------------===//

struct TestAllOpLowering : public ConvertOpToLLVMPattern<openshmem::TestAllOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TestAllOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // int shmem_test_all(TYPE *ivars, size_t nelems, const int *status,
    //                    int cmp, TYPE cmp_value)
    Type sizeType = getTypeConverter()->getIndexType();
    Type cmpValueType = adaptor.getCmpValue().getType();
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(),
        {ptrType, sizeType, ptrType, rewriter.getI32Type(), cmpValueType});

    std::string funcName =
        getPt2ptSyncSizedFunctionName("test_all", cmpValueType);
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivars is a symmetric memref (already converted to pointer)
    Value ivarsPtr = adaptor.getIvars();
    Value statusPtr = getMemRefDataPtr(loc, rewriter, adaptor.getStatus());

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ivarsPtr, adaptor.getNelems(), statusPtr, adaptor.getCmp(),
                   adaptor.getCmpValue()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TestAnyOp Lowering
//===----------------------------------------------------------------------===//

struct TestAnyOpLowering : public ConvertOpToLLVMPattern<openshmem::TestAnyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TestAnyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // size_t shmem_test_any(TYPE *ivars, size_t nelems, const int *status,
    //                       int cmp, TYPE cmp_value)
    Type sizeType = getTypeConverter()->getIndexType();
    Type cmpValueType = adaptor.getCmpValue().getType();
    auto funcType = LLVM::LLVMFunctionType::get(
        sizeType,
        {ptrType, sizeType, ptrType, rewriter.getI32Type(), cmpValueType});

    std::string funcName =
        getPt2ptSyncSizedFunctionName("test_any", cmpValueType);
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivars is a symmetric memref (already converted to pointer)
    Value ivarsPtr = adaptor.getIvars();
    Value statusPtr = getMemRefDataPtr(loc, rewriter, adaptor.getStatus());

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ivarsPtr, adaptor.getNelems(), statusPtr, adaptor.getCmp(),
                   adaptor.getCmpValue()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TestSomeOp Lowering
//===----------------------------------------------------------------------===//

struct TestSomeOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TestSomeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TestSomeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // size_t shmem_test_some(TYPE *ivars, size_t nelems, size_t *indices,
    //                        const int *status, int cmp, TYPE cmp_value)
    Type sizeType = getTypeConverter()->getIndexType();
    Type cmpValueType = adaptor.getCmpValue().getType();
    auto funcType = LLVM::LLVMFunctionType::get(
        sizeType, {ptrType, sizeType, ptrType, ptrType, rewriter.getI32Type(),
                   cmpValueType});

    std::string funcName =
        getPt2ptSyncSizedFunctionName("test_some", cmpValueType);
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivars is a symmetric memref (already converted to pointer)
    Value ivarsPtr = adaptor.getIvars();
    // indices and status are regular memrefs (need to extract data pointers)
    Value indicesPtr = getMemRefDataPtr(loc, rewriter, adaptor.getIndices());
    Value statusPtr = getMemRefDataPtr(loc, rewriter, adaptor.getStatus());

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ivarsPtr, adaptor.getNelems(), indicesPtr, statusPtr,
                   adaptor.getCmp(), adaptor.getCmpValue()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TestAllVectorOp Lowering
//===----------------------------------------------------------------------===//

struct TestAllVectorOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TestAllVectorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TestAllVectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // int shmem_test_all_vector(TYPE *ivars, size_t nelems, const int *status,
    //                           int cmp, TYPE *cmp_values)
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(),
        {ptrType, sizeType, ptrType, rewriter.getI32Type(), ptrType});

    // Vector operations use simple naming without size suffixes
    std::string funcName = getPt2ptSyncVectorFunctionName("test_all_vector");
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivars is a symmetric memref (already converted to pointer)
    Value ivarsPtr = adaptor.getIvars();
    // status and cmp_values are regular memrefs (need to extract data pointers)
    Value statusPtr = getMemRefDataPtr(loc, rewriter, adaptor.getStatus());
    Value cmpValuesPtr =
        getMemRefDataPtr(loc, rewriter, adaptor.getCmpValues());

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ivarsPtr, adaptor.getNelems(), statusPtr, adaptor.getCmp(),
                   cmpValuesPtr});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TestAnyVectorOp Lowering
//===----------------------------------------------------------------------===//

struct TestAnyVectorOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TestAnyVectorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TestAnyVectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // size_t shmem_test_any_vector(TYPE *ivars, size_t nelems, const int
    // *status,
    //                              int cmp, TYPE *cmp_values)
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        sizeType, {ptrType, sizeType, ptrType, rewriter.getI32Type(), ptrType});

    // Vector operations use simple naming without size suffixes
    std::string funcName = getPt2ptSyncVectorFunctionName("test_any_vector");
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivars is a symmetric memref (already converted to pointer)
    Value ivarsPtr = adaptor.getIvars();
    // status and cmp_values are regular memrefs (need to extract data pointers)
    Value statusPtr = getMemRefDataPtr(loc, rewriter, adaptor.getStatus());
    Value cmpValuesPtr =
        getMemRefDataPtr(loc, rewriter, adaptor.getCmpValues());

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ivarsPtr, adaptor.getNelems(), statusPtr, adaptor.getCmp(),
                   cmpValuesPtr});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TestSomeVectorOp Lowering
//===----------------------------------------------------------------------===//

struct TestSomeVectorOpLowering
    : public ConvertOpToLLVMPattern<openshmem::TestSomeVectorOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::TestSomeVectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // size_t shmem_test_some_vector(TYPE *ivars, size_t nelems, size_t
    // *indices,
    //                               const int *status, int cmp, TYPE
    //                               *cmp_values)
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        sizeType,
        {ptrType, sizeType, ptrType, ptrType, rewriter.getI32Type(), ptrType});

    // Vector operations use simple naming without size suffixes
    std::string funcName = getPt2ptSyncVectorFunctionName("test_some_vector");
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // ivars is a symmetric memref (already converted to pointer)
    Value ivarsPtr = adaptor.getIvars();
    // indices, status, and cmp_values are regular memrefs (need to extract data
    // pointers)
    Value indicesPtr = getMemRefDataPtr(loc, rewriter, adaptor.getIndices());
    Value statusPtr = getMemRefDataPtr(loc, rewriter, adaptor.getStatus());
    Value cmpValuesPtr =
        getMemRefDataPtr(loc, rewriter, adaptor.getCmpValues());

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{ivarsPtr, adaptor.getNelems(), indicesPtr, statusPtr,
                   adaptor.getCmp(), cmpValuesPtr});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SignalWaitUntilOp Lowering
//===----------------------------------------------------------------------===//

struct SignalWaitUntilOpLowering
    : public ConvertOpToLLVMPattern<openshmem::SignalWaitUntilOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::SignalWaitUntilOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // uint64_t shmem_signal_wait_until(uint64_t *sig_addr, int cmp,
    //                                  uint64_t cmp_value)
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI64Type(),
        {ptrType, rewriter.getI32Type(), rewriter.getI64Type()});

    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_signal_wait_until", funcType);

    // sig_addr is a symmetric memref (already converted to pointer)
    Value sigAddrPtr = adaptor.getSigAddr();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{sigAddrPtr, adaptor.getCmp(), adaptor.getCmpValue()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

} // namespace

void openshmem::populatePt2ptSyncOpsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {

  patterns
      .add<WaitUntilOpLowering, WaitUntilAllOpLowering, WaitUntilAnyOpLowering,
           WaitUntilSomeOpLowering, WaitUntilAllVectorOpLowering,
           WaitUntilAnyVectorOpLowering, WaitUntilSomeVectorOpLowering,
           TestOpLowering, TestAllOpLowering, TestAnyOpLowering,
           TestSomeOpLowering, TestAllVectorOpLowering, TestAnyVectorOpLowering,
           TestSomeVectorOpLowering, SignalWaitUntilOpLowering>(converter);
}

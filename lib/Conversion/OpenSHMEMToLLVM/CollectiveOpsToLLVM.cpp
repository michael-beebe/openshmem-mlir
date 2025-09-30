//===- CollectiveOpsToLLVM.cpp - Collective operations conversion patterns ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CollectiveOpsToLLVM.h"
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

// Helper function to get OpenSHMEM type suffix from MLIR type
static std::string getOpenSHMEMTypeSuffix(Type type) {
  if (type.isInteger(8))
    return "uchar"; // or "char" depending on signedness
  if (type.isInteger(16))
    return "short";
  if (type.isInteger(32))
    return "int";
  if (type.isInteger(64))
    return "long";
  if (type.isF32())
    return "float";
  if (type.isF64())
    return "double";
  if (type.isBF16() || type.isF16())
    return "float"; // Promote to float

  // Default to using the memory-based version for unsupported types
  return "";
}

//===----------------------------------------------------------------------===//
// AlltoallOp Lowering (Typed)
//===----------------------------------------------------------------------===//

struct AlltoallOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AlltoallOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::AlltoallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // Get the element type from the memref with symmetric memory space
    auto destType = cast<MemRefType>(op.getDest().getType());
    auto elementType = destType.getElementType();

    // Generate function name based on element type
    std::string typeSuffix = getOpenSHMEMTypeSuffix(elementType);
    std::string funcName;
    if (typeSuffix.empty()) {
      // Fall back to memory-based version for unsupported types
      funcName = "shmem_alltoallmem";
    } else {
      funcName = "shmem_" + typeSuffix + "_alltoall";
    }

    // int shmem_<type>_alltoall(shmem_team_t team, TYPE *dest, const TYPE
    // *source, size_t nelems)
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, adaptor.getNelems()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AlltoallsOp Lowering (Typed Strided)
//===----------------------------------------------------------------------===//

struct AlltoallsOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AlltoallsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::AlltoallsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // Get the element type from the memref with symmetric memory space
    auto destType = cast<MemRefType>(op.getDest().getType());
    auto elementType = destType.getElementType();

    // Generate function name based on element type
    std::string typeSuffix = getOpenSHMEMTypeSuffix(elementType);
    std::string funcName;
    if (typeSuffix.empty()) {
      // Fall back to memory-based version for unsupported types
      funcName = "shmem_alltoallsmem";
    } else {
      funcName = "shmem_" + typeSuffix + "_alltoalls";
    }

    // int shmem_<type>_alltoalls(shmem_team_t team, TYPE *dest, const TYPE
    // *source,
    //                            ptrdiff_t dst, ptrdiff_t sst, size_t nelems);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(),
        {ptrType, ptrType, ptrType, sizeType, sizeType, sizeType});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value dst = adaptor.getDst();
    Value sst = adaptor.getSst();
    Value nelems = adaptor.getNelems();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, dst, sst, nelems});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BroadcastOp Lowering (Typed)
//===----------------------------------------------------------------------===//

struct BroadcastOpLowering
    : public ConvertOpToLLVMPattern<openshmem::BroadcastOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // Get the element type from the memref with symmetric memory space
    auto destType = cast<MemRefType>(op.getDest().getType());
    auto elementType = destType.getElementType();

    // Generate function name based on element type
    std::string typeSuffix = getOpenSHMEMTypeSuffix(elementType);
    std::string funcName;
    if (typeSuffix.empty()) {
      // Fall back to memory-based version for unsupported types
      funcName = "shmem_broadcastmem";
    } else {
      funcName = "shmem_" + typeSuffix + "_broadcast";
    }

    // int shmem_<type>_broadcast(shmem_team_t team, TYPE *dest, const TYPE
    // *source,
    //                            size_t nelems, int PE_root);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(),
        {ptrType, ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nelems = adaptor.getNelems();
    Value peRoot = adaptor.getPERoot();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nelems, peRoot});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CollectOp Lowering (Typed)
//===----------------------------------------------------------------------===//

struct CollectOpLowering : public ConvertOpToLLVMPattern<openshmem::CollectOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::CollectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // Get the element type from the memref with symmetric memory space
    auto destType = cast<MemRefType>(op.getDest().getType());
    auto elementType = destType.getElementType();

    // Generate function name based on element type
    std::string typeSuffix = getOpenSHMEMTypeSuffix(elementType);
    std::string funcName;
    if (typeSuffix.empty()) {
      // Fall back to memory-based version for unsupported types
      funcName = "shmem_collectmem";
    } else {
      funcName = "shmem_" + typeSuffix + "_collect";
    }

    // int shmem_<type>_collect(shmem_team_t team, TYPE *dest, const TYPE
    // *source, size_t nelems);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nelems = adaptor.getNelems();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nelems});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FCollectOp Lowering (Typed)
//===----------------------------------------------------------------------===//

struct FCollectOpLowering
    : public ConvertOpToLLVMPattern<openshmem::FCollectOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::FCollectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // Get the element type from the memref with symmetric memory space
    auto destType = cast<MemRefType>(op.getDest().getType());
    auto elementType = destType.getElementType();

    // Generate function name based on element type
    std::string typeSuffix = getOpenSHMEMTypeSuffix(elementType);
    std::string funcName;
    if (typeSuffix.empty()) {
      // Fall back to memory-based version for unsupported types
      funcName = "shmem_fcollectmem";
    } else {
      funcName = "shmem_" + typeSuffix + "_fcollect";
    }

    // int shmem_<type>_fcollect(shmem_team_t team, TYPE *dest, const TYPE
    // *source, size_t nelems);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl =
        getOrDefineFunction(moduleOp, loc, rewriter, funcName, funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nelems = adaptor.getNelems();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nelems});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AlltoallmemOp Lowering
//===----------------------------------------------------------------------===//

struct AlltoallmemOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AlltoallmemOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::AlltoallmemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // int shmem_alltoallmem(shmem_team_t team, void *dest, const void *source,
    // size_t nelems) size_t is typically the same as index type on the target
    // platform
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_alltoallmem", funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, adaptor.getNelems()});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AlltoallsmemOp Lowering
//===----------------------------------------------------------------------===//

struct AlltoallsmemOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AlltoallsmemOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::AlltoallsmemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // int shmem_alltoallsmem(shmem_team_t team, void *dest, const void *source,
    //                        ptrdiff_t dst, ptrdiff_t sst, size_t nelems);
    Type sizeType = getTypeConverter()->getIndexType();
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(),
        {ptrType, ptrType, ptrType, sizeType, sizeType, sizeType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_alltoallsmem", funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value dst = adaptor.getDst();
    Value sst = adaptor.getSst();
    Value nelems = adaptor.getNelems();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, dst, sst, nelems});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BroadcastmemOp Lowering
//===----------------------------------------------------------------------===//

struct BroadcastmemOpLowering
    : public ConvertOpToLLVMPattern<openshmem::BroadcastmemOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::BroadcastmemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // int shmem_broadcastmem(shmem_team_t team, void *dest, const void *source,
    //                        size_t nelems, int PE_root);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(),
        {ptrType, ptrType, ptrType, sizeType, rewriter.getI32Type()});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_broadcastmem", funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nelems = adaptor.getNelems();
    Value peRoot = adaptor.getPERoot();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nelems, peRoot});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CollectmemOp Lowering
//===----------------------------------------------------------------------===//

struct CollectmemOpLowering
    : public ConvertOpToLLVMPattern<openshmem::CollectmemOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::CollectmemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // int shmem_collectmem(shmem_team_t team, void *dest, const void *source,
    // size_t nelems);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_collectmem", funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nelems = adaptor.getNelems();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nelems});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FCollectmemOp Lowering
//===----------------------------------------------------------------------===//

struct FCollectmemOpLowering
    : public ConvertOpToLLVMPattern<openshmem::FCollectmemOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::FCollectmemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // int shmem_fcollectmem(shmem_team_t team, void *dest, const void *source,
    // size_t nelems);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_fcollectmem", funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nelems = adaptor.getNelems();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nelems});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AndReduceOp Lowering
//===----------------------------------------------------------------------===//

struct AndReduceOpLowering
    : public ConvertOpToLLVMPattern<openshmem::AndReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::AndReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // int shmem_and_reduce(shmem_team_t team, void *dest, const void *source,
    // size_t nreduce);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_and_reduce", funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nreduce = adaptor.getNreduce();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nreduce});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// OrReduceOp Lowering
//===----------------------------------------------------------------------===//

struct OrReduceOpLowering
    : public ConvertOpToLLVMPattern<openshmem::OrReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::OrReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // int shmem_orreduce(shmem_team_t team, void *dest, const void *source,
    // size_t nreduce);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_or_reduce", funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nreduce = adaptor.getNreduce();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nreduce});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// XorReduceOp Lowering
//===----------------------------------------------------------------------===//

struct XorReduceOpLowering
    : public ConvertOpToLLVMPattern<openshmem::XorReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::XorReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // int shmem_xor_reduce(shmem_team_t team, void *dest, const void *source,
    // size_t nreduce);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_xor_reduce", funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nreduce = adaptor.getNreduce();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nreduce});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MaxReduceOp Lowering
//===----------------------------------------------------------------------===//

struct MaxReduceOpLowering
    : public ConvertOpToLLVMPattern<openshmem::MaxReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::MaxReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // int shmem_max_reduce(shmem_team_t team, void *dest, const void *source,
    // size_t nreduce);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_max_reduce", funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nreduce = adaptor.getNreduce();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nreduce});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MinReduceOp Lowering
//===----------------------------------------------------------------------===//

struct MinReduceOpLowering
    : public ConvertOpToLLVMPattern<openshmem::MinReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::MinReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // int shmem_min_reduce(shmem_team_t team, void *dest, const void *source,
    // size_t nreduce);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_min_reduce", funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nreduce = adaptor.getNreduce();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nreduce});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SumReduceOp Lowering
//===----------------------------------------------------------------------===//

struct SumReduceOpLowering
    : public ConvertOpToLLVMPattern<openshmem::SumReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::SumReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // int shmem_sum_reduce(shmem_team_t team, void *dest, const void *source,
    // size_t nreduce);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_sum_reduce", funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nreduce = adaptor.getNreduce();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nreduce});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ProdReduceOp Lowering
//===----------------------------------------------------------------------===//

struct ProdReduceOpLowering
    : public ConvertOpToLLVMPattern<openshmem::ProdReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(openshmem::ProdReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type sizeType = getTypeConverter()->getIndexType();

    // int shmem_prod_reduce(shmem_team_t team, void *dest, const void *source,
    // size_t nreduce);
    auto funcType = LLVM::LLVMFunctionType::get(
        rewriter.getI32Type(), {ptrType, ptrType, ptrType, sizeType});
    LLVM::LLVMFuncOp funcDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "shmem_prod_reduce", funcType);

    // dest and source are already pointers (memref with symmetric memory space converts to
    // pointer)
    Value destPtr = adaptor.getDest();
    Value sourcePtr = adaptor.getSource();
    Value nreduce = adaptor.getNreduce();

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, funcDecl,
        ValueRange{adaptor.getTeam(), destPtr, sourcePtr, nreduce});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void openshmem::populateCollectiveOpsToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns
      .add<AlltoallmemOpLowering, AlltoallsmemOpLowering,
           BroadcastmemOpLowering, CollectmemOpLowering, FCollectmemOpLowering,
           AlltoallOpLowering, AndReduceOpLowering, OrReduceOpLowering,
           XorReduceOpLowering, MaxReduceOpLowering, MinReduceOpLowering,
           SumReduceOpLowering, ProdReduceOpLowering, AlltoallsOpLowering,
           BroadcastOpLowering, CollectOpLowering, FCollectOpLowering>(
          converter);
}

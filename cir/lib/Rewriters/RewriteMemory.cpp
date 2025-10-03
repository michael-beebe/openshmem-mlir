//===- RewriteMemory.cpp - CIR to OpenSHMEM Memory Rewriters -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Patterns for converting CIR memory-related OpenSHMEM calls (malloc/free)
// to the OpenSHMEM dialect.
//
//===----------------------------------------------------------------------===//

#include "OpenSHMEMCIR/OpenSHMEMCIR.h"
#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace mlir::openshmem;

namespace mlir {
namespace openshmem {
namespace cir {

/// Pattern to convert shmem_malloc() calls to openshmem.malloc
struct ConvertShmemMallocPattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_malloc")
      return failure();

    if (callOp.getArgOperands().size() != 1)
      return failure();

    Value sizeArg = callOp.getArgOperands()[0];

    // Convert size argument to index type using unrealized conversion cast
    auto indexType = rewriter.getIndexType();
    auto convertedSize = rewriter
                             .create<UnrealizedConversionCastOp>(
                                 callOp.getLoc(), indexType, sizeArg)
                             .getResult(0);

    // Create symmetric memory space attribute
    auto symmetricMemSpace =
        openshmem::SymmetricMemorySpaceAttr::get(rewriter.getContext());

    // Create memref type with symmetric memory space
    auto elementType = rewriter.getI8Type();
    auto memRefType =
        MemRefType::get({ShapedType::kDynamic}, elementType,
                        MemRefLayoutAttrInterface{}, symmetricMemSpace);

    auto mallocOp = rewriter.create<openshmem::MallocOp>(
        callOp.getLoc(), memRefType, convertedSize);

    // Cast the result back to a CIR pointer type to match the original return
    // type
    auto resultCast = rewriter.create<UnrealizedConversionCastOp>(
        callOp.getLoc(), callOp.getResultTypes(), mallocOp.getResult());

    rewriter.replaceOp(callOp, resultCast.getResult(0));
    return success();
  }
};

/// Pattern to convert shmem_free() calls to openshmem.free
struct ConvertShmemFreePattern : public OpRewritePattern<::cir::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || callee.value() != "shmem_free")
      return failure();

    if (callOp.getArgOperands().size() != 1)
      return failure();

    Value ptrArg = callOp.getArgOperands()[0];

    // Create symmetric memory space attribute for the type conversion
    auto symmetricMemSpace =
        openshmem::SymmetricMemorySpaceAttr::get(rewriter.getContext());

    // Create memref type with symmetric memory space for the conversion
    auto elementType = rewriter.getI8Type();
    auto memRefType =
        MemRefType::get({ShapedType::kDynamic}, elementType,
                        MemRefLayoutAttrInterface{}, symmetricMemSpace);

    // Cast the CIR pointer to memref type for the free operation
    auto convertedPtr = rewriter
                            .create<UnrealizedConversionCastOp>(
                                callOp.getLoc(), memRefType, ptrArg)
                            .getResult(0);

    rewriter.replaceOpWithNewOp<openshmem::FreeOp>(callOp, convertedPtr);
    return success();
  }
};

// Pattern registration for memory patterns
void populateCIRToOpenSHMEMMemoryPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertShmemMallocPattern, ConvertShmemFreePattern>(
      patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

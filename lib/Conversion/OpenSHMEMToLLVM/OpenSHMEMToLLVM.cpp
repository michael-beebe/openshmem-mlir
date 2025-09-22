//===- OpenSHMEMToLLVM.cpp - OpenSHMEM to LLVM conversion ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenSHMEMToLLVM/OpenSHMEMToLLVM.h"
#include "AtomicOpsToLLVM.h"
#include "CollectiveOpsToLLVM.h"
#include "ContextOpsToLLVM.h"
#include "MemoryOpsToLLVM.h"
#include "OpenSHMEMConversionUtils.h"
#include "Pt2ptSyncOpsToLLVM.h"
#include "RMAOpsToLLVM.h"
#include "SetupOpsToLLVM.h"
#include "SyncOpsToLLVM.h"
#include "TeamOpsToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DEF_CONVERTOPENSHMEMTOLLVMPASS
#include "mlir/Conversion/OpenSHMEMToLLVM/OpenSHMEMConvertPasses.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::openshmem;

namespace {} // namespace

//===----------------------------------------------------------------------===//
// Pass and conversion setup
//===----------------------------------------------------------------------===//

struct ConvertOpenSHMEMToLLVMPass
    : public impl::ConvertOpenSHMEMToLLVMPassBase<ConvertOpenSHMEMToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, openshmem::OpenSHMEMDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext());

    // Configure conversion target
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<openshmem::OpenSHMEMDialect>();

    // Populate conversion patterns
    openshmem::populateOpenSHMEMToLLVMConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

/// Implement the interface to convert OpenSHMEM to LLVM.
struct OpenSHMEMToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    openshmem::populateOpenSHMEMToLLVMConversionPatterns(typeConverter,
                                                         patterns);
  }
};

//===----------------------------------------------------------------------===//
// Custom memref conversion patterns for pointer-based approach
//===----------------------------------------------------------------------===//

struct MemRefAllocOpLowering : public ConvertOpToLLVMPattern<memref::AllocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // For our pointer-based approach, we just allocate memory and return a
    // pointer This is a simplified approach that works with OpenSHMEM
    // operations
    Value size = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(4)); // Assume 4 bytes for now
    Value alloc = rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType, size);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

struct MemRefDeallocOpLowering
    : public ConvertOpToLLVMPattern<memref::DeallocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For our pointer-based approach, we don't need to do anything special
    // The memory will be freed when the function returns
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern population and pass creation
//===----------------------------------------------------------------------===//

void openshmem::populateOpenSHMEMToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {

  // Add type conversions for OpenSHMEM types
  // Note: OpenSHMEM_Retval has been removed and replaced with I32

  // Add conversion for memref types
  converter.addConversion([](MemRefType type) -> Type {
    // Check if this is a memref with symmetric memory space
    if (type.getMemorySpace() &&
        llvm::isa<openshmem::SymmetricMemorySpaceAttr>(type.getMemorySpace())) {
      // For symmetric memref types, convert to LLVM pointer type
      // The element type information is lost, but this matches the original
      // symmetric_memref behavior where we only had a pointer
      return LLVM::LLVMPointerType::get(type.getContext());
    }
    // For regular memref types, convert to LLVM pointer type as well
    // This is needed for OpenSHMEM operations that use AnyMemRef
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  converter.addConversion([](openshmem::TeamType type) -> Type {
    // Convert team to LLVM pointer type (shmem_team_t is typically a pointer)
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  converter.addConversion([](openshmem::CtxType type) -> Type {
    // Convert ctx to LLVM pointer type (shmem_ctx_t is typically a pointer)
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  // Add custom memref-to-LLVM conversion patterns that work with our
  // pointer-based approach
  patterns.add<MemRefAllocOpLowering>(converter);
  patterns.add<MemRefDeallocOpLowering>(converter);

  // Populate OpenSHMEM-specific patterns
  populateSetupOpsToLLVMConversionPatterns(converter, patterns);
  populateMemoryOpsToLLVMConversionPatterns(converter, patterns);
  populateRMAOpsToLLVMConversionPatterns(converter, patterns);
  populateCollectiveOpsToLLVMConversionPatterns(converter, patterns);
  populateTeamOpsToLLVMConversionPatterns(converter, patterns);
  populateContextOpsToLLVMConversionPatterns(converter, patterns);
  populateAtomicOpsToLLVMConversionPatterns(converter, patterns);
  populateSyncOpsToLLVMConversionPatterns(converter, patterns);
  populatePt2ptSyncOpsToLLVMConversionPatterns(converter, patterns);
}

void openshmem::registerConvertOpenSHMEMToLLVMInterface(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, openshmem::OpenSHMEMDialect *dialect) {
        dialect->addInterfaces<OpenSHMEMToLLVMDialectInterface>();
      });
}

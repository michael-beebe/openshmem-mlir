#include "OpenSHMEMCIR/OpenSHMEMCIR.h"
#include "Utils.h"

#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace mlir::openshmem;

namespace mlir {
namespace openshmem {
namespace cir {

// Helper function to create symmetric memref type
static MemRefType createSymmetricMemRefType(MLIRContext *ctx) {
  auto symmetricMemSpace = openshmem::SymmetricMemorySpaceAttr::get(ctx);
  auto elementType = IntegerType::get(ctx, 8);
  return MemRefType::get({ShapedType::kDynamic}, elementType,
                         MemRefLayoutAttrInterface{}, symmetricMemSpace);
}

// Helper function to convert CIR pointer to memref
static Value convertPtrToMemRef(Value ptr, Location loc,
                                PatternRewriter &rewriter) {
  auto memRefType = createSymmetricMemRefType(rewriter.getContext());
  return rewriter.create<UnrealizedConversionCastOp>(loc, memRefType, ptr)
      .getResult(0);
}

// Helper function to convert CIR int to index
static Value convertToIndex(Value val, Location loc,
                            PatternRewriter &rewriter) {
  auto indexType = rewriter.getIndexType();
  return rewriter.create<UnrealizedConversionCastOp>(loc, indexType, val)
      .getResult(0);
}

// Helper function to convert CIR int to i32
static Value convertToI32(Value val, Location loc,
                          PatternRewriter &rewriter) {
  auto i32Type = rewriter.getI32Type();
  return rewriter.create<UnrealizedConversionCastOp>(loc, i32Type, val)
      .getResult(0);
}

// Helper function to convert CIR pointer to context type
static Value convertToCtx(Value val, Location loc,
                         PatternRewriter &rewriter) {
  auto ctxType = openshmem::CtxType::get(rewriter.getContext());
  return rewriter.create<UnrealizedConversionCastOp>(loc, ctxType, val)
      .getResult(0);
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

//===----------------------------------------------------------------------===//
// CIR RMA Patterns
//===----------------------------------------------------------------------===//

// Generic Put operation
class ConvertPutPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_put")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    // Convert operands to appropriate types
    auto dest =
        openshmem::cir::convertPtrToMemRef(operands[0], op.getLoc(), rewriter);
    auto src =
        openshmem::cir::convertPtrToMemRef(operands[1], op.getLoc(), rewriter);
    auto nelems =
        openshmem::cir::convertToIndex(operands[2], op.getLoc(), rewriter);
    auto pe = openshmem::cir::convertToI32(operands[3], op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<openshmem::PutOp>(op, dest, src, nelems, pe);
    return success();
  }
};

// Non-blocking Put operation
class ConvertPutNbiPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_put_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto dest =
        openshmem::cir::convertPtrToMemRef(operands[0], op.getLoc(), rewriter);
    auto src =
        openshmem::cir::convertPtrToMemRef(operands[1], op.getLoc(), rewriter);
    auto nelems =
        openshmem::cir::convertToIndex(operands[2], op.getLoc(), rewriter);
    auto pe = openshmem::cir::convertToI32(operands[3], op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<openshmem::PutNbiOp>(op, dest, src, nelems, pe);
    return success();
  }
};

// Generic Get operation
class ConvertGetPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_get")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto dest =
        openshmem::cir::convertPtrToMemRef(operands[0], op.getLoc(), rewriter);
    auto src =
        openshmem::cir::convertPtrToMemRef(operands[1], op.getLoc(), rewriter);
    auto nelems =
        openshmem::cir::convertToIndex(operands[2], op.getLoc(), rewriter);
    auto pe = openshmem::cir::convertToI32(operands[3], op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<openshmem::GetOp>(op, dest, src, nelems, pe);
    return success();
  }
};

// Non-blocking Get operation
class ConvertGetNbiPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_get_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto dest =
        openshmem::cir::convertPtrToMemRef(operands[0], op.getLoc(), rewriter);
    auto src =
        openshmem::cir::convertPtrToMemRef(operands[1], op.getLoc(), rewriter);
    auto nelems =
        openshmem::cir::convertToIndex(operands[2], op.getLoc(), rewriter);
    auto pe = openshmem::cir::convertToI32(operands[3], op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<openshmem::GetNbiOp>(op, dest, src, nelems, pe);
    return success();
  }
};

// Putmem operation (byte-level)
class ConvertPutmemPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_putmem")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto dest =
        openshmem::cir::convertPtrToMemRef(operands[0], op.getLoc(), rewriter);
    auto size =
        openshmem::cir::convertToIndex(operands[2], op.getLoc(), rewriter);
    auto pe = openshmem::cir::convertToI32(operands[3], op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<openshmem::PutmemOp>(op, dest, operands[1],
                                                     size, pe);
    return success();
  }
};

// Non-blocking Putmem operation
class ConvertPutmemNbiPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_putmem_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto dest =
        openshmem::cir::convertPtrToMemRef(operands[0], op.getLoc(), rewriter);
    auto nelems =
        openshmem::cir::convertToIndex(operands[2], op.getLoc(), rewriter);
    auto pe = openshmem::cir::convertToI32(operands[3], op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<openshmem::PutmemNbiOp>(op, dest, operands[1],
                                                        nelems, pe);
    return success();
  }
};

// Getmem operation (byte-level)
class ConvertGetmemPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_getmem")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto src =
        openshmem::cir::convertPtrToMemRef(operands[1], op.getLoc(), rewriter);
    auto size =
        openshmem::cir::convertToIndex(operands[2], op.getLoc(), rewriter);
    auto pe = openshmem::cir::convertToI32(operands[3], op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<openshmem::GetmemOp>(op, operands[0], src, size,
                                                     pe);
    return success();
  }
};

// Non-blocking Getmem operation
class ConvertGetmemNbiPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_getmem_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    auto src =
        openshmem::cir::convertPtrToMemRef(operands[1], op.getLoc(), rewriter);
    auto nelems =
        openshmem::cir::convertToIndex(operands[2], op.getLoc(), rewriter);
    auto pe = openshmem::cir::convertToI32(operands[3], op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<openshmem::GetmemNbiOp>(op, operands[0], src,
                                                        nelems, pe);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Context-aware RMA Patterns
//===----------------------------------------------------------------------===//

// Context-aware Put operation
class ConvertCtxPutPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_ctx_put")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ctx = openshmem::cir::convertToCtx(operands[0], op.getLoc(), rewriter);
    auto dest =
        openshmem::cir::convertPtrToMemRef(operands[1], op.getLoc(), rewriter);
    auto src =
        openshmem::cir::convertPtrToMemRef(operands[2], op.getLoc(), rewriter);
    auto nelems =
        openshmem::cir::convertToIndex(operands[3], op.getLoc(), rewriter);
    auto pe = openshmem::cir::convertToI32(operands[4], op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<openshmem::CtxPutOp>(op, ctx, dest, src,
                                                     nelems, pe);
    return success();
  }
};

// Context-aware non-blocking Put operation
class ConvertCtxPutNbiPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_ctx_put_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ctx = openshmem::cir::convertToCtx(operands[0], op.getLoc(), rewriter);
    auto dest =
        openshmem::cir::convertPtrToMemRef(operands[1], op.getLoc(), rewriter);
    auto src =
        openshmem::cir::convertPtrToMemRef(operands[2], op.getLoc(), rewriter);
    auto nelems =
        openshmem::cir::convertToIndex(operands[3], op.getLoc(), rewriter);
    auto pe = openshmem::cir::convertToI32(operands[4], op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<openshmem::CtxPutNbiOp>(op, ctx, dest,
                                                        src, nelems, pe);
    return success();
  }
};

// Context-aware Get operation
class ConvertCtxGetPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_ctx_get")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ctx = openshmem::cir::convertToCtx(operands[0], op.getLoc(), rewriter);
    auto dest =
        openshmem::cir::convertPtrToMemRef(operands[1], op.getLoc(), rewriter);
    auto src =
        openshmem::cir::convertPtrToMemRef(operands[2], op.getLoc(), rewriter);
    auto nelems =
        openshmem::cir::convertToIndex(operands[3], op.getLoc(), rewriter);
    auto pe = openshmem::cir::convertToI32(operands[4], op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<openshmem::CtxGetOp>(op, ctx, dest, src,
                                                     nelems, pe);
    return success();
  }
};

// Context-aware non-blocking Get operation
class ConvertCtxGetNbiPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_ctx_get_nbi")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    auto ctx = openshmem::cir::convertToCtx(operands[0], op.getLoc(), rewriter);
    auto dest =
        openshmem::cir::convertPtrToMemRef(operands[1], op.getLoc(), rewriter);
    auto src =
        openshmem::cir::convertPtrToMemRef(operands[2], op.getLoc(), rewriter);
    auto nelems =
        openshmem::cir::convertToIndex(operands[3], op.getLoc(), rewriter);
    auto pe = openshmem::cir::convertToI32(operands[4], op.getLoc(), rewriter);

    rewriter.replaceOpWithNewOp<openshmem::CtxGetNbiOp>(op, ctx, dest,
                                                        src, nelems, pe);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

namespace mlir {
namespace openshmem {
namespace cir {

void populateCIRToOpenSHMEMRMAPatterns(RewritePatternSet &patterns) {
  // Basic RMA patterns
  patterns.add<ConvertPutPattern>(patterns.getContext());
  patterns.add<ConvertPutNbiPattern>(patterns.getContext());
  patterns.add<ConvertGetPattern>(patterns.getContext());
  patterns.add<ConvertGetNbiPattern>(patterns.getContext());
  
  // Byte-level RMA patterns
  patterns.add<ConvertPutmemPattern>(patterns.getContext());
  patterns.add<ConvertPutmemNbiPattern>(patterns.getContext());
  patterns.add<ConvertGetmemPattern>(patterns.getContext());
  patterns.add<ConvertGetmemNbiPattern>(patterns.getContext());
  
  // Context-aware RMA patterns
  patterns.add<ConvertCtxPutPattern>(patterns.getContext());
  patterns.add<ConvertCtxPutNbiPattern>(patterns.getContext());
  patterns.add<ConvertCtxGetPattern>(patterns.getContext());
  patterns.add<ConvertCtxGetNbiPattern>(patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

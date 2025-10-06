#include "OpenSHMEMCIR/OpenSHMEMCIR.h"
#include "Utils.h"

#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace mlir::openshmem;

//===----------------------------------------------------------------------===//
// CIR Collective Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BroadcastOp Pattern
//===----------------------------------------------------------------------===//
class ConvertBroadcastPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_broadcast")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 5)
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::BroadcastmemOp>(
        op, op.getResult().getType(), operands[0], operands[1], operands[2],
        operands[3], operands[4]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CollectOp Pattern
//===----------------------------------------------------------------------===//
class ConvertCollectPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_collect")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::CollectmemOp>(
        op, op.getResult().getType(), operands[0], operands[1], operands[2],
        operands[3]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FCollectOp Pattern
//===----------------------------------------------------------------------===//
class ConvertFCollectPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_fcollect")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::FCollectmemOp>(
        op, op.getResult().getType(), operands[0], operands[1], operands[2],
        operands[3]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AlltoallOp Pattern
//===----------------------------------------------------------------------===//
class ConvertAlltoallPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_alltoall")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::AlltoallmemOp>(
        op, op.getResult().getType(), operands[0], operands[1], operands[2],
        operands[3]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AlltoallsOp Pattern
//===----------------------------------------------------------------------===//
class ConvertAlltoallsPattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_alltoalls")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 6)
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::AlltoallsmemOp>(
        op, op.getResult().getType(), operands[0], operands[1], operands[2],
        operands[3], operands[4], operands[5]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Reduction Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SumReduceOp Pattern
//===----------------------------------------------------------------------===//
class ConvertSumReducePattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_sum_reduce")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::SumReduceOp>(
        op, op.getResult().getType(), operands[0], operands[1], operands[2],
        operands[3]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MaxReduceOp Pattern
//===----------------------------------------------------------------------===//
class ConvertMaxReducePattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_max_reduce")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::MaxReduceOp>(
        op, op.getResult().getType(), operands[0], operands[1], operands[2],
        operands[3]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MinReduceOp Pattern
//===----------------------------------------------------------------------===//
class ConvertMinReducePattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_min_reduce")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::MinReduceOp>(
        op, op.getResult().getType(), operands[0], operands[1], operands[2],
        operands[3]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ProdReduceOp Pattern
//===----------------------------------------------------------------------===//
class ConvertProdReducePattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_prod_reduce")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::ProdReduceOp>(
        op, op.getResult().getType(), operands[0], operands[1], operands[2],
        operands[3]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AndReduceOp Pattern
//===----------------------------------------------------------------------===//
class ConvertAndReducePattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_and_reduce")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::AndReduceOp>(
        op, op.getResult().getType(), operands[0], operands[1], operands[2],
        operands[3]);
    return success();
  }
};

//===----------------------------------------------------------------------===////
// OrReduceOp Pattern
//===----------------------------------------------------------------------===//
class ConvertOrReducePattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_or_reduce")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::OrReduceOp>(
        op, op.getResult().getType(), operands[0], operands[1], operands[2],
        operands[3]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// XorReduceOp Pattern
//===----------------------------------------------------------------------===//
class ConvertXorReducePattern : public OpRewritePattern<::cir::CallOp> {
public:
  using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(::cir::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee || callee.value() != "shmem_xor_reduce")
      return failure();

    auto operands = op.getOperands();
    if (operands.size() != 4)
      return failure();

    rewriter.replaceOpWithNewOp<openshmem::XorReduceOp>(
        op, op.getResult().getType(), operands[0], operands[1], operands[2],
        operands[3]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace openshmem {
namespace cir {

void populateCIRToOpenSHMEMCollectivePatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertBroadcastPattern, ConvertCollectPattern,
               ConvertFCollectPattern, ConvertAlltoallPattern,
               ConvertAlltoallsPattern, ConvertSumReducePattern,
               ConvertMaxReducePattern, ConvertMinReducePattern,
               ConvertProdReducePattern, ConvertAndReducePattern,
               ConvertOrReducePattern, ConvertXorReducePattern>(
      patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

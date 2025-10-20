//===- RewriteRMA.cpp - CIR to OpenSHMEM RMA Rewriters -------------------===//
//
// Implement CIR -> OpenSHMEM conversion patterns for RMA APIs (put/get,
// putmem/getmem) including context-aware and non-blocking variants. Patterns
// follow the same helper conventions used by other rewriters in this
// directory.
//
//===----------------------------------------------------------------------===//

#include "OpenSHMEMCIR/OpenSHMEMCIR.h"
#include "Utils.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace cir;

namespace mlir {
namespace openshmem {
namespace cir {

// Use centralized helper utilities implemented in Utils.cpp to avoid
// duplication and keep consistent bridge-op usage across rewriters.
using mlir::openshmem::cir::wrapSymmetricPtr;
using mlir::openshmem::cir::wrapValueToIndex;
using mlir::openshmem::cir::wrapValueToI32;
using mlir::openshmem::cir::wrapValueToCtx;
using mlir::openshmem::cir::wrapValueToType;

//===----------------------------------------------------------------------===//
// Basic RMA Patterns
//===----------------------------------------------------------------------===//

// shmem_put / typed variants -> openshmem.put / put_nbi
struct ConvertPutPattern : public OpRewritePattern<::cir::CallOp> {
	using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(::cir::CallOp op,
																PatternRewriter &rewriter) const override {
		if (!op.getCallee())
			return failure();

		StringRef callee = op.getCallee().value();
		// handle putmem separately
		if (callee.contains("putmem"))
			return failure();

		// Match put or typed-put (e.g., shmem_double_put) and non-blocking
		if (!callee.contains("_put"))
			return failure();

		auto operands = op.getArgOperands();
		if (operands.size() != 4)
			return failure();

		Value dest = wrapSymmetricPtr(rewriter, op.getLoc(), operands[0]);
		Value source = wrapSymmetricPtr(rewriter, op.getLoc(), operands[1]);
		Value nelems = wrapValueToIndex(rewriter, op.getLoc(), operands[2]);
		Value pe = wrapValueToI32(rewriter, op.getLoc(), operands[3]);

		if (callee.contains("_nbi")) {
			rewriter.replaceOpWithNewOp<openshmem::PutNbiOp>(op, dest, source,
																											nelems, pe);
		} else if (callee.starts_with("shmem_ctx_")) {
			// context-aware typed ctx_put handled elsewhere; avoid here
			return failure();
		} else {
			rewriter.replaceOpWithNewOp<openshmem::PutOp>(op, dest, source,
																									 nelems, pe);
		}

		return success();
	}
};

// shmem_get / typed variants -> openshmem.get / get_nbi
struct ConvertGetPattern : public OpRewritePattern<::cir::CallOp> {
	using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(::cir::CallOp op,
																PatternRewriter &rewriter) const override {
		if (!op.getCallee())
			return failure();

		StringRef callee = op.getCallee().value();
		// handle getmem separately
		if (callee.contains("getmem"))
			return failure();

		if (!callee.contains("_get"))
			return failure();

		auto operands = op.getArgOperands();
		if (operands.size() != 4)
			return failure();

		Value dest = wrapSymmetricPtr(rewriter, op.getLoc(), operands[0]);
		Value source = wrapSymmetricPtr(rewriter, op.getLoc(), operands[1]);
		Value nelems = wrapValueToIndex(rewriter, op.getLoc(), operands[2]);
		Value pe = wrapValueToI32(rewriter, op.getLoc(), operands[3]);

		if (callee.contains("_nbi")) {
			rewriter.replaceOpWithNewOp<openshmem::GetNbiOp>(op, dest, source,
																											nelems, pe);
		} else if (callee.starts_with("shmem_ctx_")) {
			// context-aware handled in ctx pattern
			return failure();
		} else {
			rewriter.replaceOpWithNewOp<openshmem::GetOp>(op, dest, source,
																									 nelems, pe);
		}

		return success();
	}
};

// shmem_putmem / shmem_putmem_nbi -> openshmem.putmem / putmem_nbi
struct ConvertPutmemPattern : public OpRewritePattern<::cir::CallOp> {
	using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(::cir::CallOp op,
																PatternRewriter &rewriter) const override {
		if (!op.getCallee() || !op.getCallee().value().contains("putmem"))
			return failure();

		auto operands = op.getArgOperands();
		if (operands.size() != 4)
			return failure();

		Value dest = wrapSymmetricPtr(rewriter, op.getLoc(), operands[0]);
		// src is treated as AnyType in the op; cast pointer to memref for now
		Value src = wrapSymmetricPtr(rewriter, op.getLoc(), operands[1]);
		Value size = wrapValueToIndex(rewriter, op.getLoc(), operands[2]);
		Value pe = wrapValueToI32(rewriter, op.getLoc(), operands[3]);

		if (op.getCallee().value().contains("_nbi")) {
			rewriter.replaceOpWithNewOp<openshmem::PutmemNbiOp>(op, dest, src, size,
																												 pe);
		} else {
			rewriter.replaceOpWithNewOp<openshmem::PutmemOp>(op, dest, src, size,
																											pe);
		}

		return success();
	}
};

// shmem_getmem / shmem_getmem_nbi -> openshmem.getmem / getmem_nbi
struct ConvertGetmemPattern : public OpRewritePattern<::cir::CallOp> {
	using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(::cir::CallOp op,
																PatternRewriter &rewriter) const override {
		if (!op.getCallee() || !op.getCallee().value().contains("getmem"))
			return failure();

		auto operands = op.getArgOperands();
		if (operands.size() != 4)
			return failure();

		Value dest = wrapSymmetricPtr(rewriter, op.getLoc(), operands[0]);
		Value src = wrapSymmetricPtr(rewriter, op.getLoc(), operands[1]);
		Value size = wrapValueToIndex(rewriter, op.getLoc(), operands[2]);
		Value pe = wrapValueToI32(rewriter, op.getLoc(), operands[3]);

		if (op.getCallee().value().contains("_nbi")) {
			rewriter.replaceOpWithNewOp<openshmem::GetmemNbiOp>(op, dest, src, size,
																												 pe);
		} else {
			rewriter.replaceOpWithNewOp<openshmem::GetmemOp>(op, dest, src, size,
																											pe);
		}

		return success();
	}
};

// Context-aware variants: shmem_ctx_put / shmem_ctx_get (and nbi)
struct ConvertCtxPutPattern : public OpRewritePattern<::cir::CallOp> {
	using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(::cir::CallOp op,
																PatternRewriter &rewriter) const override {
		if (!op.getCallee())
			return failure();
		StringRef callee = op.getCallee().value();
		if (!callee.contains("ctx_put"))
			return failure();

		auto operands = op.getArgOperands();
		if (operands.size() != 5)
			return failure();

		Value ctx = wrapValueToCtx(rewriter, op.getLoc(), operands[0]);
		Value dest = wrapSymmetricPtr(rewriter, op.getLoc(), operands[1]);
		Value source = wrapSymmetricPtr(rewriter, op.getLoc(), operands[2]);
		Value nelems = wrapValueToIndex(rewriter, op.getLoc(), operands[3]);
		Value pe = wrapValueToI32(rewriter, op.getLoc(), operands[4]);

		if (callee.contains("_nbi")) {
			rewriter.replaceOpWithNewOp<openshmem::CtxPutNbiOp>(op, ctx, dest,
																												source, nelems, pe);
		} else {
			rewriter.replaceOpWithNewOp<openshmem::CtxPutOp>(op, ctx, dest, source,
																										 nelems, pe);
		}
		return success();
	}
};

struct ConvertCtxGetPattern : public OpRewritePattern<::cir::CallOp> {
	using OpRewritePattern<::cir::CallOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(::cir::CallOp op,
																PatternRewriter &rewriter) const override {
		if (!op.getCallee())
			return failure();
		StringRef callee = op.getCallee().value();
		if (!callee.contains("ctx_get"))
			return failure();

		auto operands = op.getArgOperands();
		if (operands.size() != 5)
			return failure();

		Value ctx = wrapValueToCtx(rewriter, op.getLoc(), operands[0]);
		Value dest = wrapSymmetricPtr(rewriter, op.getLoc(), operands[1]);
		Value source = wrapSymmetricPtr(rewriter, op.getLoc(), operands[2]);
		Value nelems = wrapValueToIndex(rewriter, op.getLoc(), operands[3]);
		Value pe = wrapValueToI32(rewriter, op.getLoc(), operands[4]);

		if (callee.contains("_nbi")) {
			rewriter.replaceOpWithNewOp<openshmem::CtxGetNbiOp>(op, ctx, dest,
																												source, nelems, pe);
		} else {
			rewriter.replaceOpWithNewOp<openshmem::CtxGetOp>(op, ctx, dest, source,
																										 nelems, pe);
		}
		return success();
	}
};

// Registration
void populateCIRToOpenSHMEMRMAPatterns(RewritePatternSet &patterns) {
	patterns.add<ConvertPutPattern, ConvertGetPattern, ConvertPutmemPattern,
							 ConvertGetmemPattern, ConvertCtxPutPattern, ConvertCtxGetPattern>(
			patterns.getContext());
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

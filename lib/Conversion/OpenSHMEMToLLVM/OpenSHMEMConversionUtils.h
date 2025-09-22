//===- OpenSHMEMConversionUtils.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENSHMEM_LIB_CONVERSION_UTILS_WRAPPER_H
#define OPENSHMEM_LIB_CONVERSION_UTILS_WRAPPER_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::openshmem {

LLVM::LLVMFuncOp getOrDefineFunction(ModuleOp &moduleOp, const Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     StringRef name,
                                     LLVM::LLVMFunctionType type);

Value getMemRefDataPtr(Location loc, ConversionPatternRewriter &rewriter,
                       Value memref);

Type getSymmetricMemRefElementType(Value symmetricMemRef);

std::string getTypedFunctionName(StringRef baseName, Type elementType);
std::string getSizedFunctionName(StringRef baseName, Type elementType);
std::string getRMASizedFunctionName(StringRef baseName, Type elementType);
std::string getPt2ptSyncSizedFunctionName(StringRef baseName,
                                          Type cmpValueType);
std::string getPt2ptSyncVectorFunctionName(StringRef baseName);

} // namespace mlir::openshmem

#endif // OPENSHMEM_LIB_CONVERSION_UTILS_WRAPPER_H

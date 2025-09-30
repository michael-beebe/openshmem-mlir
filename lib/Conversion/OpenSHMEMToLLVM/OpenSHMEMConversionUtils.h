//===- OpenSHMEMConversionUtils.h - Shared conversion utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utility functions shared across OpenSHMEM to LLVM
// conversion patterns. These utilities help eliminate code duplication
// across the modular conversion files.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_CONVERSION_OPENSHMEMTOLLVM_OPENSHMEMCONVERSIONUTILS_H
#define MLIR_LIB_CONVERSION_OPENSHMEMTOLLVM_OPENSHMEMCONVERSIONUTILS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace openshmem {

//===----------------------------------------------------------------------===//
// Function declaration utilities
//===----------------------------------------------------------------------===//

/// Utility to get or define a function in the module. If the function with
/// the given name already exists, returns it. Otherwise, creates a new
/// function declaration with external linkage.
LLVM::LLVMFuncOp getOrDefineFunction(ModuleOp &moduleOp, const Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     StringRef name,
                                     LLVM::LLVMFunctionType type);

//===----------------------------------------------------------------------===//
// Memory reference utilities
//===----------------------------------------------------------------------===//

/// Utility to extract the data pointer from a memref descriptor.
/// Assumes memref is a MemRef descriptor (struct) and extracts the pointer
/// from field 0.
Value getMemRefDataPtr(Location loc, ConversionPatternRewriter &rewriter,
                       Value memref);

/// Utility to extract element type from memref with symmetric memory space.
/// Returns nullptr if the value is not a memref with symmetric memory space.
Type getSymmetricMemRefElementType(Value symmetricMemRef);

//===----------------------------------------------------------------------===//
// OpenSHMEM function naming utilities
//===----------------------------------------------------------------------===//

/// Utility to generate typed function names based on element type.
/// Maps MLIR types to OpenSHMEM type names (e.g., "put" + f32 ->
/// "shmem_float_put"). For unsupported types, falls back to generic name.
std::string getTypedFunctionName(StringRef baseName, Type elementType);

/// Utility to generate sized function names based on element type.
/// Maps MLIR types to OpenSHMEM sized type names (e.g., "wait_until" + i32 ->
/// "shmem_int32_wait_until"). Used for pt2pt sync operations that require sized
/// names (especially vectors).
std::string getSizedFunctionName(StringRef baseName, Type elementType);

/// Utility to generate RMA sized function names based on element type.
/// Maps MLIR types to OpenSHMEM RMA sized function names (e.g., "put" + i32 ->
/// "shmem_put32"). Used for RMA operations that use sized names.
std::string getRMASizedFunctionName(StringRef baseName, Type elementType);

/// Utility to generate pt2pt sync sized function names based on comparison
/// value type. Maps MLIR types to OpenSHMEM pt2pt sync sized function names
/// (e.g., "wait_until" + i32 -> "shmem_wait_until32"). Used for pt2pt sync
/// operations that use sized names.
std::string getPt2ptSyncSizedFunctionName(StringRef baseName,
                                          Type cmpValueType);

/// Utility to generate pt2pt sync vector function names.
/// Maps operation names to OpenSHMEM pt2pt sync vector function names (e.g.,
/// "wait_until_all_vector" -> "shmem_wait_until_all_vector"). Used for pt2pt
/// sync vector operations that use simple naming.
std::string getPt2ptSyncVectorFunctionName(StringRef baseName);

} // namespace openshmem
} // namespace mlir

#endif // MLIR_LIB_CONVERSION_OPENSHMEMTOLLVM_OPENSHMEMCONVERSIONUTILS_H
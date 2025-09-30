//===- OpenSHMEMConversionUtils.cpp - Shared conversion utilities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpenSHMEMConversionUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace mlir {
namespace openshmem {

//===----------------------------------------------------------------------===//
// Function declaration utilities
//===----------------------------------------------------------------------===//

LLVM::LLVMFuncOp getOrDefineFunction(ModuleOp &moduleOp,
                                    const Location loc,
                                    ConversionPatternRewriter &rewriter,
                                    StringRef name,
                                    LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp funcOp;
  if (!(funcOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    funcOp = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type,
                                               LLVM::Linkage::External);
  }
  return funcOp;
}

//===----------------------------------------------------------------------===//
// Memory reference utilities  
//===----------------------------------------------------------------------===//

Value getMemRefDataPtr(Location loc, ConversionPatternRewriter &rewriter,
                       Value memref) {
  // Check if the memref is already a pointer (our OpenSHMEM conversion)
  if (llvm::isa<LLVM::LLVMPointerType>(memref.getType())) {
    return memref;
  }
  
  // Otherwise, assume it's a MemRef descriptor (struct), extract the pointer (field 0)
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  return rewriter.create<LLVM::ExtractValueOp>(loc, ptrType, memref, 0);
}

Type getSymmetricMemRefElementType(Value symmetricMemRef) {
  auto memRefType = llvm::dyn_cast<MemRefType>(symmetricMemRef.getType());
  if (!memRefType || !memRefType.getMemorySpace() ||
      !llvm::isa<openshmem::SymmetricMemorySpaceAttr>(memRefType.getMemorySpace())) {
    return nullptr;
  }
  return memRefType.getElementType();
}

//===----------------------------------------------------------------------===//
// OpenSHMEM function naming utilities
//===----------------------------------------------------------------------===//

std::string getTypedFunctionName(StringRef baseName, Type elementType) {
  std::string funcName = "shmem_";
  
  // Map MLIR types to OpenSHMEM type names (not sizes!)
  // These must match the actual function names in the OpenSHMEM library
  if (elementType.isInteger(8)) {
    funcName += "char_";
  } else if (elementType.isInteger(16)) {
    funcName += "short_";
  } else if (elementType.isInteger(32)) {
    funcName += "int_";
  } else if (elementType.isInteger(64)) {
    funcName += "long_";
  } else if (elementType.isF32()) {
    funcName += "float_";
  } else if (elementType.isF64()) {
    funcName += "double_";
  } else if (elementType.isF128()) {
    // F128 uses sized functions, not typed functions
    // Return early with sized pattern: shmem_put128, shmem_get128
    funcName += baseName.str() + "128";
    return funcName;
  } else {
    // For unsupported types, fall back to generic name
    // This should be validated earlier in the process
    funcName = "shmem_" + baseName.str();
    return funcName;
  }
  
  funcName += baseName.str();
  return funcName;
}

std::string getSizedFunctionName(StringRef baseName, Type elementType) {
  std::string funcName = "shmem_";
  
  // Map MLIR types to OpenSHMEM sized type names
  // Used for pt2pt sync operations that require sized names (especially vectors)
  // Note: OpenSHMEM only has int32 and int64 sized functions for sync operations
  if (!elementType) {
    // If element type is null, fall back to int32
    funcName += "int32_";
  } else if (elementType.isInteger(64)) {
    funcName += "int64_";
  } else if (elementType.isF64()) {
    // Double operations in pt2pt sync use int64 for storage
    funcName += "int64_";
  } else {
    // All other types (i8, i16, i32, f32, etc.) use int32
    // This includes most common cases
    funcName += "int32_";
  }
  
  funcName += baseName.str();
  return funcName;
}

std::string getRMASizedFunctionName(StringRef baseName, Type elementType) {
  std::string funcName = "shmem_";
  
  // Map MLIR types to OpenSHMEM RMA sized function names
  // These use the pattern: shmem_put32, shmem_put64, shmem_get32, etc.
  if (elementType.isInteger(8)) {
    funcName += baseName.str() + "8";
  } else if (elementType.isInteger(16)) {
    funcName += baseName.str() + "16";
  } else if (elementType.isInteger(32)) {
    funcName += baseName.str() + "32";
  } else if (elementType.isInteger(64)) {
    funcName += baseName.str() + "64";
  } else if (elementType.isF16()) {
    funcName += baseName.str() + "16";
  } else if (elementType.isF32()) {
    funcName += baseName.str() + "32";
  } else if (elementType.isF64()) {
    funcName += baseName.str() + "64";
  } else if (elementType.isF128()) {
    funcName += baseName.str() + "128";
  } else {
    // For unsupported types, fall back to generic name
    funcName = "shmem_" + baseName.str();
    return funcName;
  }
  
  return funcName;
}

std::string getPt2ptSyncSizedFunctionName(StringRef baseName, Type cmpValueType) {
  std::string funcName = "shmem_";
  
  // Map MLIR types to OpenSHMEM pt2pt sync sized function names
  // These use the pattern: shmem_wait_until32, shmem_wait_until64, etc.
  if (!cmpValueType) {
    // If cmp value type is null, fall back to 32
    funcName += baseName.str() + "32";
  } else if (cmpValueType.isInteger(8)) {
    funcName += baseName.str() + "8";
  } else if (cmpValueType.isInteger(16)) {
    funcName += baseName.str() + "16";
  } else if (cmpValueType.isInteger(32)) {
    funcName += baseName.str() + "32";
  } else if (cmpValueType.isInteger(64)) {
    funcName += baseName.str() + "64";
  } else if (cmpValueType.isF16()) {
    funcName += baseName.str() + "16";
  } else if (cmpValueType.isF32()) {
    funcName += baseName.str() + "32";
  } else if (cmpValueType.isF64()) {
    funcName += baseName.str() + "64";
  } else if (cmpValueType.isF128()) {
    funcName += baseName.str() + "128";
  } else {
    // For unsupported types, fall back to 32
    funcName += baseName.str() + "32";
  }
  
  return funcName;
}

std::string getPt2ptSyncVectorFunctionName(StringRef baseName) {
  std::string funcName = "shmem_";
  
  // Vector operations use simple naming without size suffixes
  // These use the pattern: shmem_wait_until_all_vector, shmem_wait_until_any_vector, etc.
  funcName += baseName.str();
  
  return funcName;
}

} // namespace openshmem
} // namespace mlir 

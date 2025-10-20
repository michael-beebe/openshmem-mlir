//===- Utils.cpp - OpenSHMEM CIR Utility Functions -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for OpenSHMEM CIR transformations.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::openshmem;

namespace mlir {
namespace openshmem {
namespace cir {

bool isSymmetricMemoryType(Type type) {
  // TODO: Implement logic to detect symmetric memory types
  // This would typically check for specific CIR attributes or type properties
  // that indicate symmetric memory allocation
  return false;
}

Type convertCIRPtrToOpenSHMEMMemRef(Type cirPtrType, MLIRContext *context) {
  // TODO: Implement CIR pointer to OpenSHMEM memref conversion
  // This requires understanding the CIR type system

  // For now, return a generic dynamic memref of i8
  auto elementType = IntegerType::get(context, 8);
  return MemRefType::get({ShapedType::kDynamic}, elementType);
}

Type getElementTypeFromCIRPtr(Type cirPtrType) {
  // TODO: Extract element type from CIR pointer type
  // This requires CIR dialect integration

  // For now, return i8 as a placeholder
  return IntegerType::get(cirPtrType.getContext(), 8);
}

MemRefType createSymmetricMemRefType(MLIRContext *ctx, Type elementType) {
  auto symmetricMemSpace = openshmem::SymmetricMemorySpaceAttr::get(ctx);
  Type elem = elementType;
  if (!elem)
    elem = IntegerType::get(ctx, 8);
  return MemRefType::get({ShapedType::kDynamic}, elem,
                         MemRefLayoutAttrInterface{}, symmetricMemSpace);
}

Value wrapSymmetricPtr(OpBuilder &builder, Location loc, Value ptr) {
  auto memRefType = createSymmetricMemRefType(builder.getContext(), nullptr);
  return builder.create<openshmem::WrapSymmetricPtrOp>(loc, memRefType, ptr)
      .getResult();
}

Value wrapValueToIndex(OpBuilder &builder, Location loc, Value value) {
  return builder.create<openshmem::WrapValueOp>(loc, builder.getIndexType(),
                                                value)
      .getResult();
}

Value wrapValueToI32(OpBuilder &builder, Location loc, Value value) {
  return builder.create<openshmem::WrapValueOp>(loc, builder.getI32Type(),
                                                value)
      .getResult();
}

Value wrapValueToCtx(OpBuilder &builder, Location loc, Value value) {
  auto ctxType = openshmem::CtxType::get(builder.getContext());
  return builder.create<openshmem::WrapCtxOp>(loc, ctxType, value).getResult();
}

Value wrapValueToType(OpBuilder &builder, Location loc, Value value, Type targetType) {
  return builder.create<openshmem::WrapValueOp>(loc, targetType, value)
      .getResult();
}

} // namespace cir
} // namespace openshmem
} // namespace mlir

//===- OpenSHMEMBridge.td - Bridge helper operations --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMBRIDGE_TD
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMBRIDGE_TD

include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMBase.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMTypes.td"

//===----------------------------------------------------------------------===//
// Bridge Helpers
//===----------------------------------------------------------------------===//
//
// These operations provide lightweight adapters that allow the CIR rewriters
// to express pointer and integer values using the OpenSHMEM dialect's operand
// types without relying on builtin.unrealized_conversion_cast. They are
// eliminated during the OpenSHMEM → LLVM lowering and therefore have no runtime
// meaning.
//===----------------------------------------------------------------------===//

def OpenSHMEM_WrapSymmetricPtrOp : OpenSHMEM_Op<"wrap_symmetric_ptr"> {
  let summary = "Wrap a raw pointer as a symmetric memref value";
  let description = [{
    Bridges a pointer-typed SSA value produced in the CIR dialect into a
    `SymmetricMemRef` so that downstream OpenSHMEM ops can consume it. The
    result memref type is provided by the builder and typically models a
    `memref<?xi8, symmetric>` placeholder. The lowering rewrites the operation
    to an LLVM pointer without generating additional IR.
  }];

  let arguments = (ins AnyType:$ptr);
  let results = (outs SymmetricMemRef:$result);
  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($result)";
}

def OpenSHMEM_WrapLocalPtrOp : OpenSHMEM_Op<"wrap_local_ptr"> {
  let summary = "Wrap a raw pointer as a regular memref value";
  let description = [{
    Wraps a pointer value so that it can be treated as an "any" memref by the
    OpenSHMEM dialect. Like the symmetric variant, this is erased during
    lowering.
  }];

  let arguments = (ins AnyType:$ptr);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($result)";
}

def OpenSHMEM_WrapCtxOp : OpenSHMEM_Op<"wrap_ctx"> {
  let summary = "Wrap a raw pointer as shmem_ctx_t";
  let description = [{
    Converts an arbitrary pointer value into the dialect's `CtxType`. This is
    required because the CIR frontend represents contexts as plain pointers.
    The lowering simply forwards the pointer value after type conversion.
  }];

  let arguments = (ins AnyType:$ptr);
  let results = (outs OpenSHMEM_Ctx:$ctx);
  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($ctx)";
}

def OpenSHMEM_WrapValueOp : OpenSHMEM_Op<"wrap_value"> {
  let summary = "Wrap a value to reinterpret it with a different type";
  let description = [{
    Generic helper used to bridge scalar values (e.g. integers) from CIR types
    into the builtin integer and index types expected by the OpenSHMEM dialect.
    The operation carries no semantics beyond acting as an annotation for the
    lowering pipeline, which replaces it with an appropriate LLVM conversion or
    bitcast.

    The producer must provide the desired result type explicitly. No implicit
    shape or size changes are performed at this stage.
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMBRIDGE_TD

//===- OpenSHMEMOps.cpp - OpenSHMEM dialect ops implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;
using namespace mlir::openshmem;

//===----------------------------------------------------------------------===//
// RegionOp
//===----------------------------------------------------------------------===//

LogicalResult openshmem::Region::verify() {
  // With SingleBlockImplicitTerminator, MLIR automatically ensures
  // the region has the proper terminator, so we just need to verify
  // the region is not empty
  if (getBody().empty()) {
    return emitOpError("region must contain at least one block");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Atomic Operation Folding
//===----------------------------------------------------------------------===//

// Note: Atomic operations (atomic_add, atomic_inc, atomic_or, atomic_xor)
// have ZeroResults trait, so they don't produce results to fold.
// Folding is typically used for operations that produce values that can be
// replaced with constants. For atomic operations, the optimization happens
// through patterns in the AtomicFusionPass instead.

// Register the dialect operations
#define GET_OP_CLASSES
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMOps.cpp.inc"

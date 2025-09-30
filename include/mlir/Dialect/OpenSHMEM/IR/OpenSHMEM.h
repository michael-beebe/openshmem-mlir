//===- OpenSHMEM.h - OpenSHMEM dialect -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEM_H_
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEM_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// OpenSHMEMDialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMTypesGen.h.inc"

#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMOps.h.inc"

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEM_H_

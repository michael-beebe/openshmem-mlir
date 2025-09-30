//===- OpenSHMEM.cpp - OpenSHMEM dialect implementation
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::openshmem;

//===----------------------------------------------------------------------===//
/// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.cpp.inc"
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMDialect.cpp.inc"

void OpenSHMEMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMTypesGen.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMAttrDefs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
/// Type and Attribute Parsing/Printing
//===----------------------------------------------------------------------===//

// parseType, printType, parseAttribute, and printAttribute are auto-generated
// by TableGen No custom implementation needed since we use the default
// parsing/printing

//===----------------------------------------------------------------------===//
/// TableGen'd dialect, and op definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMTypesGen.cpp.inc"

#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMAttrDefs.cpp.inc"

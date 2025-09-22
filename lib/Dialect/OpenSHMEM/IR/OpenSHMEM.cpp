#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::openshmem;

#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMDialect.cpp.inc"

void OpenSHMEMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMAttrs.cpp.inc"
      >();
}

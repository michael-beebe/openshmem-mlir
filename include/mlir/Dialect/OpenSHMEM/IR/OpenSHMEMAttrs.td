//===- OpenSHMEMAttrs.td - OpenSHMEM dialect attributes --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMATTRS_TD
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMATTRS_TD

include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Base OpenSHMEM attribute
//===----------------------------------------------------------------------===//

class OpenSHMEM_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
    : AttrDef<OpenSHMEM_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Memory Space Attributes
//===----------------------------------------------------------------------===//

def OpenSHMEM_SymmetricMemorySpaceAttr : OpenSHMEM_Attr<"SymmetricMemorySpace", "symmetric_memory"> {
  let summary = "Memory space attribute for symmetric memory";
  let description = [{
    This attribute represents the symmetric memory space in OpenSHMEM.
    Symmetric memory is accessible from all PEs and has the same virtual
    address across all PEs. This memory space is used with memref types
    to distinguish symmetric memory allocations from local memory.
    
    Example:
    ```mlir
    #openshmem.symmetric_memory
    ```
    
    Used in memref types:
    ```mlir
    memref<100xf32, #openshmem.symmetric_memory>
    ```
  }];
  
  let assemblyFormat = "";
}

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMATTRS_TD

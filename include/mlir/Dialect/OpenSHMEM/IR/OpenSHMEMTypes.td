//===- OpenSHMEMTypes.td - OpenSHMEM dialect types --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMTYPES_TD
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMTYPES_TD

include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Base OpenSHMEM type
//===----------------------------------------------------------------------===//

class OpenSHMEM_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<OpenSHMEM_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Processing Element (PE) type
//
// TODO: potentially extend this to represent global PE number?
// TODO: potentially add a second type for PE number within a team?
//===----------------------------------------------------------------------===//

def OpenSHMEM_PE : OpenSHMEM_Type<"PE", "pe"> {
  let summary = "OpenSHMEM Processing Element number";
  let description = [{
    Represents a Processing Element (PE) number in an OpenSHMEM program.
    PE numbers range from 0 to n_pes()-1.
  }];
}

//===----------------------------------------------------------------------===//
// Type constraints for symmetric memory
//===----------------------------------------------------------------------===//

// Constraint for memref types with symmetric memory space
def SymmetricMemRef : TypeConstraint<
  CPred<"llvm::isa<MemRefType>($_self) && "
        "llvm::cast<MemRefType>($_self).getMemorySpace() && "
        "llvm::isa<::mlir::openshmem::SymmetricMemorySpaceAttr>(llvm::cast<MemRefType>($_self).getMemorySpace())">,
  "memref with symmetric memory space">;

//===----------------------------------------------------------------------===//
// Team type
//===----------------------------------------------------------------------===//

def OpenSHMEM_Team : OpenSHMEM_Type<"Team", "team"> {
  let summary = "Team type for OpenSHMEM operations";
  let description = [{
    Represents a team of PEs in an OpenSHMEM program. Teams are subsets of PEs
    that can perform collective operations independently. A team is created by
    splitting an existing team (or the world team) and can be used for scoped
    collective operations like barriers, broadcasts, and reductions.
    
    Teams provide a way to organize PEs into logical groups for communication
    and synchronization. Each PE can be a member of multiple teams simultaneously.
  }];
}

//===----------------------------------------------------------------------===//
// Context type
//===----------------------------------------------------------------------===//

def OpenSHMEM_Ctx : OpenSHMEM_Type<"Ctx", "ctx"> {
  let summary = "Context type for ordered OpenSHMEM operations";
  let description = [{
    Represents an OpenSHMEM context. Contexts are containers for communication
    operations. Each context provides an environment where the operations
    performed on that context are ordered and completed independently of other
    operations performed by the application.
  }];
}

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMTYPES_TD

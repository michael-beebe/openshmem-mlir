//===- OpenSHMEMSetup.td - OpenSHMEM setup operations ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMSETUP_TD
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMSETUP_TD

include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMBase.td"

//===----------------------------------------------------------------------===//
// InitOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_InitOp : OpenSHMEM_Op<"init", []> {
  let summary = "Equivalent to the shmem_init C API";
  let description = [{
    Initializes the OpenSHMEM runtime environment. This must be the first
    OpenSHMEM routine called by a program.

    Arguments:
    - (none)

    ```c
    void shmem_init(void);
    ```
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// FinalizeOp  
//===----------------------------------------------------------------------===//

def OpenSHMEM_FinalizeOp : OpenSHMEM_Op<"finalize", []> {
  let summary = "Equivalent to the shmem_finalize C API";
  let description = [{
    Finalizes the OpenSHMEM runtime environment and cleans up resources.
    This should be the last OpenSHMEM routine called by a program.

    Arguments:
    - (none)

    ```c
    void shmem_finalize(void);
    ```
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// RegionOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_Region : OpenSHMEM_Op<"region", [
  SingleBlockImplicitTerminator<"openshmem::YieldOp">
]> {
  let summary = "OpenSHMEM program region";
  let description = [{
    Defines a region containing OpenSHMEM operations. This operation implicitly
    handles initialization at the beginning and finalization at the end of the
    region. All OpenSHMEM operations must be contained within this region.

    The region ensures that:
    - OpenSHMEM runtime is properly initialized before any operations
    - OpenSHMEM runtime is properly finalized after all operations
    - OpenSHMEM operations cannot be moved outside this region
    - The lifecycle of the OpenSHMEM program is clearly defined

    ```mlir
    openshmem.region {
      // All OpenSHMEM operations go here
      %pe = openshmem.my_pe : i32
      %src = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      // ... more OpenSHMEM operations
      // openshmem.yield is automatically inserted if not present
    }
    ```
  }];
  
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_YieldOp : OpenSHMEM_Op<"yield", [Terminator]> {
  let summary = "Terminator for OpenSHMEM region";
  let description = [{
    Terminator operation for OpenSHMEM region. This operation must be present
    at the end of every OpenSHMEM region.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// MyPeOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_MyPeOp : OpenSHMEM_Op<"my_pe", []> {
  let summary = "Equivalent to the shmem_my_pe C API";
  let description = [{
    Returns the processing element (PE) number of the calling PE. PE numbers
    are integers from 0 to n_pes() - 1.

    Arguments:
    - (none)

    Results:
    - pe: The PE number of the calling PE (typically i32 or !cir.int<s,32>)

    ```c
    int shmem_my_pe(void);
    ```
  }];

  let results = (outs AnyType:$pe);
  let assemblyFormat = "attr-dict `:` type($pe)";
}

//===----------------------------------------------------------------------===//
// NPesOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_NPesOp : OpenSHMEM_Op<"n_pes", []> {
  let summary = "Equivalent to the shmem_n_pes C API";
  let description = [{
    Returns the total number of processing elements (PEs) in the OpenSHMEM
    program.

    Arguments:
    - (none)

    Results:
    - n_pes: The total number of PEs (typically i32 or !cir.int<s,32>)

    ```c
    int shmem_n_pes(void);
    ```
  }];

  let results = (outs AnyType:$n_pes);
  let assemblyFormat = "attr-dict `:` type($n_pes)";
}

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMSETUP_TD 

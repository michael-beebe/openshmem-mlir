//===- OpenSHMEMAtomics.td - OpenSHMEM atomic operations ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHEMATOMICS_TD
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHEMATOMICS_TD

include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMBase.td"

//===----------------------------------------------------------------------===//
// Atomic Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AtomicFetchOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicFetchOp : OpenSHMEM_Op<"atomic_fetch", []> {
  let summary = "Atomic fetch operation (generic)";
  let description = [{
    This operation atomically fetches the value of a remote data object from
    the specified PE. The operation returns the value that was fetched.
    
    This is the generic typed equivalent of atomic fetch operations, providing
    type safety and better performance for known data types.

    Signature:
    ```c
    TYPE shmem_atomic_fetch(const TYPE *source, int pe);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$source,
    I32:$pe
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "`(` $source `,` $pe `)` attr-dict `:` "
                       "type($source) `,` type($pe) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicFetchOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicFetchOp : OpenSHMEM_Op<"ctx_atomic_fetch", []> {
  let summary = "Atomic fetch operation with context";
  let description = [{
    This operation atomically fetches the value of a remote data object from
    the specified PE using a specific context. The operation returns the value
    that was fetched.
  
    Signature:
    ```c
    TYPE shmem_atomic_fetch(shmem_ctx_t ctx, const TYPE *source, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$source,
    I32:$pe
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "`(` $ctx `,` $source `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($source) `,` type($pe) `->`"
                       " type($result)";
}

//===----------------------------------------------------------------------===//
// AtomicSetOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicSetOp : OpenSHMEM_Op<"atomic_set", []> {
  let summary = "Atomic set operation (generic)";
  let description = [{
    Atomically sets the value of a remote data object.

    Signature:
    ```c
    void shmem_atomic_set(TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );
  
  let assemblyFormat = "`(` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($dest) `,` type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicSetOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicSetOp : OpenSHMEM_Op<"ctx_atomic_set", []> {
  let summary = "Atomic set operation with context";
  let description = [{
    Atomically sets the value of a remote data object.
    
    Signature:
    ```c
    void shmem_ctx_atomic_set(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );
  
  let assemblyFormat = "`(` $ctx `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// AtomicCompareSwapOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicCompareSwapOp : OpenSHMEM_Op<"atomic_compare_swap", []> {
  let summary = "Atomic compare-and-swap operation (generic)";
  let description = [{
    Atomically compares the value of a remote data object with a given value and
    swaps it with a new value if they match.

    Signature:
    ```c
    TYPE shmem_atomic_compare_swap(TYPE *dest, TYPE cond, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$dest,
    AnyType:$cond,
    AnyType:$value,
    I32:$pe
  );
  
  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $dest `,` $cond `,` $value `,` $pe `)` attr-dict `:` "
                       "type($dest) `,` type($cond) `,` type($value) `,`"
                       "type($pe) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicCompareSwapOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicCompareSwapOp :
  OpenSHMEM_Op<"ctx_atomic_compare_swap", []> {
  let summary = "Atomic compare-and-swap operation with context";
  let description = [{
    Atomically compares the value of a remote data object with a given value and
    swaps it with a new value if they match.

    Signature:
    ```c
    TYPE shmem_ctx_atomic_compare_swap(shmem_ctx_t ctx, TYPE *dest, TYPE cond,
                                       TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyType:$cond,
    AnyType:$value,
    I32:$pe
  );
  
  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $ctx `,` $dest `,` $cond `,` $value `,` $pe `)` "
                       "attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($cond) `,`"
                       " type($value) `,` type($pe) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// AtomicSwapOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicSwapOp : OpenSHMEM_Op<"atomic_swap", []> {
  let summary = "Atomic swap operation (generic)";
  let description = [{
    Atomically swaps the value of a remote data object with a given value.

    Signature:
    ```c
    TYPE shmem_atomic_swap(TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );
  
  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($dest) `,` type($value) `,` type($pe) `->`"
                       " type($result)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicSwapOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicSwapOp : OpenSHMEM_Op<"ctx_atomic_swap", []> {
  let summary = "Atomic swap operation with context";
  let description = [{
    Atomically swaps the value of a remote data object with a given value.

    Signature:
    ```c
    TYPE shmem_ctx_atomic_swap(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $ctx `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($value) `,`"
                       " type($pe) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// AtomicFetchIncOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicFetchIncOp : OpenSHMEM_Op<"atomic_fetch_inc", []> {
  let summary = "Atomic fetch-and-increment operation (generic)";
  let description = [{
    Atomically fetches the value of a remote data object and increments it by 1.

    Signature:
    ```c
    TYPE shmem_atomic_fetch_inc(TYPE *dest, int pe);
    ```
    where TYPE is one of the standard AMO types from Table 6.
  }];

  let arguments = (ins 
    SymmetricMemRef:$dest,
    I32:$pe
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $dest `,` $pe `)` attr-dict `:` "
                       "type($dest) `,` type($pe) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicFetchIncOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicFetchIncOp : OpenSHMEM_Op<"ctx_atomic_fetch_inc", []> {
  let summary = "Atomic fetch-and-increment operation with context";
  let description = [{
    Atomically fetches the value of a remote data object and increments it by 1.

    Signature:
    ```c
    TYPE shmem_ctx_atomic_fetch_inc(shmem_ctx_t ctx, TYPE *dest, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    I32:$pe
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $ctx `,` $dest `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($pe) `->`"
                       " type($result)";
}

//===----------------------------------------------------------------------===//
// AtomicIncOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicIncOp : OpenSHMEM_Op<"atomic_inc", []> {
  let summary = "Atomic increment operation (generic)";
  let description = [{
    Atomically increments the value of a remote data object by 1.

    Signature:
    ```c
    void shmem_atomic_inc(TYPE *dest, int pe);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$dest,
    I32:$pe
  );

  let assemblyFormat = "`(` $dest `,` $pe `)` attr-dict `:` "
                       "type($dest) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicIncOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicIncOp : OpenSHMEM_Op<"ctx_atomic_inc", []> {
  let summary = "Atomic increment operation with context";
  let description = [{
    Atomically increments the value of a remote data object by 1.

    Signature:
    ```c
    void shmem_ctx_atomic_inc(shmem_ctx_t ctx, TYPE *dest, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $dest `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// AtomicFetchAddOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicFetchAddOp : OpenSHMEM_Op<"atomic_fetch_add", []> {
  let summary = "Atomic fetch-and-add operation (generic)";
  let description = [{
    Atomically fetches the value of a remote data object and adds a given
    value to it.

    Signature:
    ```c
    TYPE shmem_atomic_fetch_add(TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($dest) `,` type($value) `,` type($pe) `->`"
                       " type($result)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicFetchAddOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicFetchAddOp : OpenSHMEM_Op<"ctx_atomic_fetch_add", []> {
  let summary = "Atomic fetch-and-add operation with context";
  let description = [{
    Atomically fetches the value of a remote data object and adds a given
    value to it.

    Signature:
    ```c
    TYPE shmem_ctx_atomic_fetch_add(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $ctx `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($value) `,`"
                       " type($pe) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// AtomicAddOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicAddOp : OpenSHMEM_Op<"atomic_add", []> {
  let summary = "Atomic add operation (generic)";
  let description = [{
    Atomically adds a given value to the value of a remote data object.

    Signature:
    ```c
    void shmem_atomic_add(TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($dest) `,` type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicAddOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicAddOp : OpenSHMEM_Op<"ctx_atomic_add", []> {
  let summary = "Atomic add operation with context";
  let description = [{
    Atomically adds a given value to the value of a remote data object.

    Signature:
    ```c
    void shmem_ctx_atomic_add(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// AtomicFetchAndOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicFetchAndOp : OpenSHMEM_Op<"atomic_fetch_and", []> {
  let summary = "Atomic fetch-and operation (generic)";
  let description = [{
    Atomically fetches the value of a remote data object and performs a bitwise
    AND operation with a given value.

    Signature:
    ```c
    TYPE shmem_atomic_fetch_and(TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($dest) `,` type($value) `,` type($pe) `->`"
                       " type($result)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicFetchAndOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicFetchAndOp : OpenSHMEM_Op<"ctx_atomic_fetch_and", []> {
  let summary = "Atomic fetch-and operation with context";
  let description = [{
    Atomically fetches the value of a remote data object and performs a bitwise AND
    operation with a given value.

    Signature:
    ```c
    TYPE shmem_ctx_atomic_fetch_and(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $ctx `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($value) `,`"
                       " type($pe) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// AtomicFetchOrOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicFetchOrOp : OpenSHMEM_Op<"atomic_fetch_or", []> {
  let summary = "Atomic fetch-or operation (generic)";
  let description = [{
    Atomically fetches the value of a remote data object and performs a bitwise
    OR operation with a given value.

    Signature:
    ```c
    TYPE shmem_atomic_fetch_or(TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($dest) `,` type($value) `,` type($pe) `->`"
                       " type($result)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicFetchOrOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicFetchOrOp : OpenSHMEM_Op<"ctx_atomic_fetch_or", []> {
  let summary = "Atomic fetch-or operation with context";
  let description = [{
    Atomically fetches the value of a remote data object and performs a bitwise
    OR operation with a given value.
    
    Signature:
    ```c
    TYPE shmem_ctx_atomic_fetch_or(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $ctx `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($value) `,`"
                       " type($pe) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// AtomicOrOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicOrOp : OpenSHMEM_Op<"atomic_or", []> {
  let summary = "Atomic OR operation (generic)";
  let description = [{
    Atomically performs a bitwise OR operation with a given value.

    Signature:
    ```c
    void shmem_atomic_or(TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($dest) `,` type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicOrOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicOrOp : OpenSHMEM_Op<"ctx_atomic_or", []> {
  let summary = "Atomic OR operation with context";
  let description = [{
    Atomically performs a bitwise OR operation with a given value.

    Signature:
    ```c
    void shmem_ctx_atomic_or(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// AtomicFetchXorOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicFetchXorOp : OpenSHMEM_Op<"atomic_fetch_xor", []> {
  let summary = "Atomic fetch-xor operation (generic)";
  let description = [{
    Atomically fetches the value of a remote data object and performs a bitwise
    XOR operation with a given value.

    Signature:
    ```c
    TYPE shmem_atomic_fetch_xor(TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($dest) `,` type($value) `,` type($pe) `->`"
                       " type($result)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicFetchXorOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicFetchXorOp : OpenSHMEM_Op<"ctx_atomic_fetch_xor", []> {
  let summary = "Atomic fetch-xor operation with context";
  let description = [{
    Atomically fetches the value of a remote data object and performs a
    bitwise XOR operation with a given value.

    Signature:
    ```c
    TYPE shmem_ctx_atomic_fetch_xor(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $ctx `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($value) `,`"
                       " type($pe) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// AtomicXorOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicXorOp : OpenSHMEM_Op<"atomic_xor", []> {
  let summary = "Atomic XOR operation (generic)";
  let description = [{
    Atomically performs a bitwise XOR operation with a given value.

    Signature:
    ```c
    void shmem_atomic_xor(TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($dest) `,` type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicXorOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicXorOp : OpenSHMEM_Op<"ctx_atomic_xor", []> {
  let summary = "Atomic XOR operation with context";
  let description = [{
    Atomically performs a bitwise XOR operation with a given value.

    Signature:
    ```c
    void shmem_ctx_atomic_xor(shmem_ctx_t ctx, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// AtomicFetchNbiOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicFetchNbiOp : OpenSHMEM_Op<"atomic_fetch_nbi", []> {
  let summary = "Atomic fetch-nbi operation (generic)";
  let description = [{
    Perform a nonblocking fetch of a value atomically from a remote
    data object. This routine returns after initiating the operation.
    The operation is considered complete after a subsequent call to
    shmem_quiet. At the completion of shmem_quiet, contents of the source data
    object from PE has been fetched into fetch local data object.

    Signature:
    ```c
    void shmem_atomic_fetch_nbi(TYPE *fetch, const TYPE *source, int pe);
    ```
  }];

  let arguments = (ins 
    AnyMemRef:$fetch,
    SymmetricMemRef:$source,
    I32:$pe
  );

  let assemblyFormat = "`(` $fetch `,` $source `,` $pe `)` attr-dict `:` "
                       "type($fetch) `,` type($source) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicFetchNbiOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicFetchNbiOp : OpenSHMEM_Op<"ctx_atomic_fetch_nbi", []> {
  let summary = "Atomic fetch-nbi operation with context";
  let description = [{
    Perform a nonblocking fetch of a value atomically from a remote
    data object using a context. This routine returns after initiating
    the operation. The operation is considered complete after a subsequent
    call to shmem_quiet. At the completion of shmem_quiet, contents of the
    source data object from PE has been fetched into fetch local data object.

    Signature:
    ```c
    void shmem_ctx_atomic_fetch_nbi(shmem_ctx_t ctx, TYPE *fetch,
                                    const TYPE *source, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    AnyMemRef:$fetch,
    SymmetricMemRef:$source,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $fetch `,` $source `,` $pe `)` "
                       "attr-dict `:` "
                       "type($ctx) `,` type($fetch) `,` type($source) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// AtomicCompareSwapNbiOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicCompareSwapNbiOp :
  OpenSHMEM_Op<"atomic_compare_swap_nbi", []> {
  let summary = "Atomic compare-swap operation with nbi (generic)";
  let description = [{
    Atomically compares the value of a remote data object with a given value and
    swaps it with a new value if the comparison is successful.  

    Signature:
    ```c
    void shmem_atomic_compare_swap_nbi(TYPE *fetch, TYPE *dest, TYPE cond,
                                       TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    AnyType:$cond,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $fetch `,` $dest `,` $cond `,` $value `,` $pe `)` "
                       "attr-dict `:` "
                       "type($fetch) `,` type($dest) `,` type($cond) `,`"
                       " type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicCompareSwapNbiOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicCompareSwapNbiOp :
  OpenSHMEM_Op<"ctx_atomic_compare_swap_nbi", []> {
  let summary = "Atomic compare-swap operation with nbi (context-aware)";
  let description = [{
    Atomically compares the value of a remote data object with a given value and
    swaps it with a new value if the comparison is successful.

    Signature:
    ```c
    void shmem_ctx_atomic_compare_swap_nbi(shmem_ctx_t ctx,
                          TYPE *fetch, TYPE *dest, TYPE cond, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    AnyType:$cond,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $fetch `,` $dest `,` $cond `,` $value `,`"
                       " $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($fetch) `,` type($dest) `,`"
                       " type($cond) `,` type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// AtomicSwapNbiOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicSwapNbiOp : OpenSHMEM_Op<"atomic_swap_nbi", []> {
  let summary = "Atomic swap operation with nbi (generic)";
  let description = [{
    Atomically swaps the value of a remote data object with a given value.

    Signature:
    ```c
    void shmem_atomic_swap_nbi(TYPE *fetch, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $fetch `,` $dest `,` $value `,` $pe `)` "
                       "attr-dict `:` "
                       "type($fetch) `,` type($dest) `,` type($value) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicSwapNbiOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicSwapNbiOp : OpenSHMEM_Op<"ctx_atomic_swap_nbi", []> {
  let summary = "Atomic swap operation with nbi (context-aware)";
  let description = [{
    Atomically swaps the value of a remote data object with a given value.

    Signature:
    ```c
    void shmem_ctx_atomic_swap_nbi(shmem_ctx_t ctx, TYPE *fetch, TYPE *dest,
                                   TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $fetch `,` $dest `,` $value `,` $pe `)` "
                       "attr-dict `:` "
                       "type($ctx) `,` type($fetch) `,` type($dest) `,`"
                       " type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// AtomicFetchIncNbiOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicFetchIncNbiOp : OpenSHMEM_Op<"atomic_fetch_inc_nbi", []> {
  let summary = "Atomic fetch-inc operation with nbi (generic)";
  let description = [{
    Atomically fetches the value of a remote data object and increments it by 1.

    Signature:
    ```c
    void shmem_atomic_fetch_inc_nbi(TYPE *fetch, TYPE *dest, int pe);
    ```
  }];

  let arguments = (ins 
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    I32:$pe
  );

  let assemblyFormat = "`(` $fetch `,` $dest `,` $pe `)` attr-dict `:` "
                       "type($fetch) `,` type($dest) `,` type($pe)";
} 

//===----------------------------------------------------------------------===//
// CtxAtomicFetchIncNbiOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicFetchIncNbiOp :
  OpenSHMEM_Op<"ctx_atomic_fetch_inc_nbi", []> {
  let summary = "Atomic fetch-inc operation with nbi (context-aware)";
  let description = [{
    Atomically fetches the value of a remote data object and increments it by 1.

    Signature:
    ```c
    void shmem_ctx_atomic_fetch_inc_nbi(shmem_ctx_t ctx, TYPE *fetch,
                                        TYPE *dest, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $fetch `,` $dest `,` $pe `)` attr-dict `:` "
                       "type($ctx) `,` type($fetch) `,` type($dest) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// AtomicFetchAddNbiOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicFetchAddNbiOp : OpenSHMEM_Op<"atomic_fetch_add_nbi", []> {
  let summary = "Atomic fetch-add operation with nbi (generic)";
  let description = [{
    Atomically fetches the value of a remote data object and adds a given
    value to it.

    Signature:
    ```c
    void shmem_atomic_fetch_add_nbi(TYPE *fetch, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $fetch `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($fetch) `,` type($dest) `,` type($value) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicFetchAddNbiOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicFetchAddNbiOp :
  OpenSHMEM_Op<"ctx_atomic_fetch_add_nbi", []> {
  let summary = "Atomic fetch-add operation with nbi (context-aware)";
  let description = [{
    Atomically fetches the value of a remote data object and adds a given
    value to it.

    Signature:
    ```c
    void shmem_ctx_atomic_fetch_add_nbi(shmem_ctx_t ctx, TYPE *fetch,
                                        TYPE *dest, TYPE value, int pe);
    ```
  }];
  
  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $fetch `,` $dest `,` $value `,` $pe `)`"
                       " attr-dict `:` "
                       "type($ctx) `,` type($fetch) `,` type($dest) `,`"
                       " type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// AtomicFetchAndNbiOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicFetchAndNbiOp : OpenSHMEM_Op<"atomic_fetch_and_nbi", []> {
  let summary = "Atomic fetch-and operation with nbi (generic)";
  let description = [{
    Atomically fetches the value of a remote data object and performs a bitwise
    AND operation with a given value.

    Signature:
    ```c
    void shmem_atomic_fetch_and_nbi(TYPE *fetch, TYPE *dest, TYPE value, int pe);
    ```
  }];
  
  let arguments = (ins 
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $fetch `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($fetch) `,` type($dest) `,` type($value) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicFetchAndNbiOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicFetchAndNbiOp :
  OpenSHMEM_Op<"ctx_atomic_fetch_and_nbi", []> {
  let summary = "Atomic fetch-and operation with nbi (context-aware)";
  let description = [{
    Atomically fetches the value of a remote data object and performs a bitwise
    AND operation with a given value.

    Signature:
    ```c
    void shmem_ctx_atomic_fetch_and_nbi(shmem_ctx_t ctx, TYPE *fetch,
                                        TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $fetch `,` $dest `,` $value `,` $pe `)` "
                       "attr-dict `:` "
                       "type($ctx) `,` type($fetch) `,` type($dest) `,`"
                       " type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// AtomicFetchOrNbiOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicFetchOrNbiOp : OpenSHMEM_Op<"atomic_fetch_or_nbi", []> {
  let summary = "Atomic fetch-or operation with nbi (generic)";
  let description = [{
    Atomically fetches the value of a remote data object and performs a bitwise
    OR operation with a given value.

    Signature:
    ```c
    void shmem_atomic_fetch_or_nbi(TYPE *fetch, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $fetch `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($fetch) `,` type($dest) `,` type($value) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxAtomicFetchOrNbiOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicFetchOrNbiOp :
  OpenSHMEM_Op<"ctx_atomic_fetch_or_nbi", []> {
  let summary = "Atomic fetch-or operation with nbi (context-aware)";
  let description = [{
    Atomically fetches the value of a remote data object and performs a bitwise
    OR operation with a given value.

    Signature:
    ```c
    void shmem_ctx_atomic_fetch_or_nbi(shmem_ctx_t ctx, TYPE *fetch,
                                       TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $fetch `,` $dest `,` $value `,` $pe `)` "
  "attr-dict `:` "
                       "type($ctx) `,` type($fetch) `,` type($dest) `,`"
                       " type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// AtomicFetchXorNbiOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AtomicFetchXorNbiOp : OpenSHMEM_Op<"atomic_fetch_xor_nbi", []> {
  let summary = "Atomic fetch-xor operation with nbi (generic)";
  let description = [{
    Atomically fetches the value of a remote data object and performs a bitwise
    XOR operation with a given value.

    Signature:
    ```c
    void shmem_atomic_fetch_xor_nbi(TYPE *fetch, TYPE *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins 
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $fetch `,` $dest `,` $value `,` $pe `)` attr-dict `:` "
                       "type($fetch) `,` type($dest) `,` type($value) `,`"
                       " type($pe)";
}
//===----------------------------------------------------------------------===//
// CtxAtomicFetchXorNbiOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxAtomicFetchXorNbiOp :
  OpenSHMEM_Op<"ctx_atomic_fetch_xor_nbi", []> {
  let summary = "Atomic fetch-xor operation with nbi (context-aware)";
  let description = [{
    Atomically fetches the value of a remote data object and performs a bitwise
    XOR operation with a given value.

    Signature:
    ```c
    void shmem_ctx_atomic_fetch_xor_nbi(shmem_ctx_t ctx, TYPE *fetch,
                                        TYPE *dest, TYPE value, int pe);
    ```
  }];
  
  let arguments = (ins 
    OpenSHMEM_Ctx:$ctx,
    AnyMemRef:$fetch,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $fetch `,` $dest `,` $value `,` $pe `)` "
                       "attr-dict `:` "
                       "type($ctx) `,` type($fetch) `,` type($dest) `,`"
                       " type($value) `,` type($pe)";
}

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHEMATOMICS_TD 
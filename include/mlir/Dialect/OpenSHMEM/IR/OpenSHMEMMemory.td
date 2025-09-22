//===- OpenSHMEMMemory.td - OpenSHMEM memory operations ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHEMMMEMORY_TD
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHEMMMEMORY_TD

include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMBase.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// MallocOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_MallocOp : OpenSHMEM_Op<"malloc", []> {
  let summary = "Equivalent to the shmem_malloc C API";
  let description = [{
    Allocates symmetric memory that is accessible from all PEs. This is a
    collective operation that must be called by all PEs.

    Arguments:
    - size: Number of bytes to allocate

    Results:
    - ptr: Symmetric memory reference to the allocated memory

    ```c
    void *shmem_malloc(size_t size);
    ```
  }];

  let arguments = (ins Index:$size);
  
  let results = (outs 
    MemRefOf<[AnyTypeOf<[I8, I16, I32, I64, F16, F32, F64, F128]>]>:$ptr
  );
  
  let assemblyFormat = "`(` $size `)` attr-dict `:` type($size) `->` "
                       "type($ptr)";
}

//===----------------------------------------------------------------------===//
// FreeOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_FreeOp : OpenSHMEM_Op<"free", []> {
  let summary = "Equivalent to the shmem_free C API";
  let description = [{
    Frees symmetric memory that was previously allocated with shmem_malloc.
    This is a collective operation that must be called by all PEs.

    Arguments:
    - ptr: Symmetric memory address to free

    ```c
    void shmem_free(void *ptr);
    ```
  }];
  
  let arguments = (ins SymmetricMemRef:$ptr);
  let assemblyFormat = "`(` $ptr `)` attr-dict `:` type($ptr)";
}

//===----------------------------------------------------------------------===//
// ReallocOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_ReallocOp : OpenSHMEM_Op<"realloc", []> {
  let summary = "Equivalent to the shmem_realloc C API";
  let description = [{
    Resizes a block of symmetric memory. The contents of the block are
    unchanged up to the lesser of the new and old sizes. This is a
    collective operation on the world team.

    Arguments:
    - ptr: Symmetric memory address to resize
    - size: New size in bytes

    Results:
    - retval: Symmetric memory reference to the resized block

    ```c
    void *shmem_realloc(void *ptr, size_t size);
    ```
  }];

  let arguments = (ins
    SymmetricMemRef:$ptr,
    Index:$size
  );

  let results = (outs 
    MemRefOf<[AnyTypeOf<[I8, I16, I32, I64, F16, F32, F64, F128]>]>:$retval
  );

  let assemblyFormat = "`(` $ptr `,` $size `)` attr-dict `:` "
                       "type($ptr) `,` type($size) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// AlignOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_AlignOp : OpenSHMEM_Op<"align", []> {
  let summary = "Equivalent to the shmem_align C API";
  let description = [{
    Allocates symmetric memory with a specified byte alignment. The
    alignment must be a multiple of sizeof(void *) and a power of two.
    This is a collective operation on the world team.

    Arguments:
    - alignment: Byte alignment of the allocated block
    - size: Size in bytes of the block to allocate

    Results:
    - retval: Symmetric memory reference to the allocated block

    ```c
    void *shmem_align(size_t alignment, size_t size);
    ```
  }];

  let arguments = (ins
    Index:$alignment,
    Index:$size
  );

  let results = (outs 
    MemRefOf<[AnyTypeOf<[I8, I16, I32, I64, F16, F32, F64, F128]>]>:$retval
  );

  let assemblyFormat = "`(` $alignment `,` $size `)` attr-dict `:` "
                       "type($alignment) `,` type($size) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// CallocOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_CallocOp : OpenSHMEM_Op<"calloc", []> {
  let summary = "Equivalent to the shmem_calloc C API";
  let description = [{
    Allocates and zero-initializes symmetric memory for an array of objects.
    This is a collective operation on the world team.

    Arguments:
    - count: Number of objects to allocate
    - size: Size of each object in bytes

    Results:
    - retval: Symmetric memory reference to the allocated block

    ```c
    void *shmem_calloc(size_t count, size_t size);
    ```
  }];

  let arguments = (ins
    Index:$count,
    Index:$size
  );

  let results = (outs 
    MemRefOf<[AnyTypeOf<[I8, I16, I32, I64, F16, F32, F64, F128]>]>:$retval
  );

  let assemblyFormat = "`(` $count `,` $size `)` attr-dict `:` "
                       "type($count) `,` type($size) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// OffsetOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_OffsetOp : OpenSHMEM_Op<"offset", []> {
  let summary = "Element-wise offset into symmetric memory";
  let description = [{
    Returns a symmetric pointer advanced by `offsetElems` elements of the
    underlying element type. This is the IR equivalent of `&base[offsetElems]`
    in C and preserves symmetric provenance. The result can be used anywhere
    a memref with symmetric memory space is expected (e.g., RMA ops).
  }];

  let arguments = (ins
    SymmetricMemRef:$base,
    Index:$offsetElems
  );

  let results = (outs
    SymmetricMemRef:$result
  );

  let assemblyFormat = "`(` $base `,` $offsetElems `)` attr-dict `:` "
                       "type($base) `,` type($offsetElems) `->` type($result)";
}

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHEMMMEMORY_TD 
